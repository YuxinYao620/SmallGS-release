import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import imageio
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import tyro
import viser
from gs.datasets.colmap import Dataset, Parser, fov2focal, scale_aware_icp
from gs.datasets.traj import generate_interpolated_path
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from gs.utils import AppearanceOptModule, CameraOptModule, knn, rgb_to_sh, set_random_seed, matrix_to_quaternion, quaternion_to_matrix

from gs.utils_poses.align_traj import align_ate_c2b_use_a2b
from gs.utils_poses.comp_ate import compute_rpe, compute_ATE
from gs.vis_utils import plot_pose
from gsplat.strategy.ops import reset_opa

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy

from gsplat.distributed import cli
from gsplat.rendering import rasterization
from gsplat.strategy import DefaultStrategy

# from lietorch import SO3, SE3, Sim3, LieGroupParameter
from kornia.geometry.depth import depth_to_3d
import cv2
from PIL import Image

from plyfile import PlyData, PlyElement


import sys

import sys
import random

def get_scheduler(optimizer, policy, num_epochs_fix=None, num_epochs=None):
    if policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch - num_epochs_fix) / float(num_epochs - num_epochs_fix + 1)
            if lr_l <=  0.0:
                lr_l = 1.0 - max(0, num_epochs - num_epochs_fix) / float(num_epochs - num_epochs_fix + 1)
            return lr_l
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif policy == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, patience=2)
    else:
        return NotImplementedError('scheduler with {} is not implemented'.format(policy))
    return scheduler


def get_model_config(motion_model_name, elastic_loss_weight, homo_loss_weight,max_len,n_iter=1000):
    if motion_model_name in ['affinefield4d','transfield4d','se3field4d','scaledse3field4d']:
        model_opt = {
                'motion_model_name': motion_model_name,
                'device': "cuda",
                'motion_model_opt': {
                    'dpf_opt': {
                        'in_features':4, # 3D + t
                        'hidden_features': 128,
                        'n_hidden_layers': 8,
                        'max_lens': max_len,
                        },
                    'n_frames': max_len,
                    'homo_loss_weight': homo_loss_weight,
                    'elastic_loss_weight': elastic_loss_weight,
                    'n_iter': n_iter,
                    'lr':1e-4
                }
            }
    else:
        raise NotImplementedError

    return model_opt
def normalize_points(points):
    min_val = points.min(dim=0, keepdim=True).values
    max_val = points.max(dim=0, keepdim=True).values
    points_normalized = (points - min_val) / (max_val - min_val)
    points_scaled = points_normalized * 2 - 1
    return points_scaled, max_val-min_val, min_val

def create_splats_with_optimizers(
    parser: Parser,
    init_type: str = "sfm",
    init_num_pts: int = 100_000,
    init_extent: float = 3.0,
    init_opacity: float = 0.1,
    init_scale: float = 1.0,
    scene_scale: float = 1.0,
    sh_degree: int = 3,
    sparse_grad: bool = False,
    batch_size: int = 1,
    feature_dim: Optional[int] = None,
    device: str = "cuda",
    world_rank: int = 0,
    world_size: int = 1,
    predict_depth = None,
    predict_rgb = None,
) -> Tuple[torch.nn.ParameterDict, Dict[str, torch.optim.Optimizer]]:
    if init_type == "sfm":
        points = torch.from_numpy(parser.points).float()
        rgbs = torch.from_numpy(parser.points_rgb / 255.0).float()
    elif init_type == "random":
        points = init_extent * scene_scale * (torch.rand((init_num_pts, 3)) * 2 - 1)
        rgbs = torch.rand((init_num_pts, 3))
    elif init_type == "predict":
        points = predict_depth.float() * scene_scale
        rgbs = predict_rgb.float()
    else:
        raise ValueError("Please specify a correct init_type: sfm or random")
    # Initialize the GS size to be the average dist of the 3 nearest neighbors
    dist2_avg = (knn(points, 4)[:, 1:] ** 2).mean(dim=-1)  # [N,]
    dist_avg = torch.sqrt(dist2_avg)
    scales = torch.log(dist_avg * init_scale).unsqueeze(-1).repeat(1, 3)  # [N, 3]

    # Distribute the GSs to different ranks (also works for single rank)
    points = points[world_rank::world_size]
    rgbs = rgbs[world_rank::world_size]
    scales = scales[world_rank::world_size]

    N = points.shape[0]
    quats = torch.rand((N, 4))  # [N, 4]
    opacities = torch.logit(torch.full((N,), init_opacity))  # [N,]

    params = [
        # name, value, lr
        ("means", torch.nn.Parameter(points), 1.6e-4 * scene_scale),
        ("scales", torch.nn.Parameter(scales), 5e-3),
        ("quats", torch.nn.Parameter(quats), 1e-3),
        ("opacities", torch.nn.Parameter(opacities), 5e-2),
    ]

    if feature_dim is None:
        # color is SH coefficients.
        colors = torch.zeros((N, (sh_degree + 1) ** 2, 3))  # [N, K, 3]
        colors[:, 0, :] = rgb_to_sh(rgbs)
        params.append(("sh0", torch.nn.Parameter(colors[:, :1, :]), 2.5e-3))
        params.append(("shN", torch.nn.Parameter(colors[:, 1:, :]), 2.5e-3 / 20))
    else:
        # features will be used for appearance and view-dependent shading
        features = torch.rand(N, feature_dim)  # [N, feature_dim]
        params.append(("features", torch.nn.Parameter(features), 2.5e-3))
        colors = torch.logit(rgbs)  # [N, 3]
        params.append(("colors", torch.nn.Parameter(colors), 2.5e-3))

    splats = torch.nn.ParameterDict({n: v for n, v, _ in params}).to(device)
    # Scale learning rate based on batch size, reference:
    # https://www.cs.princeton.edu/~smalladi/blog/2024/01/22/SDEs-ScalingRules/
    # Note that this would not make the training exactly equivalent, see
    # https://arxiv.org/pdf/2402.18824v1
    BS = batch_size * world_size
    optimizers = {
        name: (torch.optim.SparseAdam if sparse_grad else torch.optim.Adam)(
            [{"params": splats[name], "lr": lr * math.sqrt(BS), "name": name}],
            eps=1e-15 / math.sqrt(BS),
            # TODO: check betas logic when BS is larger than 10 betas[0] will be zero.
            betas=(1 - BS * (1 - 0.9), 1 - BS * (1 - 0.999)) if BS < 10 else (0.9, 0.999),
        )
        for name, _, lr in params
    }
    return splats, optimizers

def create_splats_with_optimizers_sem(
    parser: Parser,
    init_type: str = "predict",
    init_num_pts: int = 100_000,
    init_extent: float = 3.0,
    init_opacity: float = 0.1,
    init_scale: float = 1.0,
    scene_scale: float = 1.0,
    sh_degree: int = 3,
    sparse_grad: bool = False,
    batch_size: int = 1,
    feature_dim: Optional[int] = None,
    device: str = "cuda",
    world_rank: int = 0,
    world_size: int = 1,
    predict_depth = None,
    predict_rgb = None,
    semantic_label = None,
) -> Tuple[torch.nn.ParameterDict, Dict[str, torch.optim.Optimizer]]:
    if init_type == "sfm":
        points = torch.from_numpy(parser.points).float()
        rgbs = torch.from_numpy(parser.points_rgb / 255.0).float()
    elif init_type == "random":
        points = init_extent * scene_scale * (torch.rand((init_num_pts, 3)) * 2 - 1)
        rgbs = torch.rand((init_num_pts, 3))
    elif init_type == "predict":
        points = predict_depth.float() * scene_scale
        rgbs = predict_rgb.float()
    else:
        raise ValueError("Please specify a correct init_type: sfm or random")
    # Initialize the GS size to be the average dist of the 3 nearest neighbors
    dist2_avg = (knn(points, 4)[:, 1:] ** 2).mean(dim=-1)  # [N,]
    dist_avg = torch.sqrt(dist2_avg)
    scales = torch.log(dist_avg * init_scale).unsqueeze(-1).repeat(1, 3)  # [N, 3]

    # Distribute the GSs to different ranks (also works for single rank)
    points = points[world_rank::world_size]
    rgbs = rgbs[world_rank::world_size]
    scales = scales[world_rank::world_size]

    N = points.shape[0]
    quats = torch.rand((N, 4))  # [N, 4]
    opacities = torch.logit(torch.full((N,), init_opacity))  # [N,]
    labels = semantic_label.float()

    params = [
        # name, value, lr
        ("means", torch.nn.Parameter(points), 1.6e-4 * scene_scale),
        ("scales", torch.nn.Parameter(scales), 5e-3),
        ("quats", torch.nn.Parameter(quats), 1e-3),
        ("opacities", torch.nn.Parameter(opacities), 5e-2),
        ("labels", torch.nn.Parameter(labels), 1e-2),
    ]

    if feature_dim is None:
        # color is SH coefficients.
        colors = torch.zeros((N, (sh_degree + 1) ** 2, 3))  # [N, K, 3]
        colors[:, 0, :] = rgb_to_sh(rgbs)
        params.append(("sh0", torch.nn.Parameter(colors[:, :1, :]), 2.5e-3))
        params.append(("shN", torch.nn.Parameter(colors[:, 1:, :]), 2.5e-3 / 20))
    else:
        # features will be used for appearance and view-dependent shading
        features = torch.rand(N, feature_dim)  # [N, feature_dim]
        params.append(("features", torch.nn.Parameter(features), 2.5e-3))
        colors = torch.logit(rgbs)  # [N, 3]
        params.append(("colors", torch.nn.Parameter(colors), 2.5e-3))
    
    splats = torch.nn.ParameterDict({n: v for n, v, _ in params}).to(device)
    # Scale learning rate based on batch size, reference:
    # https://www.cs.princeton.edu/~smalladi/blog/2024/01/22/SDEs-ScalingRules/
    # Note that this would not make the training exactly equivalent, see
    # https://arxiv.org/pdf/2402.18824v1
    BS = batch_size * world_size
    optimizers = {
        name: (torch.optim.SparseAdam if sparse_grad else torch.optim.Adam)(
            [{"params": splats[name], "lr": lr * math.sqrt(BS), "name": name}],
            eps=1e-15 / math.sqrt(BS),
            # TODO: check betas logic when BS is larger than 10 betas[0] will be zero.
            betas=(1 - BS * (1 - 0.9), 1 - BS * (1 - 0.999)) if BS < 10 else (0.9, 0.999),
        )
        for name, _, lr in params
    }
    return splats, optimizers

def create_optimizers(splats,batch_size,world_size,scene_scale, sparse_grad,feature_dim = 32, sh_degree = 3):
    BS = batch_size * world_size
    params = [
    # name, value, lr
    ("means", torch.nn.Parameter(splats['means']), 1.6e-4 * scene_scale),
    ("scales", torch.nn.Parameter(splats['scales']), 5e-3),
    ("quats", torch.nn.Parameter(splats['quats']), 1e-3),
    ("opacities", torch.nn.Parameter(splats['opacities']), 5e-2),
]   
    if 'labels' in splats.keys():
        params.append(("labels", torch.nn.Parameter(splats['labels']), 1e-2))
    if 'sh0' in splats.keys():
        params.append(("sh0", torch.nn.Parameter(splats['sh0']), 2.5e-3))
        params.append(("shN", torch.nn.Parameter(splats['shN']), 2.5e-3 / 20))
    else:
        features = torch.nn.Parameter(splats['features'])
        params.append(("features", features, 2.5e-3))
        colors = torch.nn.Parameter(splats['colors'])
        params.append(("colors", colors, 2.5e-3))
    # if feature_dim is None:
    #     # color is SH coefficients.
    #     # colors = torch.zeros((N, (sh_degree + 1) ** 2, 3))  # [N, K, 3]
    #     # colors[:, 0, :] = rgb_to_sh(rgbs)
    #     params.append(("sh0", torch.nn.Parameter(colors[:, :1, :]), 2.5e-3))
    #     params.append(("shN", torch.nn.Parameter(colors[:, 1:, :]), 2.5e-3 / 20))
    # else:
    #     # features will be used for appearance and view-dependent shading
    #     features = torch.rand(N, feature_dim)  # [N, feature_dim]
    #     params.append(("features", torch.nn.Parameter(features), 2.5e-3))
    #     colors = torch.logit(rgbs)  # [N, 3]
    #     params.append(("colors", torch.nn.Parameter(colors), 2.5e-3))

    optimizers = {
        name: (torch.optim.SparseAdam if sparse_grad else torch.optim.Adam)(
            [{"params": splats[name], "lr": lr * math.sqrt(BS), "name": name}],
            eps=1e-15 / math.sqrt(BS),
            # TODO: check betas logic when BS is larger than 10 betas[0] will be zero.
            betas=(1 - BS * (1 - 0.9), 1 - BS * (1 - 0.999)) if BS < 10 else (0.9, 0.999),
        )
        for name, _, lr in params
    }
    return optimizers


class Runner:
    """Engine for training and testing."""

    def __init__(
        self, local_rank: int, world_rank, world_size: int, opt
    ) -> None:
        set_random_seed(42 + local_rank)

        self.opt = opt
        self.world_rank = world_rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.device = f"cuda:{local_rank}"

        # Where to dump results.
        os.makedirs(opt.workspace, exist_ok=True)

        # Setup output directories.
        self.ckpt_dir = f"{opt.workspace}/ckpts"
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.stats_dir = f"{opt.workspace}/stats"
        os.makedirs(self.stats_dir, exist_ok=True)
        self.render_dir = f"{opt.workspace}/renders"
        self.intermediate_result = f"{opt.workspace}/intermediate_result"
        os.makedirs(self.intermediate_result, exist_ok=True)
        os.makedirs(self.render_dir, exist_ok=True)
        self.traj_dir = f"{opt.workspace}/traj"
        os.makedirs(self.traj_dir, exist_ok=True)
        self.depth_dir = f"{opt.workspace}/depth"
        os.makedirs(self.depth_dir, exist_ok=True)
        self.depth_gt_dir = f"{opt.workspace}/depth_gt"
        os.makedirs(self.depth_gt_dir, exist_ok=True)
        self.sem_dir = f"{opt.workspace}/sem"
        os.makedirs(self.sem_dir, exist_ok=True)
        self.point_cloud_dir = f"{opt.workspace}/point_cloud"
        os.makedirs(self.point_cloud_dir, exist_ok=True)
        self.debug_dir = f"{opt.workspace}/debug"
        os.makedirs(self.debug_dir, exist_ok=True)

        # Tensorboard
        self.writer = SummaryWriter(log_dir=f"{opt.workspace}/tb")
        # Load data: Training data should contain initial points and colors.
        self.parser = Parser(
            data_dir=opt.data_dir,
            factor=opt.data_factor,
            normalize=True,
            test_every=opt.test_every,
            type=opt.datatype,
            monst3r_dir = opt.monst3r_dir if hasattr(opt,'monst3r_dir') else None,
            monst3r_use_cam = not opt.gs_pose
        )
        self.trainset = Dataset(
            self.parser,
            split="train",
            patch_size=opt.patch_size,
            load_depths=opt.depth_loss,
            load_semantics=opt.semantics_opt,
            load_canny=opt.canny_opt,
            datatype = opt.datatype,
            # background_color = opt.background_color,
            dino = opt.dino,
            opt = opt
        )
        self.valset = Dataset(self.parser, split="val")
        self.scene_scale = self.parser.scene_scale * 1.1 * opt.global_scale
        print("Scene scale:", self.scene_scale)
        self.trainloader = None
        # Model
        # feature_dim = 32 if opt.app_opt else None
        feature_dim = opt.dino_dim if opt.dino else None
        assert (opt.app_opt or opt.dino) == (feature_dim is not None)

        if opt.pose_noise > 0.0:
            self.pose_perturb = CameraOptModule(len(self.trainset)).to(self.device)
            self.pose_perturb.random_init(opt.pose_noise)
            if world_size > 1:
                self.pose_perturb = DDP(self.pose_perturb)

        self.app_optimizers = []
        if opt.app_opt:
            self.app_module = AppearanceOptModule(
                len(self.trainset), feature_dim, opt.app_embed_dim, opt.sh_degree
            ).to(self.device)
            # initialize the last layer to be zero so that the initial output is zero.
            torch.nn.init.zeros_(self.app_module.color_head[-1].weight)
            torch.nn.init.zeros_(self.app_module.color_head[-1].bias)
            self.app_optimizers = [
                torch.optim.Adam(
                    self.app_module.embeds.parameters(),
                    lr=opt.app_opt_lr * math.sqrt(opt.batch_size) * 10.0,
                    weight_decay=opt.app_opt_reg,
                ),
                torch.optim.Adam(
                    self.app_module.color_head.parameters(),
                    lr=opt.app_opt_lr * math.sqrt(opt.batch_size),
                ),
            ]
            if world_size > 1:
                self.app_module = DDP(self.app_module)

        # Losses & Metrics.
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True).to(
            self.device
        )
        if self.opt.dino:
            # get all data for once to getting the dino maps
            for i in range(len(self.trainset)):
                self.trainset[i]
            

    def parameters(self):
        return list(self.model.splats.parameters())

    def rasterize_splats(
        self,
        camtoworlds,
        Ks: Tensor,
        width: int,
        height: int,
        **kwargs,
    ) -> Tuple[Tensor, Tensor, Dict]:
        
        means = self.splats["means"]
        # rasterization does normalization internally
        quats = self.splats["quats"]  # [N, 4]
        scales = torch.exp(self.splats["scales"])
        opacities = torch.sigmoid(self.splats["opacities"])  # [N,]

        image_ids = kwargs.pop("image_ids", None)
        if self.opt.app_opt:
            colors = self.app_module(
                features=self.splats["features"],
                embed_ids=image_ids,
                dirs=means[None, :, :] - camtoworlds[:, None, :3, 3],
                sh_degree=kwargs.pop("sh_degree", self.opt.sh_degree),
            )
            colors = colors + self.splats["colors"]
            colors = torch.sigmoid(colors)
        elif self.opt.dino:
            colors = self.splats["features"]
        else:
            colors = torch.cat([self.splats["sh0"], self.splats["shN"]], 1)  # [N, K, 3]

        rasterize_mode = "antialiased" if self.opt.antialiased else "classic"
        render_colors, render_alphas, info = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=torch.linalg.inv(camtoworlds),  # [C, 4, 4]
            Ks=Ks,  # [C, 3, 3]
            width=width,
            height=height,
            packed=self.opt.packed,
            absgrad=self.opt.absgrad,
            sparse_grad=self.opt.sparse_grad,
            rasterize_mode=rasterize_mode,
            distributed=self.world_size > 1,
            **kwargs,
        )
        return render_colors, render_alphas, info
    
    
    def get_expon_lr_func(self, lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000):
        """
        Copied from Plenoxels

        Continuous learning rate decay function. Adapted from JaxNeRF
        The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
        is log-linearly interpolated elsewhere (equivalent to exponential decay).
        If lr_delay_steps>0 then the learning rate will be scaled by some smooth
        function of lr_delay_mult, such that the initial learning rate is
        lr_init*lr_delay_mult at the beginning of optimization but will be eased back
        to the normal learning rate when steps>lr_delay_steps.
        :param conf: config subtree 'lr' or similar
        :param max_steps: int, the number of steps during optimization.
        :return HoF which takes step as input
        """

        def helper(step):
            if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
                # Disable this parameter
                return 0.0
            if lr_delay_steps > 0:
                # A kind of reverse cosine decay.
                delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                    0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
                )
            else:
                delay_rate = 1.0
            t = np.clip(step / max_steps, 0, 1)
            log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
            return delay_rate * log_lerp

        return helper

    #https://github.com/nerfstudio-project/gsplat/issues/234
    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self.splats["sh0"].shape[1]*self.splats["sh0"].shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self.splats["shN"].shape[1]*self.splats["shN"].shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self.splats["scales"].shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self.splats["quats"].shape[1]):
            l.append('rot_{}'.format(i))
        return l

    # https://github.com/nerfstudio-project/gsplat/issues/234
    @torch.no_grad()
    def save_ply(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)

        xyz = self.splats["means"].detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self.splats["sh0"].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self.splats["shN"].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self.splats["opacities"].detach().unsqueeze(-1).cpu().numpy()
        scale = self.splats["scales"].detach().cpu().numpy()
        rotation = self.splats["quats"].detach().cpu().numpy()
        

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    
    def calculate_speed_smoothness(self,camera_positions):
        speed_smoothness_loss = 0.0
        for i in range(1, len(camera_positions) - 1):
            prev_speed = camera_positions[i] - camera_positions[i - 1]
            next_speed = camera_positions[i + 1] - camera_positions[i]
            speed_smoothness_loss += torch.norm(next_speed - prev_speed)
        return speed_smoothness_loss
    
    def monst3r_camera_smoothness(self,camera_pose):
        pose_loss = 0.0
        for i in range(1, len(camera_pose) - 1):
            RT1 = camera_pose[i - 1]
            RT2 = camera_pose[i]
            relative_RT = torch.matmul(torch.inverse(RT1), RT2)
            rotation_diff = relative_RT[:3, :3]
            translation_diff = relative_RT[:3, 3]

            # Frobenius norm for rotation difference
            rotation_loss = torch.norm(rotation_diff - (torch.eye(3, device=RT1.device)), dim=(0, 1))

            # L2 norm for translation difference
            translation_loss = torch.norm(translation_diff, dim=0)

            # Combined loss (one can weigh these differently if needed)
            pose_loss += rotation_loss + translation_loss
        return pose_loss
        
    def train_block(self, camtoworlds, Ks, width, height, sh_degree_to_use, image_ids, pixels, schedulers, 
                    camera_scheduler, camera_opt=False, points=None, depthmap_gt=None,eval_final=False,masks=None, dino = None):
        if eval_final:
            max_steps_local = 1000
        elif not camera_opt:
            max_steps_local = 1000
        else:   
            max_steps_local = self.opt.max_steps
        pbar = tqdm.tqdm(range(0, max_steps_local))
        for step in pbar:
                
                
            if camera_opt:
                camtoworlds_transformed = self.pose_adjust(camtoworlds, image_ids)
                if step % 100 == 0:
                # save the intermediate result
                    # write camera pose 
                    from pathlib import Path
                    from scipy.spatial.transform import Rotation
                    tostr = lambda x: " ".join([str(i) for i in x])
                    refined_pose = torch.cat([torch.Tensor(pose).unsqueeze(0) for pose in camtoworlds_transformed], dim=0)
                    gs_pose_R = refined_pose[:,:3, :3]
                    gs_pose_R = gs_pose_R.detach().cpu().numpy()
                    quat = Rotation.from_matrix(gs_pose_R)
                    quat = quat.as_quat()
                    quat = torch.Tensor(quat).to(self.device)
                    # translation
                    t = refined_pose[:,:3, 3]
                    # set the camera pose
                    camera_pose = torch.cat([quat,t], dim=1).to(self.device)
                    with Path(f"{self.opt.workspace}/traj/refined_pose_{step}.txt").open("w") as f:
                        for i in range(len(camera_pose)):
                            ith_pose = camera_pose[i]
                            f.write(
                                f"{i} {tostr(ith_pose[-3::].detach().cpu().numpy())} {tostr(ith_pose[[0,1,2,3]].detach().cpu().numpy())}\n"
                            )

            renders , alphas, info = self.rasterize_splats(
                camtoworlds= camtoworlds_transformed if camera_opt else camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=sh_degree_to_use,
                near_plane=self.opt.near_plane,
                far_plane=self.opt.far_plane,
                image_ids=image_ids,
                render_mode="RGB+ED" if self.opt.depth_loss else "RGB",
            )
            if camera_opt and (step % 100 == 0) :
                # store the render images 
                if self.opt.dino:
                    temp_renders = renders[:,:,:,:3]
                    self.store_render_image(temp_renders, f'{image_ids[0]}_render_{step}')
                else:
                    self.store_render_image(renders, f'{image_ids[0]}_render_{step}')

            if renders.shape[-1] == 4:
                colors, depths_0 = renders[..., 0:3], renders[..., 3::]
            elif renders.shape[-1] > 4:
                colors, depths_0 = renders[..., 0:-1], renders[..., -1::]
            else:
                colors, depths_0 = renders, None  
            
            if not camera_opt:
                self.strategy.step_pre_backward(
                    params=self.splats,
                    optimizers=self.optimizers,
                    state=self.strategy_state,
                    step=step,
                    info=info,
                )

            if masks is not None:
                colors = colors * masks
                pixels = pixels * masks
                depths_0 = depths_0 * masks
            # breakpoint()
            if self.opt.dino:
                if masks is not None:
                    colors = colors * masks
                    dino = dino * masks
                
                l1loss = F.l1_loss(colors, dino.detach())
                ssimloss = 1.0 - self.ssim(
                    dino.detach().permute(0, 3, 1, 2), colors.permute(0, 3, 1, 2)
                )
                
            else:
                l1loss = F.l1_loss(colors, pixels)
                ssimloss = 1.0 - self.ssim(
                    pixels.permute(0, 3, 1, 2), colors.permute(0, 3, 1, 2)
                )

            loss = l1loss * (1.0 - self.opt.ssim_lambda) + ssimloss * self.opt.ssim_lambda

            if self.opt.camera_smoothness_loss and camera_opt:
                if self.opt.monst3r_camera:
                    smoothness_loss = self.monst3r_camera_smoothness(camtoworlds_transformed)
                else:
                    smoothness_loss = self.calculate_speed_smoothness(camtoworlds_transformed)
                
                loss += (1- step/max_steps_local)*self.opt.camera_smoothness_lambda * smoothness_loss

            if self.opt.identity_prior and camera_opt:
                identity_loss = F.mse_loss(camtoworlds_transformed[:,:3,:3], torch.eye(3).unsqueeze(0).repeat(camtoworlds_transformed.shape[0],1,1).to(self.device))
                loss += self.opt.identity_prior_lambda * identity_loss

            desc = f"loss={loss.item():.3f}| " f"sh degree={sh_degree_to_use}| "
                
            pbar.set_description(desc)
            loss.backward()
            if step % 100 == 0 and camera_opt:
                with open(f"{self.debug_dir}/loss.txt", "a") as f:
                    f.write(f"render cam opt :{step} {loss.item()}\n")
            if not camera_opt:
                self.strategy.step_post_backward(
                    params=self.splats,
                    optimizers=self.optimizers,
                    state=self.strategy_state,
                    step=step,
                    info=info,
                    packed=self.opt.packed,
                )

            # Turn Gradients into Sparse Tensor before running optimizer
            if self.opt.sparse_grad:
                assert self.opt.packed, "Sparse gradients only work with packed mode."
                gaussian_ids = info["gaussian_ids"]
                for k in self.splats.keys():
                    grad = self.splats[k].grad
                    if grad is None or grad.is_sparse:
                        continue
                    self.splats[k].grad = torch.sparse_coo_tensor(
                        indices=gaussian_ids[None],  # [1, nnz]
                        values=grad[gaussian_ids],  # [nnz, ...]
                        size=self.splats[k].size(),  # [N, ...]
                        is_coalesced=len(Ks) == 1,
                    )


            # optimize
            if not camera_opt:
                for optimizer in self.optimizers.values():
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                for optimizer in self.app_optimizers:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                for scheduler in schedulers:
                    scheduler.step()
            else:
                # print('camera optimization')
                for optimizer in self.pose_optimizers:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                for scheduler in camera_scheduler:
                    scheduler.step()
            camtoworlds = camtoworlds.detach()
            

        return renders, alphas, info, loss

    def store_render_image(self, renders, path,renders2 = None):
        if renders.shape[-1] == 4:
            colors, depths_0 = renders[..., 0:3], renders[..., 3:4]
        else:
            colors, depths_0 = renders, None
        if renders2 is not None:
            if renders2.shape[-1] == 4:
                colors2, depths_02 = renders2[..., 0:3], renders2[..., 3:4]
            else:
                colors2, depths_02 = renders2, None
            colors = torch.cat([colors, colors2], dim=2)
        else:
            colors = torch.clamp(colors, 0.0, 1.0)
        torch.cuda.synchronize()

        canvas = colors.squeeze(0).cpu().detach().numpy()
        if canvas.shape[0] >1:
            for i in range(canvas.shape[0]):
                imageio.imwrite(
                    f"{self.intermediate_result}/{path}_{i}.png",
                    (canvas[i] * 255).astype(np.uint8),
                )
        else:
            imageio.imwrite(
                f"{self.intermediate_result}/{path}.png",
                (canvas* 255).astype(np.uint8),
            )

    def train(self, monst3r_pointmap = None, monst3r_pointcolor = None):
        opt = self.opt
        device = self.device
        world_rank = self.world_rank
        world_size = self.world_size

        # Dump opt.
        if world_rank == 0:
            with open(f"{opt.workspace}/opt.json", "w") as f:
                json.dump(vars(opt), f)

        max_steps = opt.max_steps
        init_step = 0


        # Training loop.
        global_tic = time.time()
        pbar = tqdm.tqdm(range(0, 1))
        # devide 
        start_frame = 1
        end_frame = len(self.trainset)
        global_cam_poses_est = []
        global_cam_poses_gt = []

        prev_cam_pose = self.trainset[0]['camtoworld'].to(device)

        for current_frame in range(start_frame, end_frame, opt.batch_size):
            # try load camera pose, if not empty, then use it 
            block_end = min(current_frame + opt.batch_size  , end_frame)
            print('optimizing from {} to {}'.format(current_frame, block_end-1))
            if opt.pose_ckpt is not None and os.path.exists(opt.pose_ckpt):
                pose_ckpt=torch.load(opt.pose_ckpt)
                global_cam_poses_est = pose_ckpt['global_cam_poses_est']
                global_cam_poses_gt = pose_ckpt['global_cam_poses_gt']
                print('load camera pose from checkpoint')
                break
            
            data_block = []
            
            #prepare data block
            print("block_end", block_end)
            for i in range(current_frame-1, block_end):
                data_block.append(self.trainset[i])
        
            # reinitalize the optimizer and gsplats
            if monst3r_pointcolor is not None:
                depth = monst3r_pointmap[current_frame-1].reshape(-1, 3).to(device)
                rgb = monst3r_pointcolor[current_frame-1].reshape(-1, 3).to(device)
            else:
                depth = data_block[0]['depth'].to(device)
                rgb = data_block[0]['image'].reshape(-1, 3).to(device) / 255.0 

            if opt.semantics_opt:
                semantics_masks = torch.stack([data["mask"] for data in data_block]).to(device)
                depth = depth[semantics_masks[0].view(-1)]
                rgb = rgb[semantics_masks[0].view(-1)]

            if opt.app_opt:
                feature_dim = 32
            elif opt.dino:
                feature_dim = opt.dino_dim
            else:
                feature_dim = None
            self.splats, self.optimizers = create_splats_with_optimizers(
                self.parser,
                init_type=opt.init_type,
                init_num_pts=opt.init_num_pts,
                init_extent=opt.init_extent,
                init_opacity=opt.init_opa,
                init_scale=opt.init_scale,
                scene_scale=self.scene_scale,
                sh_degree=opt.sh_degree if not opt.dino else None,
                sparse_grad=opt.sparse_grad,
                batch_size=opt.batch_size,
                feature_dim = feature_dim,
                device=self.device,
                world_rank=world_rank,
                world_size=world_size,
                predict_depth = depth,
                predict_rgb = rgb
            )
            self.strategy = DefaultStrategy(
                verbose=False,
                # scene_scale=self.scene_scale,
                prune_opa=opt.prune_opa,
                grow_grad2d=opt.grow_grad2d,
                grow_scale3d=opt.grow_scale3d,
                prune_scale3d=opt.prune_scale3d,
                refine_start_iter=opt.refine_start_iter,
                refine_stop_iter=opt.refine_stop_iter,
                reset_every=opt.reset_every,
                refine_every=opt.refine_every,
                absgrad=opt.absgrad,
                revised_opacity=opt.revised_opacity,
            )
            self.strategy_state = self.strategy.initialize_state()
            schedulers = [
                # means has a learning rate schedule, that end at 0.01 of the initial value
                torch.optim.lr_scheduler.ExponentialLR(
                    self.optimizers["means"], gamma=0.01 ** (1.0 / max_steps)
                ),
            ]

            self.pose_adjust = CameraOptModule(len(data_block)-1).to(self.device)
            self.pose_adjust.zero_init()
            self.pose_optimizers = [
                torch.optim.Adam(
                    self.pose_adjust.parameters(),
                    lr=opt.pose_opt_lr,
                    weight_decay=opt.pose_opt_reg,
                )
            ]
            camera_schedulers = []
            for pose_optimizer in self.pose_optimizers:
                camera_schedulers.append(torch.optim.lr_scheduler.ExponentialLR(
                    pose_optimizer, 
                    gamma=0.01 ** (0.5/ max_steps)
                    )
                )


            # reinit end 
            camtoworlds = torch.stack([data["camtoworld"] for data in data_block]).to(device)  # [b, 4, 4]
            camtoworlds_gt = torch.stack([data["camtoworld_gt"] for data in data_block]).to(device)  # [b, 4, 4]
            Ks = torch.stack([data["K"] for data in data_block]).to(device)
            # breakpoint()   
            if opt.semantics_opt:
                pixels = torch.stack([data["image"]*data['mask'] for data in data_block]).to(device) / 255.0  # [b, H, W, 3]
                masks = torch.stack([data["mask"] for data in data_block]).to(device)
                if opt.dino:
                    dino_gt = torch.stack([data["dino_embeddings"]*data['mask'] for data in data_block]).to(device)
            else:
                pixels = torch.stack([data["image"] for data in data_block]).to(device) / 255.0
                if opt.dino:
                    dino_gt = torch.stack([data["dino_embeddings"] for data in data_block]).to(device)                
                masks = None


            num_train_rays_per_step = (
                pixels.shape[0] * pixels.shape[1] * pixels.shape[2]
            )
            image_ids = torch.Tensor([data["image_id"] for data in data_block]).int().to(device)
            depthmap_gt = torch.stack([data["depth_map"] for data in data_block]).to(device)  # [b, M]
            if opt.pose_noise:
                camtoworlds = self.pose_perturb(camtoworlds, image_ids)

            if opt.save_sem_mask:
                for ind, pixel in enumerate(pixels):

                    # save semantic mask
                    if opt.semantics_opt:
                        img_mask = semantics_masks[ind].long()
                        pixels[ind] = pixel*img_mask
                        mask = Image.fromarray(img_mask.view(pixel.shape[0], pixel.shape[1]).cpu().numpy().astype(np.uint8)*255)
                        # save together with the image
                        save_pixel = pixel.cpu().numpy()*255
                        canvas = np.concatenate([save_pixel,(img_mask.repeat(1,1,3).cpu().numpy().astype(np.uint8)*255)],axis=1)
                        mask.save(f"{self.sem_dir}/{image_ids[ind]}.png")
                    depth_gt_save = Image.fromarray((depthmap_gt[ind].cpu().numpy()*255).astype(np.uint8))
                    depth_gt_save.save(f"{self.depth_gt_dir}/{image_ids[ind]}.png")
            height, width = pixels.shape[1:3]
            
            # sh schedule
            sh_degree_to_use = min(opt.max_steps // opt.sh_degree_interval, opt.sh_degree) 
            if sh_degree_to_use <1:
                sh_degree_to_use = opt.sh_degree
            if self.opt.dino:
                sh_degree_to_use = None
            # train first frame
            renders_0, alphas_0, info_0, loss = self.train_block(camtoworlds[[0]], Ks[[0]], width, height, sh_degree_to_use, image_ids[[0]], pixels[[0]],
                                                                    schedulers,camera_schedulers,camera_opt = False,depthmap_gt=depthmap_gt[[0]],masks=None,
                                                                    dino = dino_gt[[0]] if opt.dino else None) 
            # fix the first frame and train the rest
            if current_frame != end_frame:
                if self.opt.semantics_opt:
                    renders, alphas, info, loss = self.train_block(camtoworlds[1:], Ks[1:], width, height, sh_degree_to_use, image_ids[1:], pixels[1:],
                                                                schedulers,camera_schedulers,camera_opt = True,depthmap_gt=depthmap_gt[1:],
                                                                masks=masks[1:], dino = dino_gt[1:] if opt.dino else None)
                else:
                    renders, alphas, info, loss = self.train_block(camtoworlds[1:], Ks[1:], width, height, sh_degree_to_use, image_ids[1:], pixels[1:],
                                                                schedulers,camera_schedulers,camera_opt = True,depthmap_gt=depthmap_gt[1:],
                                                                masks=None, dino = dino_gt[1:] if opt.dino else None)

                # current camera pose with optimization
                est_cam_pose = self.pose_adjust(camtoworlds[1:], image_ids[1:])
                # full estimated camera pose
                # apply first frame's pose to the rest
                if current_frame == start_frame:
                    global_cam_poses_est += [camtoworlds[0].cpu().detach().numpy()]
                    global_cam_poses_gt += [camtoworlds_gt[0].cpu().detach().numpy()]

                transfomed_pose = est_cam_pose.detach() @ prev_cam_pose.detach()
                # est_cam_pose = transfomed_pose
                prev_cam_pose = transfomed_pose[-1]
                global_cam_poses_est += [ pose.cpu().numpy() for pose in transfomed_pose ]
                global_cam_poses_gt += [ pose.cpu().numpy() for pose in camtoworlds_gt[1:]]
            else:
                est_cam_pose = camtoworlds
                global_cam_poses_est += [camtoworlds.cpu().detach().numpy()]
                global_cam_poses_gt += [camtoworlds_gt.cpu().detach().numpy()]
            est_cam_pose = torch.stack([data_block[0]['camtoworld'].to(device)]+[cam_pose.to(device) for cam_pose in est_cam_pose]).to(device)

            self.eval_all(current_frame, data_block, est_cam_pose)

            self.plot_traj(np.array(global_cam_poses_est), np.array(global_cam_poses_gt), start_frame, block_end)
            step = current_frame
            data = {"step": step, "splats": self.splats.state_dict(),"start_frame": current_frame,"end_frame": block_end}
            if opt.pose_opt:
                if world_size > 1:
                    data["pose_adjust"] = self.pose_adjust.module.state_dict()
                else:
                    data["pose_adjust"] = self.pose_adjust.state_dict()
            if opt.app_opt:
                if world_size > 1:
                    data["app_module"] = self.app_module.module.state_dict()
                else:
                    data["app_module"] = self.app_module.state_dict()
            torch.save(
                data, f"{self.ckpt_dir}/ckpt_{current_frame}_rank{self.world_rank}.pt"
            )

        if block_end >= end_frame or not os.path.exists(opt.pose_ckpt):
        # if False:
            print("block_end", block_end)
            print('end_frame', end_frame)
            endtime_camera= time.time() 
            print("------saving checkpoint------")
            mem = torch.cuda.max_memory_allocated() / 1024**3
            stats = {
                "mem": mem,
                "ellipse_time": time.time() - global_tic,
                "num_GS": len(self.splats["means"]),
            }
            step = current_frame
            print("Step: ", step, stats)
            with open(
                f"{self.stats_dir}/train_step{step:04d}_rank{self.world_rank}.json",
                "w",
            ) as f:
                json.dump(stats, f)
            data = {"step": step, "splats": self.splats.state_dict(),"start_frame": current_frame,"end_frame": block_end}
            if opt.pose_opt:
                if world_size > 1:
                    data["pose_adjust"] = self.pose_adjust.module.state_dict()
                else:
                    data["pose_adjust"] = self.pose_adjust.state_dict()
            if opt.app_opt:
                if world_size > 1:
                    data["app_module"] = self.app_module.module.state_dict()
                else:
                    data["app_module"] = self.app_module.state_dict()
            torch.save(
                data, f"{self.ckpt_dir}/ckpt_{current_frame}_to{block_end}_rank{self.world_rank}.pt"
            )

            global_camera_poses = {"global_cam_poses_est": global_cam_poses_est, "global_cam_poses_gt": global_cam_poses_gt}
            # save ckpt 
            torch.save(
                global_camera_poses, f"{self.traj_dir}/gs_camera_poses.pt"
            )

            torch.cuda.empty_cache()
            return global_camera_poses

    def create_strategy(self):
        opt = self.opt
        strategy = DefaultStrategy(
                verbose=False,
                # scene_scale=self.scene_scale,
                prune_opa=opt.prune_opa,
                grow_grad2d=opt.grow_grad2d,
                grow_scale3d=opt.grow_scale3d,
                prune_scale3d=opt.prune_scale3d,
                refine_start_iter=opt.refine_start_iter,
                refine_stop_iter=opt.refine_stop_iter,
                reset_every=opt.reset_every,
                refine_every=opt.refine_every,
                absgrad=opt.absgrad,
                revised_opacity=opt.revised_opacity,
        )
        self.strategy = strategy
        if not opt.test_gs:
            if os.path.exists(f"{self.ckpt_dir}/strategy_state.pt"):
                strategy_state = torch.load(f"{self.ckpt_dir}/strategy_state.pt")
                self.strategy_state = strategy_state['strategy_state']
            else:
                self.strategy_state = self.strategy.initialize_state()
        else:
            self.strategy_state = self.strategy.initialize_state()

    def select_transparent_pixels(self,image, transparent_color=[0, 0, 0]):
        """
        Select pixels that are transparent (specified color, e.g., black) from an RGB image tensor.
        
        Args:
            image (torch.Tensor): A tensor of shape (H, W, 3) representing an RGB image.
            transparent_color (list or tuple): The RGB color to treat as transparent (default is [0, 0, 0] for black).
        
        Returns:
            torch.Tensor: A mask tensor of shape (H, W) where 1 represents the transparent pixels 
                        (matching the specified color) and 0 represents non-transparent pixels.
        """
        # Convert transparent_color to a tensor for comparison
        transparent_color_tensor = torch.tensor(transparent_color, dtype=torch.float32).to(self.device)
        
        # Compare each pixel's RGB value to the transparent color
        mask = torch.all(image == transparent_color_tensor, dim=-1)
        
        return mask
    
    def render(self, data,sh_degree_to_use): # work with dataloader
        opt = self.opt
        if len(data['camtoworld'].shape) == 2:
            camtoworlds = torch.Tensor(data['camtoworld']).unsqueeze(0).to(self.device)
            tidx = torch.Tensor([data['image_id']]).int().squeeze().to(self.device)
            Ks = data['K'].unsqueeze(0).to(self.device)
            pixels = data['image'].unsqueeze(0).to(self.device) / 255.0
            image_ids = torch.Tensor([data["image_id"]]).int().to(self.device)
            depth = data['depth'].unsqueeze(0).to(self.device)
            rgb = data['image'].reshape(-1, 3).to(self.device) / 255.0
            height, width = pixels.shape[1:3]
            depthmap_gt = data['depth_map'].unsqueeze(0).to(self.device)  
        else:
            camtoworlds = torch.Tensor(data['camtoworld']).to(self.device)
            tidx = torch.Tensor([data['image_id']]).int().squeeze().to(self.device)
            Ks = data['K'].to(self.device)
            pixels = data['image'].to(self.device) / 255.0
            image_ids = torch.Tensor([data["image_id"]]).int().to(self.device)
            depth = data['depth'].to(self.device)
            rgb = data['image'].reshape(-1, 3).to(self.device) / 255.0
            height, width = pixels.shape[1:3]
            # if opt.depth_loss:
            depthmap_gt = data['depth_map'].to(self.device)

        renders, alphas, info = self.rasterize_splats_sem(
            camtoworlds= camtoworlds,
            Ks=Ks,
            width=width,
            height=height,
            sh_degree=sh_degree_to_use,
            near_plane=opt.near_plane,
            far_plane=opt.far_plane,
            image_ids=image_ids,
            render_mode="RGB+ED" if opt.depth_loss else "RGB"
        )

        if renders.shape[-1] == 4:
            colors, depths = renders[..., 0:3], renders[..., 3::]
        else:
            colors, depths = renders, None
        
        return colors, depths, renders, info
    

    def align_pose(self, pose1, pose2):
        mtx1 = np.array(pose1, dtype=np.double, copy=True)
        mtx2 = np.array(pose2, dtype=np.double, copy=True)

        if mtx1.ndim != 2 or mtx2.ndim != 2:
            raise ValueError("Input matrices must be two-dimensional")
        if mtx1.shape != mtx2.shape:
            raise ValueError("Input matrices must be of same shape")
        if mtx1.size == 0:
            raise ValueError("Input matrices must be >0 rows and >0 cols")

        # translate all the data to the origin
        mtx1 -= np.mean(mtx1, 0)
        mtx2 -= np.mean(mtx2, 0)

        norm1 = np.linalg.norm(mtx1)
        norm2 = np.linalg.norm(mtx2)

        if norm1 == 0 or norm2 == 0:
            raise ValueError("Input matrices must contain >1 unique points")

        # change scaling of data (in rows) such that trace(mtx*mtx') = 1
        mtx1 /= norm1
        mtx2 /= norm2

        # transform mtx2 to minimize disparity
        R, s = scipy.linalg.orthogonal_procrustes(mtx1, mtx2)
        mtx2 = mtx2 * s

        return mtx1, mtx2, R


    def plot_traj(self, global_cam_poses_est, global_cam_poses_gt, start_frame, end_frame):
        # plot trajectory
        # global_cam_poses_est_aligned = global_cam_poses_est @ global_cam_poses_gt[0]
        global_cam_poses_est_aligned = global_cam_poses_est
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        est_x, est_y, est_z = [], [], []
        gt_x, gt_y, gt_z = [], [], []

        for i in range(0, end_frame-start_frame):
            cam_pose_est = global_cam_poses_est_aligned[i]
            # cam_pose_gt = global_cam_poses_gt[i]

            est_x.append(cam_pose_est[0, 3])
            est_y.append(cam_pose_est[1, 3])
            est_z.append(cam_pose_est[2, 3])

            # gt_x.append(cam_pose_gt[0, 3])
            # gt_y.append(cam_pose_gt[1, 3])
            # gt_z.append(cam_pose_gt[2, 3])
            # ax.scatter(cam_pose_est[0, 3], cam_pose_est[1, 3], cam_pose_est[2, 3], c='r', marker='o', rasterized = True)
            # ax.scatter(cam_pose_gt[0, 3], cam_pose_gt[1, 3], cam_pose_gt[2, 3], c='b', marker='x', rasterized = True)
        for i in range(0, len(global_cam_poses_gt)):
            cam_pose_gt = global_cam_poses_gt[i]
            gt_x.append(cam_pose_gt[0, 3])
            gt_y.append(cam_pose_gt[1, 3])
            gt_z.append(cam_pose_gt[2, 3])
        ax.plot(est_x, est_y, est_z, c='r', label='est')
        ax.plot(gt_x, gt_y, gt_z, c='b', label='gt')
        ax.legend(['est', 'gt'])
        plt.savefig(f"{self.traj_dir}/traj_{start_frame}_to_{end_frame}.png")
        # plt.savefig(f"{self.traj_dir}/traj_{start_frame}_to_{end_frame}.pdf", dpi = 200, bbox_inches='tight')


    @torch.no_grad()
    def eval_all(self, current_frame, data_block, est_cam_pose):
        """Entry for evaluation."""
        print("Running evaluation...")
        opt = self.opt
        device = self.device
        world_rank = self.world_rank
        world_size = self.world_size
        # camtoworlds_est = torch.stack([data_block[0]['camtoworld'].to(device)]+[cam_pose.to(device) for cam_pose in est_cam_pose]).to(device)
        camtoworlds_est = est_cam_pose
        for i,data in enumerate(data_block):
            ellipse_time = 0
            metrics = {"psnr": [], "ssim": [], "lpips": []}
            camtoworlds = camtoworlds_est[i]
            Ks = data["K"].to(device)
            pixels = data["image"].to(device) / 255.0
            if len(camtoworlds.shape) == 2:
                camtoworlds = camtoworlds.unsqueeze(0)
                Ks = Ks.unsqueeze(0)
                pixels = pixels.unsqueeze(0)

            height, width = pixels.shape[1:3]
            torch.cuda.synchronize()
            tidx = torch.Tensor([data["image_id"]]).int().squeeze().to(device)
            if 'labels' in self.splats.keys():
                dynamic_points = self.splats['means'][self.splats['labels'] == 0]
                points, scaled_factor, min_value = normalize_points(dynamic_points)

                # overwrite the corresponding points
                renders, alphas, info = self.rasterize_splats_sem(
                    camtoworlds= camtoworlds,
                    Ks=Ks,
                    width=width,
                    height=height,
                    sh_degree=opt.sh_degree,
                    near_plane=opt.near_plane,
                    far_plane=opt.far_plane,
                    render_mode="RGB+ED" if opt.depth_loss else "RGB"

            )

            else:

                renders, _, _ = self.rasterize_splats(
                    camtoworlds=camtoworlds,
                    Ks=Ks,
                    width=width,
                    height=height,
                    sh_degree=opt.sh_degree if not self.opt.dino else None,
                    near_plane=opt.near_plane,
                    far_plane=opt.far_plane,
                    render_mode="RGB+ED" if opt.depth_loss else "RGB"

                )  # [1, H, W, 3]  
            if renders.shape[-1] == 4:
                colors, depths_0 = renders[..., 0:3], renders[..., 3:4]
            else:
                colors, depths_0 = renders, None
            torch.cuda.synchronize()
            colors = torch.clamp(colors, 0.0, 1.0)
            if self.opt.dino:
                dino_emb = data['dino_embeddings'][..., 0:3].unsqueeze(0)
                dino_emb = torch.clamp(dino_emb, 0.0, 1.0)
                colors = renders[..., 0:3]
                colors = torch.clamp(colors, 0.0, 1.0)
                depths_0 = renders[..., [-1]]
                canvas = torch.cat([dino_emb, colors], dim=2).squeeze(0).cpu().numpy()
            else:
                canvas = torch.cat([pixels, colors], dim=2).squeeze(0).cpu().numpy()
            

            imageio.imwrite(
                f"{self.render_dir}/val_{current_frame}_{data['image_id']:04d}.png",
                (canvas * 255).astype(np.uint8),
            )
            print(f"Image saved to {self.render_dir}/val_{current_frame}_{data['image_id']:04d}.png")
        
        if opt.depth_loss:
            canvas_depth = (depths_0.squeeze(0).cpu().numpy()* 255)

            # Save the grayscale depth map
            cv2.imwrite(f"{self.depth_dir}/val_{data['image_id']:04d}.png",canvas_depth)
            if self.opt.dino:
                pixels = dino_emb  # [1, 3, H, W]
            pixels = pixels.permute(0, 3, 1, 2)  # [1, 3, H, W]
            colors = colors.permute(0, 3, 1, 2)  # [1, 3, H, W]
            metrics["psnr"].append(self.psnr(colors, pixels))
            metrics["ssim"].append(self.ssim(colors, pixels))
            metrics["lpips"].append(self.lpips(colors, pixels))

        if world_rank == 0:
            ellipse_time /= len(data_block)

            psnr = torch.stack(metrics["psnr"]).mean()
            ssim = torch.stack(metrics["ssim"]).mean()
            lpips = torch.stack(metrics["lpips"]).mean()
            print(
                f"PSNR: {psnr.item():.3f}, SSIM: {ssim.item():.4f}, LPIPS: {lpips.item():.3f} "
                f"Time: {ellipse_time:.3f}s/image "
                f"Number of GS: {len(self.splats['means'])}"
            )
            # save stats as json
            stats = {
                "psnr": psnr.item(),
                "ssim": ssim.item(),
                "lpips": lpips.item(),
                "ellipse_time": ellipse_time,
                "num_GS": len(self.splats["means"]),
            }
            with open(f"{self.stats_dir}/val_step{data['image_id']:04d}.json", "w") as f:
                json.dump(stats, f)
            # save stats to tensorboard
            for k, v in stats.items():
                self.writer.add_scalar(f"val/{k}", v, data["image_id"])
            self.writer.flush()

        return psnr, ssim, lpips,stats
