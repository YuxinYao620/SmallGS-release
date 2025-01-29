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
sys.path.append('/home/yuxinyao/gsplat/')
sys.path.append('/home/yuxinyao/gsplats/submodule/DOMA')
from submodule.DOMA.models.dpfs import MotionField

import sys
sys.path.append('/home/yuxinyao/gsplats')

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
        self.doma_render_dir = f"{opt.workspace}/doma_render_test"
        os.makedirs(self.doma_render_dir, exist_ok=True)
        self.point_cloud_dir = f"{opt.workspace}/point_cloud"
        os.makedirs(self.point_cloud_dir, exist_ok=True)

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
            dinos = opt.dino,
        )
        self.valset = Dataset(self.parser, split="val")
        self.scene_scale = self.parser.scene_scale * 1.1 * opt.global_scale
        print("Scene scale:", self.scene_scale)
        self.trainloader = None
        # Model
        feature_dim = 32 if opt.app_opt else None

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

    def parameters(self):
        return list(self.model.splats.parameters()) + list(self.model.doma.parameters())

    def rasterize_splats(
        self,
        camtoworlds,
        Ks: Tensor,
        width: int,
        height: int,
        doma_mean = None,
        doma_scale = None,
        **kwargs,
    ) -> Tuple[Tensor, Tensor, Dict]:
        
        means = doma_mean if doma_mean is not None else self.splats["means"]
        # quats = F.normalize(self.splats["quats"], dim=-1)  # [N, 4]
        # rasterization does normalization internally
        quats = self.splats["quats"]  # [N, 4]
        scales = torch.exp(self.splats["scales"]) if doma_scale is None else doma_scale # [N, 3]
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
    
    def rasterize_splats_sf(
        self,
        camtoworlds,
        Ks: Tensor,
        width: int,
        height: int,
        means: Tensor,
        quats: Tensor,
        scales: Tensor,
        opacities: Tensor,
        colors: Tensor,
        
        **kwargs,
    ) -> Tuple[Tensor, Tensor, Dict]:
        
        means = means
        # quats = F.normalize(self.splats["quats"], dim=-1)  # [N, 4]
        # rasterization does normalization internally
        quats = self.splats["quats"]  # [N, 4]
        scales = torch.exp(scales)
        opacities = torch.sigmoid(opacities)  # [N,]

        image_ids = kwargs.pop("image_ids", None)
        # if self.opt.app_opt:
        #     colors = self.app_module(
        #         features=self.splats["features"],
        #         embed_ids=image_ids,
        #         dirs=means[None, :, :] - camtoworlds[:, None, :3, 3],
        #         sh_degree=kwargs.pop("sh_degree", self.opt.sh_degree),
        #     )
        #     colors = colors + self.splats["colors"]
        #     colors = torch.sigmoid(colors)
        # else:
        #     colors = torch.cat([self.splats["sh0"], self.splats["shN"]], 1)  # [N, K, 3]

        colors = colors
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
            # # # backgrounds = torch.tensor(self.opt.background_color,device=self.device),
            **kwargs,
        )
        return render_colors, render_alphas, info
    

    def rasterize_splats_sem(
        self,
        camtoworlds,
        Ks: Tensor,
        width: int,
        height: int,
        doma_mean = None,
        doma_scale = None,
        **kwargs,
    ) -> Tuple[Tensor, Tensor, Dict]:
        
        # means = doma_mean if doma_mean is not None else self.splats["means"]
        means = self.splats["means"]
        quats = self.splats["quats"]  # [N, 4]
        scales = torch.exp(self.splats["scales"]) if doma_scale is None else doma_scale # [N, 3]
        opacities = torch.sigmoid(self.splats["opacities"])  # [N,]
        # select only the semantic label ==1 


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
        else:
            colors = torch.cat([self.splats["sh0"], self.splats["shN"]], 1)  # [N, K, 3]
        
        labels = torch.round(self.splats["labels"]).int()
        temp_mean = torch.zeros_like(means)
        if self.opt.doma:
            temp_mean[labels == 0] += doma_mean 
        else:
            pass
        # means[labels == 0] = doma_mean
        # means = torch.where(labels == 0, doma_mean, means)
        # means[labels == 0] *= 1e-6
        means = torch.where(labels.unsqueeze(-1) == 0, temp_mean, means)

        # means = means[labels == 0]
        # quats = quats[labels == 0]
        # scales = scales[labels == 0]
        # opacities = opacities[labels == 0]
        # colors = colors[labels == 0]

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
    def save_ply(self, path, doma_mean = None):
        os.makedirs(os.path.dirname(path), exist_ok=True)

        xyz = self.splats["means"].detach().cpu().numpy() if doma_mean is None else doma_mean.detach().cpu().numpy()
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
        # camera_smoothness_loss = 0.0
        # for i in range(1, len(camera_pose) - 1):
        #     #punish large rotation
        #     prev_rot = camera_pose[i - 1][:3,:3]
        #     next_rot = camera_pose[i][:3,:3]
        #     rot_diff = torch.matmul(prev_rot.t(), next_rot) - torch.eye(3).to(self.device)

        #     #punish large translation
        #     t_diff = torch.matmul(camera_pose[i - 1][:3,:3].t(), (camera_pose[i][:3,3] - camera_pose[i - 1][:3,3]).unsqueeze(0))
        #     L_smooth = torch.F
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
            # breakpoint()
            if camera_opt and (step % 100 == 0) :
                # store the render images 
                self.store_render_image(renders, f'{image_ids[0]}_render_{step}')
                # self.store_render_image(renders, f'doma_{step}')

            if renders.shape[-1] == 4:
                colors, depths_0 = renders[..., 0:3], renders[..., 3::]
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
            l1loss = F.l1_loss(colors, pixels)
            ssimloss = 1.0 - self.ssim(
                pixels.permute(0, 3, 1, 2), colors.permute(0, 3, 1, 2)
            )

            loss = l1loss * (1.0 - self.opt.ssim_lambda) + ssimloss * self.opt.ssim_lambda
            # if opt.depth_loss:
                # depthloss = F.l1_loss(depths_0.squeeze(-1), depthmap_gt)
                # loss += depthloss

            if self.opt.camera_smoothness_loss and camera_opt:
                if self.opt.monst3r_camera:
                    smoothness_loss = self.monst3r_camera_smoothness(camtoworlds_transformed)
                else:
                    smoothness_loss = self.calculate_speed_smoothness(camtoworlds_transformed)
                
                loss += (1- step/max_steps_local)*self.opt.camera_smoothness_lambda * smoothness_loss
            desc = f"loss={loss.item():.3f}| " f"sh degree={sh_degree_to_use}| "
            # if opt.depth_loss:
            #     desc += f"depth loss={depthloss.item():.6f}| "
            # if self.opt.camera_smoothness_loss and camera_opt:
            #     desc += f"smoothness loss={smoothness_loss.item():.6f}| "
            if dino is not None:
                dino_emb = self.trainset.get_dino_emb(colors) # the input require the color to be in 0,255, image shape H,W,3
                dino_gt = dino_emb[image_ids]

                

            pbar.set_description(desc)
            loss.backward()

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

    def init_doma(self,seq_len, device,max_steps):
        model_opt = get_model_config(self.opt.motion_model_name, 
            self.opt.elastic_loss_weight,
            self.opt.homo_loss_weight,
            seq_len, max_steps)
        motion_model_opt = model_opt['motion_model_opt']
        doma = MotionField(model_opt).to(device)
        doma_optimizer = torch.optim.Adam(doma.model.parameters(), lr=motion_model_opt['lr'])
        doma_scheduler = get_scheduler(doma_optimizer, policy='lambda',
                    num_epochs_fix=1,
                    num_epochs=max_steps)
        return doma, doma_optimizer, doma_scheduler

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
        # end_frame = 10
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
                breakpoint()
                semantics_masks = torch.stack([data["mask"] for data in data_block]).to(device)
                depth = depth[semantics_masks[0].view(-1)]
                rgb = rgb[semantics_masks[0].view(-1)]

            self.splats, self.optimizers = create_splats_with_optimizers(
                self.parser,
                init_type=opt.init_type,
                init_num_pts=opt.init_num_pts,
                init_extent=opt.init_extent,
                init_opacity=opt.init_opa,
                init_scale=opt.init_scale,
                scene_scale=self.scene_scale,
                sh_degree=opt.sh_degree,
                sparse_grad=opt.sparse_grad,
                batch_size=opt.batch_size,
                feature_dim=32 if opt.app_opt else None,
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
            else:
                pixels = torch.stack([data["image"] for data in data_block]).to(device) / 255.0
                masks = None


            num_train_rays_per_step = (
                pixels.shape[0] * pixels.shape[1] * pixels.shape[2]
            )
            image_ids = torch.Tensor([data["image_id"] for data in data_block]).int().to(device)
            depthmap_gt = torch.stack([data["depth_map"] for data in data_block]).to(device)  # [b, M]
            if opt.pose_noise:
                camtoworlds = self.pose_perturb(camtoworlds, image_ids)
            if opt.dino:
                dino_gt = torch.stack([data["dino_gt"] for data in data_block]).to(device)                

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
                        # Image.fromarray(canvas.astype(np.uint8)).save(f"{self.sem_dir}/{image_ids[ind]}_rgb.png")
                    # breakpoint()
                    depth_gt_save = Image.fromarray((depthmap_gt[ind].cpu().numpy()*255).astype(np.uint8))
                    depth_gt_save.save(f"{self.depth_gt_dir}/{image_ids[ind]}.png")
            height, width = pixels.shape[1:3]
            
            # sh schedule
            sh_degree_to_use = min(opt.max_steps // opt.sh_degree_interval, opt.sh_degree) 
            if sh_degree_to_use <1:
                sh_degree_to_use = opt.sh_degree
                
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

                print('camera optimization result')
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
                breakpoint()
                est_cam_pose = camtoworlds
                global_cam_poses_est += [camtoworlds.cpu().detach().numpy()]
                global_cam_poses_gt += [camtoworlds_gt.cpu().detach().numpy()]
            # breakpoint()
            est_cam_pose = torch.stack([data_block[0]['camtoworld'].to(device)]+[cam_pose.to(device) for cam_pose in est_cam_pose]).to(device)

            self.eval_all(current_frame, data_block, est_cam_pose)

            self.plot_traj(np.array(global_cam_poses_est), np.array(global_cam_poses_gt), start_frame, block_end)

        if opt.eval_overall_gs:
            self.train_gs_doma(opt.sh_degree, global_cam_poses_est)
        print("pose_ckpt", opt.pose_ckpt)
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
            
            # reinitalize the optimizer and gsplats and strategy and pose_adjust
            # del self.splats, self.optimizers, self.strategy, self.strategy_state, self.pose_adjust, self.pose_optimizers
            # if opt.doma:
            #     del self.doma_optimizer
            torch.cuda.empty_cache()
            return global_camera_poses

    def train_gs_doma(self,sh_degree_to_use,global_cam_poses_est):
        opt = self.opt
        eval_max_step = 5000
        pbar = tqdm.tqdm(range(0, eval_max_step))
        frame_0_depth = self.trainset[0]['depth'].to(self.device)
        frame_0_rgb = self.trainset[0]['image'].reshape(-1, 3).to(self.device) / 255.0

        self.splats, self.optimizers = create_splats_with_optimizers(
                self.parser,
                init_type=opt.init_type,
                init_num_pts=opt.init_num_pts,
                init_extent=opt.init_extent,
                init_opacity=opt.init_opa,
                init_scale=opt.init_scale,
                scene_scale=self.scene_scale,
                sh_degree=opt.sh_degree,
                sparse_grad=opt.sparse_grad,
                batch_size=opt.batch_size,
                feature_dim=32 if opt.app_opt else None,
                device=self.device,
                world_rank=self.world_rank,
                world_size=self.world_size,
                predict_depth = frame_0_depth,
                predict_rgb = frame_0_rgb
            )

        self.strategy = DefaultStrategy(
                verbose=False,
                scene_scale=self.scene_scale,
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
            torch.optim.lr_scheduler.ExponentialLR(
                self.optimizers["means"], gamma=0.01 ** (1.0 / eval_max_step)
            ),
        ]

        # train a first frame
        height,width = self.trainset[0]['image'].shape[0:2]
        
        if opt.doma:
            doma, doma_optimizer, doma_scheduler= self.init_doma(seq_len=len(self.trainset), device=self.device, max_steps = eval_max_step)

        current_index = 0
        for step in pbar:
            # random index
            current_index = current_index+1 if current_index < len(self.trainset)-1 else 0
            tidx = torch.Tensor([current_index]).int().squeeze().to(self.device)
            data = self.trainset[current_index]
            camtoworlds = torch.Tensor(global_cam_poses_est[current_index]).to(self.device).unsqueeze(0)
            Ks = data['K'].unsqueeze(0).to(self.device)
            pixels = data['image'].unsqueeze(0).to(self.device) / 255.0
            image_ids = torch.Tensor([data["image_id"]]).int().to(self.device)
            depth = data['depth'].to(self.device)
            rgb = data['image'].reshape(-1, 3).to(self.device) / 255.0
            height, width = pixels.shape[1:3]
            # if opt.depth_loss:
            depthmap_gt = data['depth_map'].to(self.device)
            if opt.doma:
                points, scaled_factor, min_value = normalize_points(self.splats['means'])
                new_means = doma.deform(points,tidx) 
                new_means = (new_means+1)/2 * scaled_factor + min_value
                weight_factor = 1
                doma_mean = (new_means * weight_factor)
            
            renders , alphas, info = self.rasterize_splats(
                camtoworlds= camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=sh_degree_to_use,
                near_plane=opt.near_plane,
                far_plane=opt.far_plane,
                image_ids=image_ids,
                render_mode="RGB+ED" if opt.depth_loss else "RGB",
                doma_mean=doma_mean if opt.doma else None
            )
            if renders.shape[-1] == 4:
                colors, depths_0 = renders[..., 0:3], renders[..., 3::]
            else:
                colors, depths_0 = renders, None  

            self.strategy.step_pre_backward(
                params=self.splats,
                optimizers=self.optimizers,
                state=self.strategy_state,
                step=step,
                info=info,
            )
            l1loss = F.l1_loss(colors, pixels)
            ssimloss = 1.0 - self.ssim(
                pixels.permute(0, 3, 1, 2), colors.permute(0, 3, 1, 2)
            )
            loss = l1loss * (1.0 - opt.ssim_lambda) + ssimloss * opt.ssim_lambda

            if opt.depth_loss:
                depthloss = F.l1_loss(depths_0.squeeze(-1), depthmap_gt)
            # if opt.doma:
            #     loss_homogeneous_reg = doma.model.homogenous_reg(self.splats['means'],tidx)
            #     # loss_elastic_reg = doma.model.elastic_reg(self.splats['means'],tidx)
            #     # loss += opt.elastic_loss_weight * loss_elastic_reg + 
            #     loss += opt.homo_loss_weight * loss_homogeneous_reg
            desc = f"loss={loss.item():.3f}| " f"sh degree={sh_degree_to_use}| "

            pbar.set_description(desc)
            loss.backward()

            camtoworlds = camtoworlds.detach()

            self.strategy.step_post_backward(
                params=self.splats,
                optimizers=self.optimizers,
                state=self.strategy_state,
                step=step,
                info=info,
                packed=opt.packed,
            )

            for optimizer in self.optimizers.values():
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.app_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for scheduler in schedulers:
                scheduler.step()

            doma_optimizer.step()
            doma_optimizer.zero_grad(set_to_none=True)
            doma_scheduler.step()

            if step == 3000:
                self.eval_all(step, self.trainset, torch.stack([torch.Torch(cam_pose).to(self.device) for cam_pose in global_cam_poses_est]).to(self.device),doma_eval=True,doma=doma)
        # render
        self.eval_all(current_index, self.trainset, torch.stack([torch.Tensor(cam_pose).to(self.device) for cam_pose in global_cam_poses_est]).to(self.device),doma_eval=True,doma=doma)
        # save gs
        data = {"splats": self.splats.state_dict()}
        torch.save(
            data, f"{self.ckpt_dir}/gs_doma_{current_index}_rank{self.world_rank}.pt"
        )
        # save doma 
        doma_weights = {"doma": doma.state_dict()}
        torch.save(
            doma_weights, f"{self.ckpt_dir}/doma_{current_index}_rank{self.world_rank}.pt"
        )
        # save doma config
        with open(f"{self.ckpt_dir}/config_{current_index}_rank{self.world_rank}.json", "w") as f:
            json.dump(vars(opt), f)

    def train_gs_doma2(self,sh_degree_to_use,initial_train_step = 5000):
        opt = self.opt
        eval_max_step = initial_train_step
        doma_pbar = tqdm.tqdm(range(0, eval_max_step))
        gs_max_step = 1000
        gs_pbar = tqdm.tqdm(range(0, gs_max_step))
        frame_0_depth = self.trainset[0]['depth'].to(self.device)
        frame_0_rgb = self.trainset[0]['image'].reshape(-1, 3).to(self.device) / 255.0

        self.splats, self.optimizers = create_splats_with_optimizers(
                self.parser,
                init_type=opt.init_type,
                init_num_pts=opt.init_num_pts,
                init_extent=opt.init_extent,
                init_opacity=opt.init_opa,
                init_scale=opt.init_scale,
                scene_scale=self.scene_scale,
                sh_degree=opt.sh_degree,
                sparse_grad=opt.sparse_grad,
                batch_size=opt.batch_size,
                feature_dim=32 if opt.app_opt else None,
                device=self.device,
                world_rank=self.world_rank,
                world_size=self.world_size,
                predict_depth = frame_0_depth,
                predict_rgb = frame_0_rgb
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
        self.schedulers = [
            torch.optim.lr_scheduler.ExponentialLR(
                self.optimizers["means"], gamma=0.01 ** (1.0 / gs_max_step)
            ),
        ]
        
        frame_0_cam = torch.Tensor(self.trainset[0]['camtoworld']).unsqueeze(0).to(self.device)
        frame_0_K = self.trainset[0]['K'].unsqueeze(0).to(self.device)
        frame_0_image_id = torch.Tensor([self.trainset[0]['image_id']]).int().to(self.device)
        frame_0_pixels = self.trainset[0]['image'].unsqueeze(0).to(self.device) / 255.0
        height,width = self.trainset[0]['image'].shape[0:2]

        for step in gs_pbar:

            renders , alphas, info = self.rasterize_splats(
                camtoworlds= frame_0_cam,
                Ks=frame_0_K,
                width=width,
                height=height,
                sh_degree=sh_degree_to_use,
                near_plane=opt.near_plane,
                far_plane=opt.far_plane,
                image_ids=frame_0_image_id,
                render_mode="RGB+ED" if opt.depth_loss else "RGB"
            )
            if renders.shape[-1] == 4:
                colors, depths_0 = renders[..., 0:3], renders[..., 3::]
            else:
                colors, depths_0 = renders, None  

            self.strategy.step_pre_backward(
                params=self.splats,
                optimizers=self.optimizers,
                state=self.strategy_state,
                step=step,
                info=info,
            )
            l1loss = F.l1_loss(colors, frame_0_pixels)
            ssimloss = 1.0 - self.ssim(
                frame_0_pixels.permute(0, 3, 1, 2), colors.permute(0, 3, 1, 2)
            )
            loss = l1loss * (1.0 - opt.ssim_lambda) + ssimloss * opt.ssim_lambda

            # if opt.depth_loss:
            #     depthloss = F.l1_loss(depths_0.squeeze(-1), depthmap_gt)
            desc = f"loss={loss.item():.3f}| " f"sh degree={sh_degree_to_use}| "

            gs_pbar.set_description(desc)
            loss.backward()

            self.strategy.step_post_backward(
                params=self.splats,
                optimizers=self.optimizers,
                state=self.strategy_state,
                step=step,
                info=info,
                packed=opt.packed,
            )

            for optimizer in self.optimizers.values():
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.app_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for scheduler in self.schedulers:
                scheduler.step()
            
                

        if opt.doma:
            # doma, doma_optimizer, doma_scheduler= self.init_doma(seq_len=len(self.trainset), device=self.device, max_steps = opt.doma_max_step)
            doma, doma_optimizer, doma_scheduler= self.init_doma(seq_len=len(self.trainset), device=self.device, max_steps = 5000)
        self.doma = doma
        self.doma_optimizer = doma_optimizer
        self.doma_scheduler = doma_scheduler
        current_index = 0
        loss_list = []
        for step in doma_pbar:
            colors, depths, renders, alphas, info = self.render_with_doma(self.trainset[current_index],sh_degree_to_use)
            
            if opt.only_increase_gs:
                self.strategy.step_pre_backward(
                    params=self.splats,
                    optimizers=self.optimizers,
                    state=self.strategy_state,
                    step=step,
                    info=info,
                )

            gt_pixel = self.trainset[current_index]['image'].unsqueeze(0).to(self.device) / 255.0

            l1loss = F.l1_loss(colors, gt_pixel)
            ssimloss = 1.0 - self.ssim(
                gt_pixel.permute(0, 3, 1, 2), colors.permute(0, 3, 1, 2)
            )
            loss = l1loss * (1.0 - opt.ssim_lambda) + ssimloss * opt.ssim_lambda

            # if opt.depth_loss:
            #     depthloss = F.l1_loss(depths_0.squeeze(-1), depthmap_gt)
            desc = f"loss={loss.item():.3f}| " f"sh degree={sh_degree_to_use}| "

            doma_pbar.set_description(desc)
            loss.backward()

            if opt.only_increase_gs:
                # do strategy step backward but without pruning
                self.strategy._update_state(self.splats, self.strategy_state,info, packed=opt.packed)
                if step > self.strategy.refine_start_iter and step % self.strategy.refine_every == 0:
                    n_dupli, n_split = self.strategy._grow_gs(self.splats, self.optimizers, self.strategy_state, step)
                self.strategy_state["grad2d"].zero_()
                self.strategy_state["count"].zero_()
                torch.cuda.empty_cache()
                if step % self.strategy.reset_every == 0:
                    reset_opa(
                        params=self.splats,
                        optimizers=self.optimizers,
                        state=self.strategy_state,
                        value=self.strategy.prune_opa * 2.0,
                    )

            for optimizer in self.optimizers.values():
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.app_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for scheduler in self.schedulers:
                scheduler.step()
            self.doma_optimizer.step()
            self.doma_optimizer.zero_grad(set_to_none=True)
            self.doma_scheduler.step()

            current_index = current_index+1 if current_index < len(self.trainset)-1 else 0
            loss_list.append(loss.item())


            # if (step+1) % 50 == 0 and step != 0:
        # # draw loss curve
        # plt.plot(np.array(loss_list))
        # plt.savefig(f"{self.ckpt_dir}/loss_curve_doma5000.png")
        # save gs
        data = {"splats": self.splats.state_dict()}
        torch.save(
            data, f"{self.ckpt_dir}/gs_doma.pt"
        )

        # save optimizer
        optimizer_dict = {k:v.state_dict() for k,v in self.optimizers.items()}
        torch.save(
            optimizer_dict, f"{self.ckpt_dir}/gs_optimizer.pt"
        )
        app_optimizer_dict = {k:v.state_dict() for k,v in self.app_optimizers}
        torch.save(
            app_optimizer_dict, f"{self.ckpt_dir}/gs_app_optimizer.pt"
        )

        torch.save(self.schedulers[0].state_dict(), f"{self.ckpt_dir}/gs_scheduler.pt")

        # save strategy state
        strategy_state = {"strategy_state": self.strategy_state}
        torch.save(
            strategy_state, f"{self.ckpt_dir}/strategy_state.pt"
        )

        # save doma
        doma_save = {"doma": doma.model.state_dict(), "doma_optimizer": doma_optimizer.state_dict(), "doma_scheduler": doma_scheduler.state_dict()}
        torch.save(
            doma_save, f"{self.ckpt_dir}/doma.pt"
        )
        # render
        self.eval_all(current_index, self.trainset, torch.stack([data['camtoworld'].squeeze(0) for data in self.trainset]).to(self.device),doma_eval=True,doma=self.doma)

        # draw loss curve
        plt.plot(np.array(loss_list))
        plt.savefig(f"{self.ckpt_dir}/loss_curve_doma8000.png")

    def train_gs_splatfield(self, sh_degree_to_use, initial_train_step = 5000):

        opt = self.opt
        eval_max_step = initial_train_step
        doma_pbar = tqdm.tqdm(range(0, eval_max_step))
        gs_max_step = 1000
        gs_pbar = tqdm.tqdm(range(0, gs_max_step))
        frame_0_depth = self.trainset[0]['depth'].to(self.device)
        frame_0_rgb = self.trainset[0]['image'].reshape(-1, 3).to(self.device) / 255.0

        self.splats, self.optimizers = create_splats_with_optimizers(
                self.parser,
                init_type=opt.init_type,
                init_num_pts=opt.init_num_pts,
                init_extent=opt.init_extent,
                init_opacity=opt.init_opa,
                init_scale=opt.init_scale,
                scene_scale=self.scene_scale,
                sh_degree=opt.sh_degree,
                sparse_grad=opt.sparse_grad,
                batch_size=opt.batch_size,
                feature_dim=32 if opt.app_opt else None,
                device=self.device,
                world_rank=self.world_rank,
                world_size=self.world_size,
                predict_depth = frame_0_depth,
                predict_rgb = frame_0_rgb
            )

        self.strategy = DefaultStrategy(
                verbose=False,
                scene_scale=self.scene_scale,
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
        self.schedulers = [
            torch.optim.lr_scheduler.ExponentialLR(
                self.optimizers["means"], gamma=0.01 ** (1.0 / gs_max_step)
            ),
        ]
        
        frame_0_cam = torch.Tensor(self.trainset[0]['camtoworld']).unsqueeze(0).to(self.device)
        frame_0_K = self.trainset[0]['K'].unsqueeze(0).to(self.device)
        frame_0_image_id = torch.Tensor([self.trainset[0]['image_id']]).int().to(self.device)
        frame_0_pixels = self.trainset[0]['image'].unsqueeze(0).to(self.device) / 255.0
        height,width = self.trainset[0]['image'].shape[0:2]

        for step in gs_pbar:

            renders , alphas, info = self.rasterize_splats(
                camtoworlds= frame_0_cam,
                Ks=frame_0_K,
                width=width,
                height=height,
                sh_degree=sh_degree_to_use,
                near_plane=opt.near_plane,
                far_plane=opt.far_plane,
                image_ids=frame_0_image_id,
                render_mode="RGB+ED" if opt.depth_loss else "RGB"
            )
            if renders.shape[-1] == 4:
                colors, depths_0 = renders[..., 0:3], renders[..., 3::]
            else:
                colors, depths_0 = renders, None  

            self.strategy.step_pre_backward(
                params=self.splats,
                optimizers=self.optimizers,
                state=self.strategy_state,
                step=step,
                info=info,
            )
            l1loss = F.l1_loss(colors, frame_0_pixels)
            ssimloss = 1.0 - self.ssim(
                frame_0_pixels.permute(0, 3, 1, 2), colors.permute(0, 3, 1, 2)
            )
            loss = l1loss * (1.0 - opt.ssim_lambda) + ssimloss * opt.ssim_lambda

            # if opt.depth_loss:
            #     depthloss = F.l1_loss(depths_0.squeeze(-1), depthmap_gt)
            desc = f"loss={loss.item():.3f}| " f"sh degree={sh_degree_to_use}| "

            gs_pbar.set_description(desc)
            loss.backward()

            self.strategy.step_post_backward(
                params=self.splats,
                optimizers=self.optimizers,
                state=self.strategy_state,
                step=step,
                info=info,
                packed=opt.packed,
            )

            for optimizer in self.optimizers.values():
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.app_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for scheduler in self.schedulers:
                scheduler.step()
            
                

        sys.path.append('/home/yuxinyao/stable-dreamfusion/splatfield')
        from scene import SplatFieldsModel
        self.deform = SplatFieldsModel(self.opt, radius=self.scene_scale)
        self.deform.train_setting(self.opt)

        current_index = 0
        loss_list = []
        self.eval_all("before", self.trainset, torch.stack([data['camtoworld'].squeeze(0) for data in self.trainset]).to(self.device),doma_eval=False,sf_eval=False)
        for step in doma_pbar:
            colors, depths, renders, info = self.render_with_sf(self.trainset[current_index],sh_degree_to_use)
            
            if step == 0:
                # evaluate the initial splatfield
                self.eval_all(step, self.trainset, torch.stack([data['camtoworld'].squeeze(0) for data in self.trainset]).to(self.device),doma_eval=False,sf_eval=True)

            if opt.only_increase_gs:
                self.strategy.step_pre_backward(
                    params=self.splats,
                    optimizers=self.optimizers,
                    state=self.strategy_state,
                    step=step,
                    info=info,
                )

            gt_pixel = self.trainset[current_index]['image'].unsqueeze(0).to(self.device) / 255.0

            l1loss = F.l1_loss(colors, gt_pixel)
            ssimloss = 1.0 - self.ssim(
                gt_pixel.permute(0, 3, 1, 2), colors.permute(0, 3, 1, 2)
            )
            loss = l1loss * (1.0 - opt.ssim_lambda) + ssimloss * opt.ssim_lambda

            # if opt.depth_loss:
            #     depthloss = F.l1_loss(depths_0.squeeze(-1), depthmap_gt)
            desc = f"loss={loss.item():.3f}| " f"sh degree={sh_degree_to_use}| "

            doma_pbar.set_description(desc)
            loss.backward()

            # if opt.only_increase_gs:
            #     # do strategy step backward but without pruning
            #     self.strategy._update_state(self.splats, self.strategy_state,info, packed=opt.packed)
            #     if step > self.strategy.refine_start_iter and step % self.strategy.refine_every == 0:
            #         n_dupli, n_split = self.strategy._grow_gs(self.splats, self.optimizers, self.strategy_state, step)
            #     self.strategy_state["grad2d"].zero_()
            #     self.strategy_state["count"].zero_()
            #     torch.cuda.empty_cache()
            #     if step % self.strategy.reset_every == 0:
            #         reset_opa(
            #             params=self.splats,
            #             optimizers=self.optimizers,
            #             state=self.strategy_state,
            #             value=self.strategy.prune_opa * 2.0,
            #         )
            self.strategy.step_post_backward(
                params=self.splats,
                optimizers=self.optimizers,
                state=self.strategy_state,
                step=step,
                info=info,
                packed=opt.packed,
            )

            for optimizer in self.optimizers.values():
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.app_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for scheduler in self.schedulers:
                scheduler.step()

            self.deform.optimizer.zero_grad()
            self.deform.update_learning_rate(step)

            current_index = current_index+1 if current_index < len(self.trainset)-1 else 0
            loss_list.append(loss.item())

            if step % 1000 == 0 and step != 0:
                self.eval_all(step, self.trainset, torch.stack([data['camtoworld'].squeeze(0) for data in self.trainset]).to(self.device),doma_eval=False,sf_eval=True)

            # if (step+1) % 50 == 0 and step != 0:
        # # draw loss curve
        # plt.plot(np.array(loss_list))
        # plt.savefig(f"{self.ckpt_dir}/loss_curve_doma5000.png")
        # save gs
        data = {"splats": self.splats.state_dict()}
        torch.save(
            data, f"{self.ckpt_dir}/gs_doma.pt"
        )

        # save optimizer
        optimizer_dict = {k:v.state_dict() for k,v in self.optimizers.items()}
        torch.save(
            optimizer_dict, f"{self.ckpt_dir}/gs_optimizer.pt"
        )
        app_optimizer_dict = {k:v.state_dict() for k,v in self.app_optimizers}
        torch.save(
            app_optimizer_dict, f"{self.ckpt_dir}/gs_app_optimizer.pt"
        )

        torch.save(self.schedulers[0].state_dict(), f"{self.ckpt_dir}/gs_scheduler.pt")

        # save strategy state
        strategy_state = {"strategy_state": self.strategy_state}
        torch.save(
            strategy_state, f"{self.ckpt_dir}/strategy_state.pt"
        )

        # # save doma
        # doma_save = {"doma": doma.model.state_dict(), "doma_optimizer": doma_optimizer.state_dict(), "doma_scheduler": doma_scheduler.state_dict()}
        # torch.save(
        #     doma_save, f"{self.ckpt_dir}/doma.pt"
        # )
        # render
        # self.eval_all(current_index, self.trainset, torch.stack([data['camtoworld'].squeeze(0) for data in self.trainset]).to(self.device),doma_eval=True,doma=self.doma)
        self.eval_all(current_index, self.trainset, torch.stack([data['camtoworld'].squeeze(0) for data in self.trainset]).to(self.device),doma_eval=False,sf_eval=True)

        # draw loss curve
        plt.plot(np.array(loss_list))
        plt.savefig(f"{self.ckpt_dir}/loss_curve_doma8000.png")


        # ret = self.deform.step(self.splats['means'],tidx)
        # # overwrite the splats points 
        # self.splats['means'] = ret['means3D']
        # self.splats['scales'] = ret['scales']
        # self.splats['opacities'] = ret['opacity'].squeeze()
        # self.splats['quats'] = ret['rotations']
        # self.splats['sh0'] = ret['rgb'].unsqueeze(1)


        
    def train_gs_doma_sem(self,sh_degree_to_use,initial_train_step = 5000):
        opt = self.opt
        eval_max_step = initial_train_step
        doma_pbar = tqdm.tqdm(range(0, eval_max_step))
        gs_max_step = 1000
        gs_pbar = tqdm.tqdm(range(0, gs_max_step))
        frame_0_depth = self.trainset[0]['depth'].to(self.device)
        frame_0_rgb = self.trainset[0]['image'].reshape(-1, 3).to(self.device) / 255.0
        # semantics_masks = torch.stack([data["mask"] for data in data_block]).to(device)
        # depth = depth[semantics_masks[0].view(-1)]
        # rgb = rgb[semantics_masks[0].view(-1)]
        frame_0_semantics = self.trainset[0]['mask'].flatten().to(self.device) 
        # if opt.use_spec_points and self.parser.spec_data_path is not None:
        #     import joblib
        #     spec_data = joblib.load(self.parser.spec_data_path)
        #     vertices = torch.Tensor(spec_data['smpl_vertices'][0]).to(self.device)
        #     # match vertices to the splats' points
        #     all_means = frame_0_depth
        #     all_labels = frame_0_semantics
        #     means_human = all_means[all_labels == 0]
        #     scale, rotation, translation, aligned_vertices = scale_aware_icp(means_human, vertices)
        #     aligned_rgb = torch.rand((aligned_vertices.shape[0], 3)).to(self.device)
        #     aligned_semantics = torch.zeros((aligned_vertices.shape[0])).to(self.device)
        #     frame_0_depth = torch.cat([frame_0_depth, aligned_vertices], dim=0)
        #     frame_0_semantics = torch.cat([frame_0_semantics, aligned_semantics], dim=0)
        #     frame_0_rgb = torch.cat([frame_0_rgb, aligned_rgb], dim=0)

            
        if opt.use_all_human_depth:
            # compose all the depth into one tensor
            all_depth = []
            all_rgb = []
            all_semantics = []
            for data in self.trainset:
                depth = data["depth"].to(self.device)
                rgb = data["image"].reshape(-1, 3).to(self.device) / 255.0
                if data['image_id'] == 0:
                    semantics_masks = data["mask"].view(-1).to(self.device)
                    all_depth.append(depth)
                    all_rgb.append(rgb)
                    all_semantics.append(semantics_masks.float())
                    
                else:
                    semantics_masks = ~data["mask"].view(-1).to(self.device)
                    depth = depth[semantics_masks]
                    rgb = rgb[semantics_masks]

                    all_semantics.append(torch.zeros(rgb.shape[0]).to(self.device))
                    all_depth.append(depth)
                    all_rgb.append(rgb)
            frame_0_depth = torch.cat(all_depth, dim=0)
            frame_0_rgb = torch.cat(all_rgb, dim=0)
            frame_0_semantics = torch.cat(all_semantics, dim=0)

        
        # duplicate depth and rgb with semantic = 0, add variation to depth 
        mask_0 = frame_0_semantics == 0
        frame_0_depth_extra = frame_0_depth[mask_0] + (torch.randn_like(frame_0_depth[mask_0])*2-1) * 0.01
        frame_0_rgb_extra = frame_0_rgb[mask_0] 
        frame_0_depth = torch.cat([frame_0_depth, frame_0_depth_extra], dim=0)
        frame_0_rgb = torch.cat([frame_0_rgb, frame_0_rgb_extra], dim=0)
        frame_0_semantics = torch.cat([frame_0_semantics, torch.zeros_like(frame_0_semantics[mask_0])], dim=0)
    



        self.splats, self.optimizers = create_splats_with_optimizers_sem(
                self.parser,
                init_type=opt.init_type,
                init_num_pts=opt.init_num_pts,
                init_extent=opt.init_extent,
                init_opacity=opt.init_opa,
                init_scale=opt.init_scale,
                # scene_scale=self.scene_scale,
                sh_degree=opt.sh_degree,
                sparse_grad=opt.sparse_grad,
                batch_size=opt.batch_size,
                feature_dim=32 if opt.app_opt else None,
                device=self.device,
                world_rank=self.world_rank,
                world_size=self.world_size,
                predict_depth = frame_0_depth,
                predict_rgb = frame_0_rgb,
                semantic_label = frame_0_semantics
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
        self.schedulers = [
            torch.optim.lr_scheduler.ExponentialLR(
                self.optimizers["means"], gamma=0.01 ** (1.0 / gs_max_step)
            ),
        ]
        
        frame_0_cam = torch.Tensor(self.trainset[0]['camtoworld']).unsqueeze(0).to(self.device)
        frame_0_K = self.trainset[0]['K'].unsqueeze(0).to(self.device)
        frame_0_image_id = torch.Tensor([self.trainset[0]['image_id']]).int().to(self.device)
        frame_0_pixels = self.trainset[0]['image'].unsqueeze(0).to(self.device) / 255.0
        frame_0_depth_map = self.trainset[0]['depth_map'].unsqueeze(0).unsqueeze(-1).to(self.device)
        height,width = self.trainset[0]['image'].shape[0:2]

        for step in gs_pbar:

            renders , alphas, info = self.rasterize_splats(
                camtoworlds= frame_0_cam,
                Ks=frame_0_K,
                width=width,
                height=height,
                sh_degree=sh_degree_to_use,
                near_plane=opt.near_plane,
                far_plane=opt.far_plane,
                image_ids=frame_0_image_id,
                render_mode="RGB+ED" if opt.depth_loss else "RGB",
                # # backgrounds = torch.tensor(self.opt.background_color,device=self.device).unsqueeze(0) if self.opt.background_color is not None else None

            )
            if renders.shape[-1] == 4:
                colors, depths_0 = renders[..., 0:3], renders[..., 3::]
            else:
                colors, depths_0 = renders, None  

            self.strategy.step_pre_backward(
                params=self.splats,
                optimizers=self.optimizers,
                state=self.strategy_state,
                step=step,
                info=info,
            )
            l1loss = F.l1_loss(colors, frame_0_pixels)
            ssimloss = 1.0 - self.ssim(
                frame_0_pixels.permute(0, 3, 1, 2), colors.permute(0, 3, 1, 2)
            )
            loss = l1loss * (1.0 - opt.ssim_lambda) + ssimloss * opt.ssim_lambda

            if opt.lambda_depth > 0:
                with torch.no_grad():
                    A = torch.cat([depths_0, torch.ones_like(depths_0)], dim=-1) # [B, 2]
                    X = torch.linalg.lstsq(A, depths_0).solution # [2, 1]
                    valid_pred_depth = A @ X # [B, 1]

                    A_gt = torch.cat([frame_0_depth_map, torch.ones_like(frame_0_depth_map)], dim=-1) # [B, 2]
                    X_gt = torch.linalg.lstsq(A_gt, frame_0_depth_map).solution # [2, 1]
                    valid_gt_depth = A_gt @ X_gt # [B, 1]
                lambda_depth = self.opt.lambda_depth #* min(1, self.global_step / self.opt.iters)
                loss = loss + lambda_depth * F.mse_loss(valid_pred_depth, valid_gt_depth)
            # if opt.depth_loss:
            #     depthloss = F.l1_loss(depths_0.squeeze(-1), depthmap_gt)
            desc = f"loss={loss.item():.3f}| " f"sh degree={sh_degree_to_use}| "

            gs_pbar.set_description(desc)
            loss.backward()

            self.strategy.step_post_backward(
                params=self.splats,
                optimizers=self.optimizers,
                state=self.strategy_state,
                step=step,
                info=info,
                packed=opt.packed,
            )

            for name, optimizer in self.optimizers.items():
                # leave labels out 
                if name == 'labels' :
                    continue
                else:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
            for optimizer in self.app_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for scheduler in self.schedulers:
                scheduler.step()
            
                

        doma, doma_optimizer, doma_scheduler= self.init_doma(seq_len=len(self.trainset), device=self.device, max_steps = 5000)
        self.doma = doma
        self.doma_optimizer = doma_optimizer
        self.doma_scheduler = doma_scheduler
        current_index = 0
        loss_list = []
        for step in doma_pbar:
            colors, depths, renders, alphas, info = self.render_with_doma(self.trainset[current_index],sh_degree_to_use)
            
            # if opt.only_increase_gs:
            self.strategy.step_pre_backward(
                params=self.splats,
                optimizers=self.optimizers,
                state=self.strategy_state,
                step=step,
                info=info,
            )

            gt_pixel = self.trainset[current_index]['image'].unsqueeze(0).to(self.device) / 255.0
            l1loss = F.l1_loss(colors, gt_pixel)


            ssimloss = 1.0 - self.ssim(
                gt_pixel.permute(0, 3, 1, 2), colors.permute(0, 3, 1, 2)
            )


            if 'labels' in self.splats.keys():
                mask = ~self.trainset[current_index]['mask'].to(self.device) 
                gt_pixel = gt_pixel*mask
                colors_dynamic = colors*mask
                l1loss_dynamic = F.l1_loss(colors_dynamic, gt_pixel)
                loss = 0.7*l1loss_dynamic + 0.3*l1loss
            else:
                loss = l1loss * (1.0 - opt.ssim_lambda) + ssimloss * opt.ssim_lambda

            if opt.lambda_depth > 0:
                with torch.no_grad():
                    A = torch.cat([depths, torch.ones_like(depths)], dim=-1) # [B, 2]
                    X = torch.linalg.lstsq(A, depths).solution # [2, 1]
                    valid_pred_depth = A @ X # [B, 1]

                    gt_depth = self.trainset[current_index]['depth_map'].unsqueeze(-1).unsqueeze(0).to(self.device)

                    A_gt = torch.cat([gt_depth, torch.ones_like(gt_depth)], dim=-1) # [B, 2]
                    X_gt = torch.linalg.lstsq(A_gt, gt_depth).solution # [2, 1]
                    valid_gt_depth = A_gt @ X_gt # [B, 1]
                lambda_depth = self.opt.lambda_depth #* min(1, self.global_step / self.opt.iters)
                loss = loss + lambda_depth * F.mse_loss(valid_pred_depth, valid_gt_depth)


            # if opt.depth_loss:
            #     depthloss = F.l1_loss(depths_0.squeeze(-1), depthmap_gt)
            desc = f"loss={loss.item():.3f}| " f"sh degree={sh_degree_to_use}| "

            doma_pbar.set_description(desc)
            loss.backward()

            if opt.only_increase_gs:
                # do strategy step backward but without pruning
                self.strategy._update_state(self.splats, self.strategy_state,info, packed=opt.packed)
                if step > self.strategy.refine_start_iter and step % self.strategy.refine_every == 0:
                    n_dupli, n_split = self.strategy._grow_gs(self.splats, self.optimizers, self.strategy_state, step)
                self.strategy_state["grad2d"].zero_()
                self.strategy_state["count"].zero_()
                torch.cuda.empty_cache()
                if step % self.strategy.reset_every == 0:
                    reset_opa(
                        params=self.splats,
                        optimizers=self.optimizers,
                        state=self.strategy_state,
                        value=self.strategy.prune_opa * 2.0,
                    )
            else:
                self.strategy.step_post_backward(
                    params=self.splats,
                    optimizers=self.optimizers,
                    state=self.strategy_state,
                    step=step,
                    info=info,
                    packed=opt.packed,
                )
            # breakpoint()
            for optimizer in self.optimizers.values():
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.app_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            # breakpoint()
            for scheduler in self.schedulers:
                scheduler.step()
            self.doma_optimizer.step()
            self.doma_optimizer.zero_grad(set_to_none=True)
            self.doma_scheduler.step()

            current_index = current_index+1 if current_index < len(self.trainset)-1 else 0
            loss_list.append(loss.item())

            # # save gs every 100 epoch 
            # if step % 10 == 0 and step != 0:
            #     self.save_model(step)
            #     self.eval_all(step, self.trainset, torch.stack([data['camtoworld'].squeeze(0) for data in self.trainset]).to(self.device),doma_eval=self.opt.doma,doma=self.doma)
            #     self.load_model(step)
            #     self.splats.train()
            #     if self.opt.doma:
            #         self.doma.model.train()


        data = {"splats": self.splats.state_dict()}
        torch.save(
            data, f"{self.ckpt_dir}/gs_doma.pt"
        )

        # save optimizer
        optimizer_dict = {k:v.state_dict() for k,v in self.optimizers.items()}
        # optimizer_dict['step'] = step
        torch.save(
            optimizer_dict, f"{self.ckpt_dir}/gs_optimizer.pt"
        )
        app_optimizer_dict = {k:v.state_dict() for k,v in self.app_optimizers}
        torch.save(
            app_optimizer_dict, f"{self.ckpt_dir}/gs_app_optimizer.pt"
        )

        torch.save(self.schedulers[0].state_dict(), f"{self.ckpt_dir}/gs_scheduler.pt")

        # save strategy state
        strategy_state = {"strategy_state": self.strategy_state}
        torch.save(
            strategy_state, f"{self.ckpt_dir}/strategy_state.pt"
        )

        # save doma
        doma_save = {"doma": doma.model.state_dict(), "doma_optimizer": doma_optimizer.state_dict(), "doma_scheduler": doma_scheduler.state_dict()}
        torch.save(
            doma_save, f"{self.ckpt_dir}/doma.pt"
        )
        # render
        self.eval_all(current_index, self.trainset, torch.stack([data['camtoworld'].squeeze(0) for data in self.trainset]).to(self.device),doma_eval=True,doma=self.doma)


    def save_model(self, epoch):
        data = {"splats": self.splats.state_dict()}
        torch.save(
            data, f"{self.ckpt_dir}/gs_doma_{epoch}.pt"
        )

        # save optimizer
        optimizer_dict = {k:v.state_dict() for k,v in self.optimizers.items()}
        # optimizer_dict['step'] = step
        torch.save(
            optimizer_dict, f"{self.ckpt_dir}/gs_optimizer_{epoch}.pt"
        )
        app_optimizer_dict = {k:v.state_dict() for k,v in self.app_optimizers}
        torch.save(
            app_optimizer_dict, f"{self.ckpt_dir}/gs_app_optimizer_{epoch}.pt"
        )

        torch.save(self.schedulers[0].state_dict(), f"{self.ckpt_dir}/gs_scheduler_{epoch}.pt")

        # save strategy state
        strategy_state = {"strategy_state": self.strategy_state}
        torch.save(
            strategy_state, f"{self.ckpt_dir}/strategy_state_{epoch}.pt"
        )
        if self.opt.doma:
            # save doma
            doma_save = {"doma": self.doma.model.state_dict(), "doma_optimizer": self.doma_optimizer.state_dict(), "doma_scheduler": self.doma_scheduler.state_dict()}
            torch.save(
                doma_save, f"{self.ckpt_dir}/doma_{epoch}.pt"
            )
    
    def load_model(self, epoch, path = None):
        if path ==None:
            path = self.ckpt_dir
        gs_pre = torch.load(f"{path}/gs_doma_{epoch}.pt")
        self.splats.load_state_dict(gs_pre['splats'])
        gs_optimizer_states = torch.load(f"{path}/gs_optimizer_{epoch}.pt")
        for name,optimizer in self.optimizers.items():
            if name == 'labels':
                continue
            if len(gs_optimizer_states[name]) > 0:
                optimizer.load_state_dict(gs_optimizer_states[name])
                #print(lr)
                print("learning rate for ", name, optimizer.param_groups[0]['lr'])
        gs_scheduler_states = torch.load(f"{path}/gs_scheduler_{epoch}.pt")
        # gs_scheduler.load_state_dict(gs_scheduler_states)
        for index, scheduler in enumerate(self.schedulers):
            scheduler.load_state_dict(gs_scheduler_states)
            print("load scheduler for ", index)
        strategy_state = torch.load(f"{path}/strategy_state_{epoch}.pt")
        self.strategy_state = strategy_state['strategy_state']

        if self.opt.doma:
            doma_save = torch.load(f"{path}/doma_{epoch}.pt")
            self.doma.model.load_state_dict(doma_save['doma'])
            self.doma_optimizer.load_state_dict(doma_save['doma_optimizer'])
            self.doma_scheduler.load_state_dict(doma_save['doma_scheduler'])



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
        if 'labels' in self.splats.keys():

            # overwrite the corresponding points

            renders, alphas, info = self.rasterize_splats_sem(
                camtoworlds= camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=sh_degree_to_use,
                near_plane=opt.near_plane,
                far_plane=opt.far_plane,
                image_ids=image_ids,
                render_mode="RGB+ED" if opt.depth_loss else "RGB",
                # # # backgrounds = torch.tensor(self.opt.background_color,device=self.device).unsqueeze(0) if self.opt.background_color is not None else None


            )
        else:
            points, scaled_factor, min_value = normalize_points(self.splats['means'])
            new_means = self.doma.deform(points,tidx) 
            new_means = (new_means+1)/2 * scaled_factor + min_value
            weight_factor = 1
            doma_mean = (new_means * weight_factor)

            renders , alphas, info = self.rasterize_splats(
                camtoworlds= camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=sh_degree_to_use,
                near_plane=opt.near_plane,
                far_plane=opt.far_plane,
                # image_ids=image_ids,
                render_mode="RGB+ED" if opt.depth_loss else "RGB",
                doma_mean=doma_mean if opt.doma else None,
                # # # backgrounds = torch.tensor(self.opt.background_color,device=self.device).unsqueeze(0) if self.opt.background_color is not None else None

            )

        if renders.shape[-1] == 4:
            colors, depths = renders[..., 0:3], renders[..., 3::]
        else:
            colors, depths = renders, None

        # breakpoint()
        # mask = self.select_transparent_pixels(data['image'], transparent_color=[0, 0, 0]).unsqueeze(-1)

        # colors = colors * (~mask.to(self.device))
        
        return colors, depths, renders, info
    

    def render_with_doma(self, data,sh_degree_to_use): # work with dataloader
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
        if 'labels' in self.splats.keys():
            dynamic_points = self.splats['means'][self.splats['labels'] == 0]
            points, scaled_factor, min_value = normalize_points(dynamic_points)
            new_means = self.doma.deform(points,tidx)
            new_means = (new_means+1)/2 * scaled_factor + min_value
            weight_factor = 1
            doma_mean = (new_means * weight_factor)

            # overwrite the corresponding points
            renders, alphas, info = self.rasterize_splats_sem(
                camtoworlds= camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=sh_degree_to_use,
                near_plane=opt.near_plane,
                far_plane=opt.far_plane,
                image_ids=image_ids,
                render_mode="RGB+ED" if opt.depth_loss else "RGB",
                doma_mean=doma_mean if opt.doma else None,
                # # # backgrounds = torch.tensor(self.opt.background_color,device=self.device).unsqueeze(0) if self.opt.background_color is not None else None


            )
        else:
            points, scaled_factor, min_value = normalize_points(self.splats['means'])
            new_means = self.doma.deform(points,tidx) 
            new_means = (new_means+1)/2 * scaled_factor + min_value
            weight_factor = 1
            doma_mean = (new_means * weight_factor)

            renders , alphas, info = self.rasterize_splats(
                camtoworlds= camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=sh_degree_to_use,
                near_plane=opt.near_plane,
                far_plane=opt.far_plane,
                # image_ids=image_ids,
                render_mode="RGB+ED" if opt.depth_loss else "RGB",
                doma_mean=doma_mean if opt.doma else None,
                # # # backgrounds = torch.tensor(self.opt.background_color,device=self.device).unsqueeze(0) if self.opt.background_color is not None else None


            )

        if renders.shape[-1] == 4:
            colors, depths = renders[..., 0:3], renders[..., 3::]
        else:
            colors, depths = renders, None

        # breakpoint()
        # mask = self.select_transparent_pixels(data['image'], transparent_color=[0, 0, 0]).unsqueeze(-1)

        # colors = colors * (~mask.to(self.device))
        
        return colors, depths, renders, alphas, info

    def render_with_sf(self, data,sh_degree_to_use): # work with dataloader
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

        # points, scaled_factor, min_value = normalize_points(self.splats['means'])
        # self.splats['means'] = points
        # new_means = self.doma.deform(points,tidx) 
        # new_means = (new_means+1)/2 * scaled_factor + min_value
        # weight_factor = 1
        # doma_mean = (new_means * weight_factor)

        # ret = self.deform.step(self.splats['means'],tidx)
        # # overwrite the splats points 
        # self.splats['means'] = ret['means3D']
        # self.splats['scales'] = ret['scales']
        # self.splats['opacities'] = ret['opacity'].squeeze()
        # self.splats['quats'] = ret['rotations']
        # self.splats['sh0'] = ret['rgb'].unsqueeze(1)

        
        renders , alphas, info = self.rasterize_splats(
            camtoworlds= camtoworlds,
            Ks=Ks,
            width=width,
            height=height,
            sh_degree=sh_degree_to_use,
            near_plane=opt.near_plane,
            far_plane=opt.far_plane,
            # image_ids=image_ids,
            render_mode="RGB+ED" if opt.depth_loss else "RGB",
            # doma_mean=doma_mean if opt.doma else None
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
    def eval_all(self, current_frame, data_block, est_cam_pose,doma_eval=False,doma=None, sf_eval = None,sds= False):
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
                if self.opt.doma:
                    new_means = self.doma.deform(points,tidx)
                    new_means = (new_means+1)/2 * scaled_factor + min_value
                    weight_factor = 1
                    doma_mean = (new_means * weight_factor)
                else:
                    pass

                # overwrite the corresponding points
                renders, alphas, info = self.rasterize_splats_sem(
                    camtoworlds= camtoworlds,
                    Ks=Ks,
                    width=width,
                    height=height,
                    sh_degree=opt.sh_degree,
                    near_plane=opt.near_plane,
                    far_plane=opt.far_plane,
                    render_mode="RGB+ED" if opt.depth_loss else "RGB",
                    doma_mean=doma_mean if opt.doma else None,
                    # # # backgrounds = torch.tensor(self.opt.background_color,device=self.device).unsqueeze(0) if self.opt.background_color is not None else None


            )

            else:
                if doma_eval:
                    points, scaled_factor, min_value = normalize_points(self.splats['means'])
                    new_means = doma.deform(points,tidx) 
                    new_means = (new_means+1)/2 * scaled_factor + min_value
                    doma_mean = new_means
                if sf_eval:
                    ret = self.deform.step(self.splats['means'],tidx)
                    # overwrite the splats points
                    self.splats['means'] = ret['means3D']
                    self.splats['scales'] = ret['scales']
                    self.splats['opacities'] = ret['opacity'].squeeze()
                    self.splats['quats'] = ret['rotations']
                    self.splats['sh0'] = ret['rgb'].unsqueeze(1)

                # render the splats
                renders, _, _ = self.rasterize_splats(
                    camtoworlds=camtoworlds,
                    Ks=Ks,
                    width=width,
                    height=height,
                    sh_degree=opt.sh_degree,
                    near_plane=opt.near_plane,
                    far_plane=opt.far_plane,
                    render_mode="RGB+ED" if opt.depth_loss else "RGB",
                    doma_mean=doma_mean if doma_eval else None,
                    # # # backgrounds = torch.tensor(self.opt.background_color,device=self.device).unsqueeze(0) if self.opt.background_color is not None else None

                )  # [1, H, W, 3]  
            if renders.shape[-1] == 4:
                colors, depths_0 = renders[..., 0:3], renders[..., 3:4]
            else:
                colors, depths_0 = renders, None
            colors = torch.clamp(colors, 0.0, 1.0)
            torch.cuda.synchronize()

            canvas = torch.cat([pixels, colors], dim=2).squeeze(0).cpu().numpy()
            if doma_eval:
                if not sds:
                    imageio.imwrite(
                        f"{self.doma_render_dir}/{data['image_id']:04d}_doma.png",
                        (canvas * 255).astype(np.uint8),
                    )
                    print(f"Image saved to {self.doma_render_dir}/{data['image_id']:04d}_doma.png")
                    # seperately save colors and pixesl 
                    imageio.imwrite(
                        f"{self.doma_render_dir}/color_{data['image_id']:04d}_doma.png",
                        (colors.squeeze(0).cpu().numpy() * 255).astype(np.uint8),
                    )
                    imageio.imwrite(
                        f"{self.doma_render_dir}/original_{data['image_id']:04d}_doma.png",
                        (pixels.squeeze(0).cpu().numpy() * 255).astype(np.uint8))
                    # save plg
                    labels = torch.round(self.splats["labels"]).int()
                    temp_mean = torch.zeros_like(self.splats["means"])
                    temp_mean[labels == 0] += doma_mean
                    # means[labels == 0] = doma_mean
                    # means = torch.where(labels == 0, doma_mean, means)
                    # means[labels == 0] *= 1e-6
                    save_means = torch.where(labels.unsqueeze(-1) == 0, temp_mean, self.splats["means"])

                    self.save_ply(f"{self.point_cloud_dir}/{data['image_id']:04d}_doma.ply",save_means)
                elif "labels" in self.splats.keys():
                    labels = torch.round(self.splats["labels"]).int()
                    temp_mean = torch.zeros_like(self.splats["means"])
                    temp_mean[labels == 0] += doma_mean
                    save_means = torch.where(labels.unsqueeze(-1) == 0, temp_mean, self.splats["means"])

                    self.save_ply(f"{self.point_cloud_dir}/sds_{current_frame}_{data['image_id']:04d}_doma.ply",save_means) 
                else:
                    self.save_ply(f"{self.point_cloud_dir}/sds_{current_frame}_{data['image_id']:04d}_doma.ply",self.splats["means"])

            if sf_eval:
                imageio.imwrite(
                    f"{self.doma_render_dir}/{current_frame}_{data['image_id']:04d}_sf.png",
                    (canvas * 255).astype(np.uint8),
                )
                print(f"Image saved to {self.doma_render_dir}/{data['image_id']:04d}_sf.png")

            else:
                imageio.imwrite(
                    f"{self.render_dir}/val_{current_frame}_{data['image_id']:04d}.png",
                    (canvas * 255).astype(np.uint8),
                )
                print(f"Image saved to {self.render_dir}/val_{current_frame}_{data['image_id']:04d}.png")
            
            if opt.depth_loss:
                canvas_depth = (depths_0.squeeze(0).cpu().numpy()* 255)

                # Save the grayscale depth map
                cv2.imwrite(f"{self.depth_dir}/val_{data['image_id']:04d}.png",canvas_depth)
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
    @torch.no_grad()
    def render_traj(self, step: int):
        """Entry for trajectory rendering."""
        print("Running trajectory rendering...")
        opt = self.opt
        device = self.device

        camtoworlds = self.parser.camtoworlds[5:-5]
        camtoworlds = generate_interpolated_path(camtoworlds, 1)  # [N, 3, 4]
        camtoworlds = np.concatenate(
            [
                camtoworlds,
                np.repeat(np.array([[[0.0, 0.0, 0.0, 1.0]]]), len(camtoworlds), axis=0),
            ],
            axis=1,
        )  # [N, 4, 4]

        camtoworlds = torch.from_numpy(camtoworlds).float().to(device)
        K = torch.from_numpy(list(self.parser.Ks_dict.values())[0]).float().to(device)
        width, height = list(self.parser.imsize_dict.values())[0]

        canvas_all = []
        for i in tqdm.trange(len(camtoworlds), desc="Rendering trajectory"):
            renders, _, _ = self.rasterize_splats(
                camtoworlds=camtoworlds[i : i + 1],
                Ks=K[None],
                width=width,
                height=height,
                sh_degree=opt.sh_degree,
                near_plane=opt.near_plane,
                far_plane=opt.far_plane,
                render_mode="RGB+ED",
            )  # [1, H, W, 4]
            colors = torch.clamp(renders[0, ..., 0:3], 0.0, 1.0)  # [H, W, 3]
            depths = renders[0, ..., 3:4]  # [H, W, 1]
            depths = (depths - depths.min()) / (depths.max() - depths.min())

            # write images
            canvas = torch.cat(
                [colors, depths.repeat(1, 1, 3)], dim=0 if width > height else 1
            )
            canvas = (canvas.cpu().numpy() * 255).astype(np.uint8)
            canvas_all.append(canvas)

        # save to video
        video_dir = f"{opt.workspace}/videos"
        os.makedirs(video_dir, exist_ok=True)
        writer = imageio.get_writer(f"{video_dir}/traj_{step}.mp4", fps=30)
        for canvas in canvas_all:
            writer.append_data(canvas)
        writer.close()
        print(f"Video saved to {video_dir}/traj_{step}.mp4")
    
    def eval_pose(self, pose_path,  interpolate=True):
        """Evaluate camera pose."""
        # pose_path = os.path.join(result_path, 'ep00_init.pth')
        # breakpoint()
        base_dir = os.path.dirname(pose_path)
        result_path = os.path.dirname(pose_path)
        poses = torch.load(pose_path)
        gt_pose = torch.from_numpy(np.load(base_dir+"/traj_est.npy")) # 
        poses_pred = torch.Tensor(poses['global_cam_poses_est'])
        # poses_gt_c2w = torch.Tensor(poses['global_cam_poses_gt']) 
        poses_gt_c2w = gt_pose
        print("poses_pred[0]", poses_pred[0])
        print("poses_gt_c2w[0]", poses_gt_c2w[0])
        poses_gt = poses_gt_c2w[:len(poses_pred)].clone() 
        if True: 
            poses_gt = poses_gt[:len(poses_pred)].clone()+ torch.randn(poses_gt[:len(poses_pred)].shape)*0.01
            # poses_gt = np.load(pose_path.split("/")[:-1]+"/traj_est.npy")
        # align scale first (we do this because scale differennt a lot)
        trans_gt_align, trans_est_align, _ = self.align_pose(poses_gt[:, :3, -1].numpy(),
                                                             poses_pred[:, :3, -1].numpy())
        print("trans_gt_align[0]", trans_gt_align[0])
        print("trans_est_align[0]", trans_est_align[0])
        poses_gt[:, :3, -1] = torch.from_numpy(trans_gt_align)
        poses_pred[:, :3, -1] = torch.from_numpy(trans_est_align)

        c2ws_est_aligned = align_ate_c2b_use_a2b(poses_pred, poses_gt)

        ate = compute_ATE(poses_gt.cpu().numpy(),
                          c2ws_est_aligned.cpu().numpy())
        rpe_trans, rpe_rot = compute_rpe(
            poses_gt.cpu().numpy(), c2ws_est_aligned.cpu().numpy())
        print("{0:.3f}".format(rpe_trans*100),
              '&' "{0:.3f}".format(rpe_rot * 180 / np.pi),
              '&', "{0:.3f}".format(ate))
        plot_pose(poses_gt, c2ws_est_aligned, pose_path)
       
        print("reults saved in ", result_path)
        with open(f"{result_path}/pose_eval.txt", 'w') as f:
            f.write("RPE_trans: {:.03f}, RPE_rot: {:.03f}, ATE: {:.03f}".format(
                rpe_trans*100,
                rpe_rot * 180 / np.pi,
                ate))
            f.close()
        



    def eval_pose_droid(self, pose_path,  interpolate=True,disable_plot=False):
        """Evaluate camera pose."""
        # pose_path = os.path.join(result_path, 'ep00_init.pth')
        base_dir = os.path.dirname(pose_path)
        result_path = os.path.dirname(pose_path)
        poses = torch.load(pose_path)
        gt_pose = torch.from_numpy(np.load(base_dir+"/traj_est.npy"))
        poses_pred = torch.Tensor(poses['global_cam_poses_est'])
        # poses_gt_c2w = gt_pose
        # poses_gt = poses_gt_c2w[:len(poses_pred)].clone()
        output_path = os.path.join(result_path, 'pose_vis_traj.png')
        # gt_pose = gt_pose[:20,:]
        # poses_pred = poses_pred[:20,:,:]

        from evo.core.trajectory import PoseTrajectory3D,PosePath3D

        from evo.tools import plot
        import copy
        from evo.core.metrics import PoseRelation
        traj_ref = PosePath3D(
        positions_xyz=gt_pose[:,:3],
        orientations_quat_wxyz=gt_pose[:,3:])
        

        

        traj_est = PosePath3D(poses_se3=poses_pred)
        traj_est_aligned = copy.deepcopy(traj_est)
        traj_est_aligned.align(traj_ref, correct_scale=True,
                           correct_only_scale=False)
        traj_est_aligned.align(traj_ref, correct_scale=True,
                           correct_only_scale=False)
        fig = plt.figure()
        traj_by_label = {
            # "estimate (not aligned)": traj_est,
            "Ours (aligned)": traj_est_aligned,
            # "Ground-truth": traj_ref
            "Droid": traj_ref
        }
        plot_mode = plot.PlotMode.xyz
        # ax = plot.prepare_axis(fig, plot_mode, 111)
        ax = fig.add_subplot(111, projection="3d")
        ax.xaxis.set_tick_params(labelbottom=True)
        ax.yaxis.set_tick_params(labelleft=True)
        ax.zaxis.set_tick_params(labelleft=True)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        colors = ['r', 'b']
        styles = ['-', '--']

        for idx, (label, traj) in enumerate(traj_by_label.items()):
            plot.traj(ax, plot_mode, traj,
                    styles[idx], colors[idx], label)
            # break
        # plot.trajectories(fig, traj_by_label, plot.PlotMode.xyz)
        ax.view_init(elev=10., azim=45)
        plt.tight_layout()
        # pose_vis_path = os.path.join(os.path.dirname(output_path), 'pose_vis.png')
        pose_vis_path = output_path 
        fig.savefig(pose_vis_path)

        dic = {
            'traj_est_aligned': traj_est_aligned.positions_xyz,
            'traj_ref': traj_ref.positions_xyz,
            'traj_est_aligned_wxyz': traj_est_aligned.orientations_quat_wxyz,
            'traj_ref_wxyz': traj_ref.orientations_quat_wxyz
        }
        # breakpoint()
        torch.save(dic, os.path.join(self.ckpt_dir, "aligned_pose.pt"))
        print("save to ", os.path.join(self.ckpt_dir, "aligned_pose.pt"))


def main(local_rank: int, world_rank, world_size: int, opt):
    global starttime
    starttime = time.time()
    # make starttime global

    runner = Runner(local_rank, world_rank, world_size, opt)


    if opt.pose_ckpt is not None and opt.eval_pose:
        runner.eval_pose_droid(opt.pose_ckpt,opt.disable_plot)
    else:
        runner.train()

    #eval camera pose
    
    endtime = time.time()
    print("Total time: ", endtime - starttime)



if __name__ == "__main__":
    """
    Usage:

    ```bash
    # Single GPU training
    CUDA_VISIBLE_DEVICES=0 python simple_trainer.py

    # Distributed training on 4 GPUs: Effectively 4x batch size so run 4x less steps.
    CUDA_VISIBLE_DEVICES=0,1,2,3 python simple_trainer.py --steps_scaler 0.25

    """

    opt = tyro.cli(Config)
    opt.adjust_steps(opt.steps_scaler)
    cli(main, opt, verbose=True)

# CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1 TORCH_USE_CUDA_DSA=1 python examples/cf3dgs.py --eval_steps -1 --disable_viewer --data_factor 1  --data_dir /home/yuxinyao/gsplat/data/co3d/ball/242_25689_51045 --workspace results/benchmark/ball/ --max_steps 1000 --batch_size 3 --pose_opt --init_type predict --depth_loss --datatype "co3d" --pose_ckpt /home/yuxinyao/gsplat/results/benchmark/ball/traj/global_camera_poses_20_to30_rank0.pt