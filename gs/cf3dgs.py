import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import imageio
import nerfview
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import tyro
import viser
from datasets.colmap import Dataset, Parser, fov2focal
from datasets.traj import generate_interpolated_path
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from utils import AppearanceOptModule, CameraOptModule, knn, rgb_to_sh, set_random_seed, matrix_to_quaternion, quaternion_to_matrix

from utils_poses.align_traj import align_ate_c2b_use_a2b
from utils_poses.comp_ate import compute_rpe, compute_ATE
from vis_utils import plot_pose

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy

from gsplat.distributed import cli
from gsplat.rendering import rasterization
from gsplat.strategy import DefaultStrategy

from lietorch import SO3, SE3, Sim3, LieGroupParameter
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

torch.autograd.set_detect_anomaly(True)

@dataclass
class Config:
    # Disable viewer
    disable_viewer: bool = False
    disable_plot : bool = False
    # Path to the .pt file. If provide, it will skip training and render a video
    ckpt: Optional[str] = None
    pose_ckpt: Optional[str] = None

    # Path to the Mip-NeRF 360 dataset
    data_dir: str = "data/360_v2/garden"
    # Downsample factor for the dataset
    data_factor: int = 4
    # Directory to save results
    result_dir: str = "results/garden"
    # Every N images there is a test image
    test_every: int = 8
    # Random crop size for training  (experimental)
    patch_size: Optional[int] = None
    # A global scaler that applies to the scene size related parameters
    global_scale: float = 1.0

    # Port for the viewer server
    port: int = 8080

    # Batch size for training. Learning rates are scaled automatically
    batch_size: int = 1
    # A global factor to scale the number of training steps
    steps_scaler: float = 1.0

    # Number of training steps
    max_steps: int = 30_000
    # Steps to evaluate the model
    eval_steps: List[int] = field(default_factory=lambda: [7_000, 30_000])
    # Steps to save the model
    save_steps: List[int] = field(default_factory=lambda: [7_000, 30_000])

    # Initialization strategy
    init_type: str = "sfm"
    # Initial number of GSs. Ignored if using sfm
    init_num_pts: int = 100_000
    # Initial extent of GSs as a multiple of the camera extent. Ignored if using sfm
    init_extent: float = 3.0
    # Degree of spherical harmonics
    sh_degree: int = 3
    # Turn on another SH degree every this steps
    sh_degree_interval: int = 1000
    # Initial opacity of GS
    init_opa: float = 0.1
    # Initial scale of GS
    init_scale: float = 1.0
    # Weight for SSIM loss
    # ssim_lambda: float = 0.2
    ssim_lambda: float = 0.0

    # Near plane clipping distance
    near_plane: float = 0.01
    # Far plane clipping distance
    far_plane: float = 1e10

    # GSs with opacity below this value will be pruned
    prune_opa: float = 0.005
    # GSs with image plane gradient above this value will be split/duplicated
    grow_grad2d: float = 0.0002
    # GSs with scale below this value will be duplicated. Above will be split
    grow_scale3d: float = 0.01
    # GSs with scale above this value will be pruned.
    prune_scale3d: float = 0.1

    # Start refining GSs after this iteration
    refine_start_iter: int = 500
    # Stop refining GSs after this iteration
    refine_stop_iter: int = 15_000
    # Reset opacities every this steps
    reset_every: int = 3000
    # Refine GSs every this steps
    refine_every: int = 100

    # Use packed mode for rasterization, this leads to less memory usage but slightly slower.
    packed: bool = False
    # Use sparse gradients for optimization. (experimental)
    sparse_grad: bool = False
    # Use absolute gradient for pruning. This typically requires larger --grow_grad2d, e.g., 0.0008 or 0.0006
    absgrad: bool = False
    # Anti-aliasing in rasterization. Might slightly hurt quantitative metrics.
    antialiased: bool = False
    # Whether to use revised opacity heuristic from arXiv:2404.06109 (experimental)
    revised_opacity: bool = False

    # Use random background for training to discourage transparency
    random_bkgd: bool = False

    # Enable camera optimization.
    pose_opt: bool = False
    # Learning rate for camera optimization
    # pose_opt_lr: float = 1e-4
    pose_opt_lr: float = 1e-2
    # pose_opt_lr: float = 1e-4
    # Regularization for camera optimization as weight decay
    pose_opt_reg: float = 1e-5
    # Add noise to camera extrinsics. This is only to test the camera pose optimization.
    pose_noise: float = 0.0

    cf3dgs_position_lr_init = 0.00016
    cf3dgs_position_lr_final = 0.0000016
    cf3dgs_position_lr_delay_mult = 0.01
    cf3dgs_position_lr_max_steps = 30_000

    # Enable appearance optimization. (experimental)
    app_opt: bool = False
    # Appearance embedding dimension
    app_embed_dim: int = 16
    # Learning rate for appearance optimization
    app_opt_lr: float = 1e-3
    # Regularization for appearance optimization as weight decay
    app_opt_reg: float = 1e-6

    # Enable depth loss. (experimental)
    depth_loss: bool = False
    # Weight for depth loss
    depth_lambda: float = 1e-2

    # Dump information to tensorboard every this steps
    tb_every: int = 100
    # Save training images to tensorboard
    tb_save_image: bool = False

    rotation_lr: float = 1e-3

    datatype: str = "colmap"

    doma : bool = False
    motion_model_name: str = "transfield4d"
    motion_color_name: str = "transfield4d"
    elastic_loss_weight: float = 0.0
    homo_loss_weight: float = 0.1
    
    semantics_opt : bool = False
    save_sem_mask : bool = False
    
    eval_overall_gs : bool = False
    eval_pose : bool = False

    canny_opt : bool = False

    camera_smoothness_loss : bool = False

    def adjust_steps(self, factor: float):
        self.eval_steps = [int(i * factor) for i in self.eval_steps]
        self.save_steps = [int(i * factor) for i in self.save_steps]
        self.max_steps = int(self.max_steps * factor)
        self.sh_degree_interval = int(self.sh_degree_interval * factor)
        self.refine_start_iter = int(self.refine_start_iter * factor)
        self.refine_stop_iter = int(self.refine_stop_iter * factor)
        self.reset_every = int(self.reset_every * factor)
        self.refine_every = int(self.refine_every * factor)

def get_scheduler(optimizer, policy, num_epochs_fix=None, num_epochs=None):
    if policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch - num_epochs_fix) / float(num_epochs - num_epochs_fix + 1)
            return lr_l
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif policy == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, patience=2)
    else:
        return NotImplementedError('scheduler with {} is not implemented'.format(policy))
    return scheduler

def get_model_config(motion_model_name, elastic_loss_weight, homo_loss_weight,max_len):
    if motion_model_name in ['affinefield4d','transfield4d','se3field4d','scaledse3field4d']:
        model_opt = {
                'motion_model_name': motion_model_name,
                'device': "cuda",
                'motion_model_opt': {
                    'dpf_opt': {
                        'in_features':4, # 3D + t
                        'hidden_features': 128,
                        'n_hidden_layers': 2,
                        'max_lens': max_len,
                        },
                    'n_frames': max_len,
                    'homo_loss_weight': homo_loss_weight,
                    'elastic_loss_weight': elastic_loss_weight,
                    'n_iter': 1000,
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


class Runner:
    """Engine for training and testing."""

    def __init__(
        self, local_rank: int, world_rank, world_size: int, cfg: Config
    ) -> None:
        set_random_seed(42 + local_rank)

        self.cfg = cfg
        self.world_rank = world_rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.device = f"cuda:{local_rank}"

        # Where to dump results.
        os.makedirs(cfg.result_dir, exist_ok=True)

        # Setup output directories.
        self.ckpt_dir = f"{cfg.result_dir}/ckpts"
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.stats_dir = f"{cfg.result_dir}/stats"
        os.makedirs(self.stats_dir, exist_ok=True)
        self.render_dir = f"{cfg.result_dir}/renders"
        self.intermediate_result = f"{cfg.result_dir}/intermediate_result"
        os.makedirs(self.intermediate_result, exist_ok=True)
        os.makedirs(self.render_dir, exist_ok=True)
        self.traj_dir = f"{cfg.result_dir}/traj"
        os.makedirs(self.traj_dir, exist_ok=True)
        self.depth_dir = f"{cfg.result_dir}/depth"
        os.makedirs(self.depth_dir, exist_ok=True)
        self.depth_gt_dir = f"{cfg.result_dir}/depth_gt"
        os.makedirs(self.depth_gt_dir, exist_ok=True)
        self.sem_dir = f"{cfg.result_dir}/sem"
        os.makedirs(self.sem_dir, exist_ok=True)
        self.doma_render_dir = f"{cfg.result_dir}/doma_render2"
        os.makedirs(self.doma_render_dir, exist_ok=True)

        # Tensorboard
        self.writer = SummaryWriter(log_dir=f"{cfg.result_dir}/tb")

        # Load data: Training data should contain initial points and colors.
        self.parser = Parser(
            data_dir=cfg.data_dir,
            factor=cfg.data_factor,
            normalize=True,
            test_every=cfg.test_every,
            type=cfg.datatype,
        )
        self.trainset = Dataset(
            self.parser,
            split="train",
            patch_size=cfg.patch_size,
            load_depths=cfg.depth_loss,
            load_semantics=cfg.semantics_opt,
            load_canny=cfg.canny_opt
        )
        self.valset = Dataset(self.parser, split="val")
        self.scene_scale = self.parser.scene_scale * 1.1 * cfg.global_scale
        print("Scene scale:", self.scene_scale)

        # Model
        feature_dim = 32 if cfg.app_opt else None

        if cfg.pose_noise > 0.0:
            self.pose_perturb = CameraOptModule(len(self.trainset)).to(self.device)
            self.pose_perturb.random_init(cfg.pose_noise)
            if world_size > 1:
                self.pose_perturb = DDP(self.pose_perturb)

        self.app_optimizers = []
        if cfg.app_opt:
            self.app_module = AppearanceOptModule(
                len(self.trainset), feature_dim, cfg.app_embed_dim, cfg.sh_degree
            ).to(self.device)
            # initialize the last layer to be zero so that the initial output is zero.
            torch.nn.init.zeros_(self.app_module.color_head[-1].weight)
            torch.nn.init.zeros_(self.app_module.color_head[-1].bias)
            self.app_optimizers = [
                torch.optim.Adam(
                    self.app_module.embeds.parameters(),
                    lr=cfg.app_opt_lr * math.sqrt(cfg.batch_size) * 10.0,
                    weight_decay=cfg.app_opt_reg,
                ),
                torch.optim.Adam(
                    self.app_module.color_head.parameters(),
                    lr=cfg.app_opt_lr * math.sqrt(cfg.batch_size),
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

        # Viewer
        if not self.cfg.disable_viewer:
            self.server = viser.ViserServer(port=cfg.port, verbose=False)
            self.viewer = nerfview.Viewer(
                server=self.server,
                render_fn=self._viewer_render_fn,
                mode="training",
            )

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
        if self.cfg.app_opt:
            colors = self.app_module(
                features=self.splats["features"],
                embed_ids=image_ids,
                dirs=means[None, :, :] - camtoworlds[:, None, :3, 3],
                sh_degree=kwargs.pop("sh_degree", self.cfg.sh_degree),
            )
            colors = colors + self.splats["colors"]
            colors = torch.sigmoid(colors)
        else:
            colors = torch.cat([self.splats["sh0"], self.splats["shN"]], 1)  # [N, K, 3]

        rasterize_mode = "antialiased" if self.cfg.antialiased else "classic"
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
            packed=self.cfg.packed,
            absgrad=self.cfg.absgrad,
            sparse_grad=self.cfg.sparse_grad,
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
    def train_block(self, camtoworlds, Ks, width, height, sh_degree_to_use, image_ids, pixels, schedulers, 
                    camera_scheduler, camera_opt=False, points=None, depthmap_gt=None,eval_final=False,masks=None):
        if eval_final:
            max_steps_local = 1000
        elif not camera_opt:
            max_steps_local = 1000
        else:   
            max_steps_local = cfg.max_steps
        pbar = tqdm.tqdm(range(0, max_steps_local))

        for step in pbar:
            if camera_opt:
                camtoworlds_transformed = self.pose_adjust(camtoworlds, image_ids)

            renders , alphas, info = self.rasterize_splats(
                camtoworlds= camtoworlds_transformed if camera_opt else camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=sh_degree_to_use,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                image_ids=image_ids,
                render_mode="RGB+ED" if cfg.depth_loss else "RGB",
            )
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

            loss = l1loss * (1.0 - cfg.ssim_lambda) + ssimloss * cfg.ssim_lambda
            # if cfg.depth_loss:
                # depthloss = F.l1_loss(depths_0.squeeze(-1), depthmap_gt)
                # loss += depthloss

            if cfg.camera_smoothness_loss and camera_opt:
                smoothness_loss = self.calculate_speed_smoothness(camtoworlds_transformed)
                # print('smoothness loss', smoothness_loss)
                loss += (1- step/max_steps_local) * smoothness_loss

            desc = f"loss={loss.item():.3f}| " f"sh degree={sh_degree_to_use}| "
            # if cfg.depth_loss:
            #     desc += f"depth loss={depthloss.item():.6f}| "
            if cfg.camera_smoothness_loss and camera_opt:
                desc += f"smoothness loss={smoothness_loss.item():.6f}| "
            pbar.set_description(desc)
            loss.backward()

            if not camera_opt:
                self.strategy.step_post_backward(
                    params=self.splats,
                    optimizers=self.optimizers,
                    state=self.strategy_state,
                    step=step,
                    info=info,
                    packed=cfg.packed,
                )

            # Turn Gradients into Sparse Tensor before running optimizer
            if cfg.sparse_grad:
                assert cfg.packed, "Sparse gradients only work with packed mode."
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
        model_opt = get_model_config(cfg.motion_model_name, 
            cfg.elastic_loss_weight,
            cfg.homo_loss_weight,
            seq_len)
        motion_model_opt = model_opt['motion_model_opt']
        doma = MotionField(model_opt).to(device)
        doma_optimizer = torch.optim.Adam(doma.model.parameters(), lr=motion_model_opt['lr'])
        doma_scheduler = get_scheduler(doma_optimizer, policy='lambda',
                    num_epochs_fix=1,
                    num_epochs=max_steps)
        return doma, doma_optimizer, doma_scheduler

    def train(self):
        cfg = self.cfg
        device = self.device
        world_rank = self.world_rank
        world_size = self.world_size

        # Dump cfg.
        if world_rank == 0:
            with open(f"{cfg.result_dir}/cfg.json", "w") as f:
                json.dump(vars(cfg), f)

        max_steps = cfg.max_steps
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

        # for step in pbar:
        if not cfg.disable_viewer:
            while self.viewer.state.status == "paused":
                time.sleep(0.01)
            self.viewer.lock.acquire()
            tic = time.time()
        prev_cam_pose = torch.eye(4).to(device)

        for current_frame in range(start_frame, end_frame, cfg.batch_size):
            # try load camera pose, if not empty, then use it 
            block_end = min(current_frame + cfg.batch_size  , end_frame)
            print('optimizing from {} to {}'.format(current_frame, block_end-1))
            if cfg.pose_ckpt is not None and os.path.exists(cfg.pose_ckpt):
                pose_ckpt=torch.load(cfg.pose_ckpt)
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

            depth = data_block[0]['depth'].to(device)
            rgb = data_block[0]['image'].reshape(-1, 3).to(device) / 255.0 

            if cfg.semantics_opt:
                semantics_masks = torch.stack([data["mask"] for data in data_block]).to(device)
                depth = depth[semantics_masks[0].view(-1)]
                rgb = rgb[semantics_masks[0].view(-1)]
            
            self.splats, self.optimizers = create_splats_with_optimizers(
                self.parser,
                init_type=cfg.init_type,
                init_num_pts=cfg.init_num_pts,
                init_extent=cfg.init_extent,
                init_opacity=cfg.init_opa,
                init_scale=cfg.init_scale,
                scene_scale=self.scene_scale,
                sh_degree=cfg.sh_degree,
                sparse_grad=cfg.sparse_grad,
                batch_size=cfg.batch_size,
                feature_dim=32 if cfg.app_opt else None,
                device=self.device,
                world_rank=world_rank,
                world_size=world_size,
                predict_depth = depth,
                predict_rgb = rgb
            )
            self.strategy = DefaultStrategy(
                verbose=False,
                scene_scale=self.scene_scale,
                prune_opa=cfg.prune_opa,
                grow_grad2d=cfg.grow_grad2d,
                grow_scale3d=cfg.grow_scale3d,
                prune_scale3d=cfg.prune_scale3d,
                refine_start_iter=cfg.refine_start_iter,
                refine_stop_iter=cfg.refine_stop_iter,
                reset_every=cfg.reset_every,
                refine_every=cfg.refine_every,
                absgrad=cfg.absgrad,
                revised_opacity=cfg.revised_opacity,
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
                    lr=cfg.pose_opt_lr,
                    weight_decay=cfg.pose_opt_reg,
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
            if cfg.semantics_opt:
                pixels = torch.stack([data["image"]*data['mask'] for data in data_block]).to(device) / 255.0  # [b, H, W, 3]
                masks = torch.stack([data["mask"] for data in data_block]).to(device)
            else:
                pixels = torch.stack([data["image"] for data in data_block]).to(device) / 255.0
                masks = None


            num_train_rays_per_step = (
                pixels.shape[0] * pixels.shape[1] * pixels.shape[2]
            )
            image_ids = torch.Tensor([data["image_id"] for data in data_block]).int().to(device)
            if cfg.depth_loss:
                depthmap_gt = torch.stack([data["depth_map"] for data in data_block]).to(device)  # [b, M]
            if cfg.pose_noise:
                camtoworlds = self.pose_perturb(camtoworlds, image_ids)
                

            if cfg.save_sem_mask:
                for ind, pixel in enumerate(pixels):

                    # save semantic mask
 
                    if cfg.semantics_opt:
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
            sh_degree_to_use = min(cfg.max_steps // cfg.sh_degree_interval, cfg.sh_degree) 

            # train first frame
            
            renders_0, alphas_0, info_0, loss = self.train_block(camtoworlds[[0]], Ks[[0]], width, height, sh_degree_to_use, image_ids[[0]], pixels[[0]],
                                                                    schedulers,camera_schedulers,camera_opt = False,depthmap_gt=depthmap_gt[[0]],masks=None) 

            # fix the first frame and train the rest
            if current_frame != end_frame:
                renders, alphas, info, loss = self.train_block(camtoworlds[1:], Ks[1:], width, height, sh_degree_to_use, image_ids[1:], pixels[1:],
                                                                schedulers,camera_schedulers,camera_opt = True,depthmap_gt=depthmap_gt[1:],masks=masks[1:])


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
                est_cam_pose = camtoworlds
                global_cam_poses_est += [camtoworlds.cpu().detach().numpy()]
                global_cam_poses_gt += [camtoworlds_gt.cpu().detach().numpy()]
            # breakpoint()
            est_cam_pose = torch.stack([data_block[0]['camtoworld'].to(device)]+[cam_pose.to(device) for cam_pose in est_cam_pose]).to(device)

            self.eval_all(current_frame, data_block, est_cam_pose)

            self.plot_traj(np.array(global_cam_poses_est), np.array(global_cam_poses_gt), start_frame, block_end)

            # save checkpoint before updating the model
            # if block_end >= end_frame:
        if cfg.eval_overall_gs:
            self.train_gs_doma(cfg.sh_degree, global_cam_poses_est)

        if block_end >= end_frame and not os.path.exists(cfg.pose_ckpt):
        # if False:
            print("block_end", block_end)
            print('end_frame', end_frame)
            endtime_camera= time.time() 
            print("------saving checkpoint------")
            print("time for camera optimization", endtime_camera - starttime)
            mem = torch.cuda.max_memory_allocated() / 1024**3
            stats = {
                "mem": mem,
                "ellipse_time": time.time() - global_tic,
                "num_GS": len(self.splats["means"]),
                "gs_camera_time": endtime_camera - starttime,
            }
            step = current_frame
            print("Step: ", step, stats)
            with open(
                f"{self.stats_dir}/train_step{step:04d}_rank{self.world_rank}.json",
                "w",
            ) as f:
                json.dump(stats, f)
            data = {"step": step, "splats": self.splats.state_dict(),"start_frame": current_frame,"end_frame": block_end}
            if cfg.pose_opt:
                if world_size > 1:
                    data["pose_adjust"] = self.pose_adjust.module.state_dict()
                else:
                    data["pose_adjust"] = self.pose_adjust.state_dict()
            if cfg.app_opt:
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
                global_camera_poses, f"{self.traj_dir}/global_camera_poses_{start_frame}_to{block_end}_rank{self.world_rank}.pt"
            )
            
            # reinitalize the optimizer and gsplats and strategy and pose_adjust
            # del self.splats, self.optimizers, self.strategy, self.strategy_state, self.pose_adjust, self.pose_optimizers
            # if cfg.doma:
            #     del self.doma_optimizer
            torch.cuda.empty_cache()

    def train_gs_doma(self,sh_degree_to_use,global_cam_poses_est):
        eval_max_step = 5000
        pbar = tqdm.tqdm(range(0, eval_max_step))
        frame_0_depth = self.trainset[0]['depth'].to(self.device)
        frame_0_rgb = self.trainset[0]['image'].reshape(-1, 3).to(self.device) / 255.0

        self.splats, self.optimizers = create_splats_with_optimizers(
                self.parser,
                init_type=cfg.init_type,
                init_num_pts=cfg.init_num_pts,
                init_extent=cfg.init_extent,
                init_opacity=cfg.init_opa,
                init_scale=cfg.init_scale,
                scene_scale=self.scene_scale,
                sh_degree=cfg.sh_degree,
                sparse_grad=cfg.sparse_grad,
                batch_size=cfg.batch_size,
                feature_dim=32 if cfg.app_opt else None,
                device=self.device,
                world_rank=self.world_rank,
                world_size=self.world_size,
                predict_depth = frame_0_depth,
                predict_rgb = frame_0_rgb
            )

        self.strategy = DefaultStrategy(
                verbose=False,
                scene_scale=self.scene_scale,
                prune_opa=cfg.prune_opa,
                grow_grad2d=cfg.grow_grad2d,
                grow_scale3d=cfg.grow_scale3d,
                prune_scale3d=cfg.prune_scale3d,
                refine_start_iter=cfg.refine_start_iter,
                refine_stop_iter=cfg.refine_stop_iter,
                reset_every=cfg.reset_every,
                refine_every=cfg.refine_every,
                absgrad=cfg.absgrad,
                revised_opacity=cfg.revised_opacity,
        )
        self.strategy_state = self.strategy.initialize_state()
        schedulers = [
            torch.optim.lr_scheduler.ExponentialLR(
                self.optimizers["means"], gamma=0.01 ** (1.0 / eval_max_step)
            ),
        ]

        # train a first frame
        height,width = self.trainset[0]['image'].shape[0:2]
        
        if cfg.doma:
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
            # if cfg.depth_loss:
            depthmap_gt = data['depth_map'].to(self.device)
            if cfg.doma:
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
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                image_ids=image_ids,
                render_mode="RGB+ED" if cfg.depth_loss else "RGB",
                doma_mean=doma_mean if cfg.doma else None
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
            loss = l1loss * (1.0 - cfg.ssim_lambda) + ssimloss * cfg.ssim_lambda

            if cfg.depth_loss:
                depthloss = F.l1_loss(depths_0.squeeze(-1), depthmap_gt)
            # if cfg.doma:
            #     loss_homogeneous_reg = doma.model.homogenous_reg(self.splats['means'],tidx)
            #     # loss_elastic_reg = doma.model.elastic_reg(self.splats['means'],tidx)
            #     # loss += cfg.elastic_loss_weight * loss_elastic_reg + 
            #     loss += cfg.homo_loss_weight * loss_homogeneous_reg
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
                packed=cfg.packed,
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
                self.eval_all(step, self.trainset, torch.stack([torch.Tensor(cam_pose).to(self.device) for cam_pose in global_cam_poses_est]).to(self.device),doma_eval=True,doma=doma)
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
            json.dump(vars(cfg), f)
        
            


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
        global_cam_poses_est_aligned = global_cam_poses_est @ global_cam_poses_gt[0]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        est_x, est_y, est_z = [], [], []
        gt_x, gt_y, gt_z = [], [], []

        for i in range(0, end_frame-start_frame+1):
            cam_pose_est = global_cam_poses_est_aligned[i]
            cam_pose_gt = global_cam_poses_gt[i]

            est_x.append(cam_pose_est[0, 3])
            est_y.append(cam_pose_est[1, 3])
            est_z.append(cam_pose_est[2, 3])

            gt_x.append(cam_pose_gt[0, 3])
            gt_y.append(cam_pose_gt[1, 3])
            gt_z.append(cam_pose_gt[2, 3])
            # ax.scatter(cam_pose_est[0, 3], cam_pose_est[1, 3], cam_pose_est[2, 3], c='r', marker='o', rasterized = True)
            # ax.scatter(cam_pose_gt[0, 3], cam_pose_gt[1, 3], cam_pose_gt[2, 3], c='b', marker='x', rasterized = True)
        ax.plot(est_x, est_y, est_z, c='r', label='est')
        ax.plot(gt_x, gt_y, gt_z, c='b', label='gt')
        ax.legend(['est', 'gt'])
        plt.savefig(f"{self.traj_dir}/traj_{start_frame}_to_{end_frame}.png")
        # plt.savefig(f"{self.traj_dir}/traj_{start_frame}_to_{end_frame}.pdf", dpi = 200, bbox_inches='tight')


    @torch.no_grad()
    def eval_all(self, current_frame: int,data_block, est_cam_pose,doma_eval=False,doma=None):
        """Entry for evaluation."""
        print("Running evaluation...")
        cfg = self.cfg
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
            if doma_eval:
                points, scaled_factor, min_value = normalize_points(self.splats['means'])
                new_means = doma.deform(points,tidx) 
                new_means = (new_means+1)/2 * scaled_factor + min_value
                doma_mean = new_means
            # render the splats
            renders, _, _ = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                render_mode="RGB+ED" if cfg.depth_loss else "RGB",
                doma_mean=doma_mean if doma_eval else None
            )  # [1, H, W, 3]  

            if renders.shape[-1] == 4:
                colors, depths_0 = renders[..., 0:3], renders[..., 3:4]
            else:
                colors, depths_0 = renders, None
            colors = torch.clamp(colors, 0.0, 1.0)
            torch.cuda.synchronize()

            # if world_rank == 0:
                # write images

            canvas = torch.cat([pixels, colors], dim=2).squeeze(0).cpu().numpy()
            if doma_eval:
                imageio.imwrite(
                    f"{self.doma_render_dir}/{data['image_id']:04d}_doma.png",
                    (canvas * 255).astype(np.uint8),
                )
                print(f"Image saved to {self.doma_render_dir}/{data['image_id']:04d}_doma.png")
            else:
                imageio.imwrite(
                    f"{self.render_dir}/val_{current_frame}_{data['image_id']:04d}.png",
                    (canvas * 255).astype(np.uint8),
                )
                print(f"Image saved to {self.render_dir}/val_{data['image_id']:04d}.png")
            
            if cfg.depth_loss:
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

    @torch.no_grad()
    def render_traj(self, step: int):
        """Entry for trajectory rendering."""
        print("Running trajectory rendering...")
        cfg = self.cfg
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
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
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
        video_dir = f"{cfg.result_dir}/videos"
        os.makedirs(video_dir, exist_ok=True)
        writer = imageio.get_writer(f"{video_dir}/traj_{step}.mp4", fps=30)
        for canvas in canvas_all:
            writer.append_data(canvas)
        writer.close()
        print(f"Video saved to {video_dir}/traj_{step}.mp4")

    @torch.no_grad()
    def _viewer_render_fn(
        self, camera_state: nerfview.CameraState, img_wh: Tuple[int, int]
    ):
        """Callable function for the viewer."""
        W, H = img_wh
        c2w = camera_state.c2w
        K = camera_state.get_K(img_wh)
        c2w = torch.from_numpy(c2w).float().to(self.device)
        K = torch.from_numpy(K).float().to(self.device)

        render_colors, _, _ = self.rasterize_splats(
            camtoworlds=c2w[None],
            Ks=K[None],
            width=W,
            height=H,
            sh_degree=self.cfg.sh_degree,  # active all SH degrees
            radius_clip=3.0,  # skip GSs that have small image radius (in pixels)
        )  # [1, H, W, 3]
        return render_colors[0].cpu().numpy()
    
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


def main(local_rank: int, world_rank, world_size: int, cfg: Config):
    if world_size > 1 and not cfg.disable_viewer:
        cfg.disable_viewer = True
        if world_rank == 0:
            print("Viewer is disabled in distributed training.")
    global starttime
    starttime = time.time()
    # make starttime global

    runner = Runner(local_rank, world_rank, world_size, cfg)


    if cfg.pose_ckpt is not None and cfg.eval_pose:
        runner.eval_pose_droid(cfg.pose_ckpt,cfg.disable_plot)
    else:
        runner.train()

    #eval camera pose
    
    endtime = time.time()
    print("Total time: ", endtime - starttime)
    if not cfg.disable_viewer:
        print("Viewer running... Ctrl+C to exit.")
        time.sleep(1000000)


if __name__ == "__main__":
    """
    Usage:

    ```bash
    # Single GPU training
    CUDA_VISIBLE_DEVICES=0 python simple_trainer.py

    # Distributed training on 4 GPUs: Effectively 4x batch size so run 4x less steps.
    CUDA_VISIBLE_DEVICES=0,1,2,3 python simple_trainer.py --steps_scaler 0.25

    """

    cfg = tyro.cli(Config)
    cfg.adjust_steps(cfg.steps_scaler)
    cli(main, cfg, verbose=True)

# CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1 TORCH_USE_CUDA_DSA=1 python examples/cf3dgs.py --eval_steps -1 --disable_viewer --data_factor 1  --data_dir /home/yuxinyao/gsplat/data/co3d/ball/242_25689_51045 --result_dir results/benchmark/ball/ --max_steps 1000 --batch_size 3 --pose_opt --init_type predict --depth_loss --datatype "co3d" --pose_ckpt /home/yuxinyao/gsplat/results/benchmark/ball/traj/global_camera_poses_20_to30_rank0.pt