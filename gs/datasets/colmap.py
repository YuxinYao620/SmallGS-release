import os
from typing import Any, Dict, List, Optional

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
from pycolmap import SceneManager

import json
import gzip
from collections import defaultdict, OrderedDict

from kornia.geometry.depth import depth_to_3d, depth_to_normals
from pytorch3d.utils import opencv_from_cameras_projection
from pytorch3d.renderer import PerspectiveCameras
from pytorch3d.transforms import matrix_to_quaternion, quaternion_to_matrix
import math
import pickle


from gs.Marigold.marigold import MarigoldPipeline

from .normalize import (
    align_principle_axes,
    similarity_from_cameras,
    transform_cameras,
    transform_points,
)

from PIL import Image

import copy

import joblib
import glob

from typing import Sequence
from torchvision import transforms
from einops import rearrange
from torchvision.transforms import Normalize
from torchvision.transforms.functional import resize
from torchvision.utils import save_image
from torchvision.io.image import read_image, ImageReadMode
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

def cartesian_to_spherical_batch(coords, target_point = None):
    """
    Convert a batch of Cartesian coordinates to spherical coordinates.
    
    Args:
        coords (np.ndarray): An array of shape (N, 3) where N is the number of points,
                             and each point is represented by (x, y, z).
                             
    Returns:
        np.ndarray: An array of shape (N, 3) where each point is represented by (phi, theta, radius).
    """
    if target_point is None:
        x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    else:
        x, y, z = coords[:, 0] - target_point[0], coords[:, 1] - target_point[1], coords[:, 2] - target_point[2]
    radius = torch.sqrt(x**2 + y**2 + z**2)
    # if radius == 0:
    #     radius += 1
    phi = torch.atan2(y, x)
    theta = torch.acos(z / radius)
    return torch.stack([phi, theta, radius], dim=-1)

def _get_rel_paths(path_dir: str) -> List[str]:
    """Recursively get relative paths of files in a directory."""
    paths = []
    for dp, dn, fn in os.walk(path_dir):
        for f in fn:
            paths.append(os.path.relpath(os.path.join(dp, f), path_dir))
    return paths

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))
def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

def scale_aware_icp(src, dst, max_iterations=100, tolerance=1e-8, scale = None):
    """
    Perform Scale-Aware ICP to align src to dst. Both src and dst can have different numbers of points.
    
    Args:
        src (numpy.ndarray): Source point cloud of shape [N, 3].
        dst (numpy.ndarray): Target point cloud of shape [M, 3].
        max_iterations (int): Maximum number of iterations for ICP.
        tolerance (float): Convergence tolerance.
        
    Returns:
        scale (float): Scaling factor.
        rotation (torch.Tensor): Rotation matrix of shape [3, 3].
        translation (torch.Tensor): Translation vector of shape [3].
    """
    # Convert inputs to PyTorch tensors
    # src = torch.tensor(src, dtype=torch.float32)
    # dst = torch.tensor(dst, dtype=torch.float32)

    # Initialize scale, rotation, and translation
    scale = 1.0 if scale is None else scale
    rotation = torch.eye(3, dtype=torch.float32).to(src.device)
    translation = torch.zeros(3, dtype=torch.float32).to(src.device)

    # Compute centroids
    src_centroid = src.mean(dim=0)
    dst_centroid = dst.mean(dim=0)

    # Center the point clouds
    src_centered = src - src_centroid
    dst_centered = dst - dst_centroid

    prev_error = float('inf')

    for i in range(max_iterations):
        # Nearest neighbor search (for each point in src, find closest in dst)
        distances = torch.cdist(src_centered @ rotation.T * scale, dst_centered)  # Pairwise distances
        nearest_indices = torch.argmin(distances, dim=1)  # Closest points in dst

        # Matched points
        matched_dst = dst_centered[nearest_indices]

        # Compute scale
        src_norm = src_centered.norm(dim=1)
        dst_norm = matched_dst.norm(dim=1)
        scale = (dst_norm / src_norm).mean()

        # Compute rotation using SVD
        H = src_centered.T @ matched_dst / scale
        U, _, Vt = torch.linalg.svd(H)
        rotation = U @ Vt

        # Correct for reflection
        if torch.det(rotation) < 0:
            Vt[-1, :] *= -1
            rotation = U @ Vt

        # Compute translation
        translation = dst_centroid - (src_centroid @ rotation.T) * scale

        # Apply transformation
        src_transformed = (src_centered @ rotation.T) * scale + translation

        # Check convergence (mean error)
        error = ((src_transformed - matched_dst) ** 2).sum(dim=1).mean().sqrt()
        if abs(prev_error - error) < tolerance:
            break
        prev_error = error
    
    scale_aligned = (src @ rotation.T) * scale + translation

    return scale, rotation, translation, scale_aligned


class CO3D:
    def __init__(self,data_dir: str):
        self.data_dir = data_dir
        data_info = data_dir.split('/')
        seq_name = data_info[-1]
        category = data_info[-2]
        dataroot = data_dir.split(category)[0]
        sequences = defaultdict(list)
        dataset = json.loads(gzip.GzipFile(os.path.join(dataroot, category,
                                                            "frame_annotations.jgz"), "rb").read().decode("utf8"))
            
        for data in dataset:
            sequences[data["sequence_name"]].append(data)
        seq_data = sequences[seq_name]
        self.data = seq_data
        


class Parser:
    """COLMAP parser."""

    def __init__(
        self,
        data_dir: str,
        factor: int = 1,
        normalize: bool = False,
        test_every: int = 8,
        type: str = "colmap",
        monst3r_dir = None,
        monst3r_cam_pose = None,
        monst3r_cam_intr = None,
        monst3r_use_cam = True,
        eval_pkl = None
    ):
        self.data_dir = data_dir
        self.factor = factor
        self.normalize = normalize
        self.test_every = test_every

        self.spec_data_path = None
        self.spec_cam_path = None
        self.semantic_path = None
        self.depth_path = None

        if type == "monst3r":

            self.data_dir = data_dir 
            self.semantic_path = os.path.join(monst3r_dir, "semantics")
            self.depth_path = os.path.join(monst3r_dir, "depths")
            self.dino_path = os.path.join(monst3r_dir, "dino_embeddings")
            os.makedirs(self.semantic_path, exist_ok=True)
            os.makedirs(self.depth_path, exist_ok=True)
            os.makedirs(self.dino_path, exist_ok=True)
            if data_dir is not list:
                image_names = glob.glob(os.path.join(monst3r_dir, "frame*.png"))
                self.image_names = [name.split('/')[-1] for name in image_names]
                self.image_paths = [os.path.join(monst3r_dir, f) for f in self.image_names]
                seq_len = len(self.image_names)
            else:
                image_names = data_dir
                self.image_names = [name.split('/')[-1] for name in image_names]
                self.image_paths = data_dir
            if monst3r_use_cam or os.path.exists(os.path.join(monst3r_dir,"pred_traj_monst3r_for_refine.txt")):
                try:
                    monst3r_cam_pose_path = os.path.join(monst3r_dir,"pred_traj_monst3r_for_refine.txt")
                    
                except:
                    monst3r_cam_pose_path = os.path.join(monst3r_dir,"pred_traj_monst3r.txt") 

                if monst3r_cam_pose is None:
                    with open(monst3r_cam_pose_path) as f:
                        monst3r_cam_pose = f.readlines()
                        monst3r_cam_pose = torch.Tensor(np.array([list(map(float,pose.split())) for pose in monst3r_cam_pose]))
                seq_len = len(monst3r_cam_pose)
                # quaternion to rotation matrix
                rot_mat = quaternion_to_matrix(monst3r_cam_pose[:,4:8])
                t_mat = monst3r_cam_pose[:,1:4]

                # compose to camera pose
                camera_pose = torch.zeros((seq_len,4,4))
                camera_pose[:,:3,:3] = rot_mat
                camera_pose[:,:3,3] = t_mat
                camera_pose[:,3,3] = 1
                
                self.camtoworlds = camera_pose
                self.camera_ids = [0] * seq_len
            else:
                camera_pose = torch.eye(4).unsqueeze(0).repeat(seq_len,1,1)
                self.camtoworlds = camera_pose
                self.camera_ids = [0] * seq_len
            if monst3r_cam_intr is None:
                intr_mat_path = os.path.join(monst3r_dir,"pred_intrinsics.txt")
                with open(intr_mat_path) as f:
                    monst3r_cam_intr = f.readlines()
                    monst3r_cam_intr = torch.Tensor(np.array([list(map(float,intr.split())) for intr in monst3r_cam_intr]))
            monst3r_cam_intr = monst3r_cam_intr[0].reshape(3,3).cpu().numpy()

            self.Ks_dict = {0: monst3r_cam_intr}

            self.camtype = "perspective"
            params = np.empty(0, dtype=np.float32) #default SIMPLE_PINHOLE for non-colmap dataset
            self.params_dict = {0: params}
            self.imsize_dict = {0: (self.Ks_dict[0][0, 2] * 2, self.Ks_dict[0][1, 2] * 2)}
            self.points = None
            self.points_err = None
            self.points_rgb = None
            self.point_indices = None
            self.transform = np.eye(4)

            camera_locations = camera_pose[:, :3, 3]
            scene_center = torch.mean(camera_locations, axis=0)
            # dists = np.linalg.norm(camera_locations - scene_center, axis=1)
            dists = torch.norm(camera_locations - scene_center, dim=1)
            self.scene_scale = torch.max(dists)
            if self.scene_scale < 1e-6:
                self.scene_scale = 1.0
            self.factor = factor

            # 
            self.scene_scale = 1.0
            return
        # if type == "eval_dataset":
        #     self.image_paths = data_dir
            



        if type == "colmap":
            colmap_dir = os.path.join(data_dir, "sparse/0/")
            if not os.path.exists(colmap_dir):
                colmap_dir = os.path.join(data_dir, "sparse")
            # assert os.path.exists(
            #     colmap_dir
            # ), f"COLMAP directory {colmap_dir} does not exist."

            manager = SceneManager(colmap_dir)
            manager.load_cameras()
            manager.load_images()
            manager.load_points3D()

            # Extract extrinsic matrices in world-to-camera format.
            imdata = manager.images
            # let images be sorted by image name
            imdata = OrderedDict(sorted(imdata.items(), key=lambda x: x[1].name))
            w2c_mats = []
            camera_ids = []
            Ks_dict = dict()
            params_dict = dict()
            imsize_dict = dict()  # width, height
            bottom = np.array([0, 0, 0, 1]).reshape(1, 4)
            for k in imdata:
                im = imdata[k]
                rot = im.R()
                trans = im.tvec.reshape(3, 1)
                w2c = np.concatenate([np.concatenate([rot, trans], 1), bottom], axis=0)
                w2c_mats.append(w2c)

                # support different camera intrinsics
                camera_id = im.camera_id
                camera_ids.append(camera_id)

                # Get camera intrinsics.
                cam = manager.cameras[camera_id]
                fx, fy, cx, cy = cam.fx, cam.fy, cam.cx, cam.cy
                K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
                # K[:2, :] /= factor
                Ks_dict[camera_id] = K

                # Get distortion parameters.
                type_ = cam.camera_type
                if type_ == 0 or type_ == "SIMPLE_PINHOLE":
                    params = np.empty(0, dtype=np.float32)
                    camtype = "perspective"
                elif type_ == 1 or type_ == "PINHOLE":
                    params = np.empty(0, dtype=np.float32)
                    camtype = "perspective"
                if type_ == 2 or type_ == "SIMPLE_RADIAL":
                    params = np.array([cam.k1], dtype=np.float32)
                    camtype = "perspective"
                elif type_ == 3 or type_ == "RADIAL":
                    params = np.array([cam.k1, cam.k2, 0.0, 0.0], dtype=np.float32)
                    camtype = "perspective"
                elif type_ == 4 or type_ == "OPENCV":
                    params = np.array([cam.k1, cam.k2, cam.p1, cam.p2], dtype=np.float32)
                    camtype = "perspective"
                elif type_ == 5 or type_ == "OPENCV_FISHEYE":
                    params = np.array([cam.k1, cam.k2, cam.k3, cam.k4], dtype=np.float32)
                    camtype = "fisheye"
                assert (
                    camtype == "perspective"
                ), f"Only support perspective camera model, got {type_}"

                params_dict[camera_id] = params

                # image size
                imsize_dict[camera_id] = (cam.width // factor, cam.height // factor)

            print(
                f"[Parser] {len(imdata)} images, taken by {len(set(camera_ids))} cameras."
            )

            if len(imdata) == 0:
                raise ValueError("No images found in COLMAP.")
            if not (type_ == 0 or type_ == 1):
                print(f"Warning: COLMAP Camera is not PINHOLE. Images have distortion.")

            w2c_mats = np.stack(w2c_mats, axis=0)

            # Convert extrinsics to camera-to-world.
            camtoworlds = np.linalg.inv(w2c_mats)

            # Image names from COLMAP. No need for permuting the poses according to
            # image names anymore.
            image_names = [imdata[k].name for k in imdata]

            # Previous Nerf results were generated with images sorted by filename,
            # ensure metrics are reported on the same test set.
            inds = np.argsort(image_names)
            image_names = [image_names[i] for i in inds]
            camtoworlds = camtoworlds[inds]
            camera_ids = [camera_ids[i] for i in inds]

            # Load images.
            if factor > 1:
                image_dir_suffix = f"_{factor}"
            else:
                image_dir_suffix = ""
            colmap_image_dir = os.path.join(data_dir, "images")
            image_dir = os.path.join(data_dir, "images" + image_dir_suffix)
            for d in [image_dir, colmap_image_dir]:
                if not os.path.exists(d):
                    raise ValueError(f"Image folder {d} does not exist.")

            # Downsampled images may have different names vs images used for COLMAP,
            # so we need to map between the two sorted lists of files.
            colmap_files = sorted(_get_rel_paths(colmap_image_dir))
            image_files = sorted(_get_rel_paths(image_dir))
            colmap_to_image = dict(zip(colmap_files, image_files))
            try:
                image_paths = [os.path.join(image_dir, colmap_to_image[f]) for f in image_names]
            except KeyError as e:
                image_names = [f.split('/')[-1] for f in image_names]
                image_paths = [os.path.join(image_dir, f) for f in image_names]

            # 3D points and {image_name -> [point_idx]}
            points = manager.points3D.astype(np.float32)
            points_err = manager.point3D_errors.astype(np.float32)
            points_rgb = manager.point3D_colors.astype(np.uint8)
            point_indices = dict()

            image_id_to_name = {v: k for k, v in manager.name_to_image_id.items()}
            for point_id, data in manager.point3D_id_to_images.items():
                for image_id, _ in data:
                    image_name = image_id_to_name[image_id]
                    point_idx = manager.point3D_id_to_point3D_idx[point_id]
                    point_indices.setdefault(image_name, []).append(point_idx)
            point_indices = {
                k: np.array(v).astype(np.int32) for k, v in point_indices.items()
            }

            # Normalize the world space.
            if normalize:
                T1 = similarity_from_cameras(camtoworlds)
                camtoworlds = transform_cameras(T1, camtoworlds)
                points = transform_points(T1, points)

                T2 = align_principle_axes(points)
                camtoworlds = transform_cameras(T2, camtoworlds)
                points = transform_points(T2, points)

                transform = T2 @ T1
            else:
                transform = np.eye(4)

            self.image_names = image_names  # List[str], (num_images,)
            self.image_paths = image_paths  # List[str], (num_images,)
            self.camtoworlds = camtoworlds  # np.ndarray, (num_images, 4, 4)
            self.camera_ids = camera_ids  # List[int], (num_images,)
            self.Ks_dict = Ks_dict  # Dict of camera_id -> K
            self.params_dict = params_dict  # Dict of camera_id -> params
            self.imsize_dict = imsize_dict  # Dict of camera_id -> (width, height)
            self.points = points  # np.ndarray, (num_points, 3)
            self.points_err = points_err  # np.ndarray, (num_points,)
            self.points_rgb = points_rgb  # np.ndarray, (num_points, 3)
            self.point_indices = point_indices  # Dict[str, np.ndarray], image_name -> [M,]
            self.transform = transform  # np.ndarray, (4, 4)

        elif type == "co3d":
            def load_camera(data, scale=1.0):
                """
                Load a camera from a CO3D annotation.
                """

                principal_point = torch.tensor(
                    data["viewpoint"]["principal_point"], dtype=torch.float)
                focal_length = torch.tensor(
                    data["viewpoint"]["focal_length"], dtype=torch.float)
                half_image_size_wh_orig = (
                    torch.tensor(
                        list(reversed(data["image"]["size"])), dtype=torch.float) / 2.0
                )
                format_ = data["viewpoint"]["intrinsics_format"]
                if format_.lower() == "ndc_norm_image_bounds":
                    # this is e.g. currently used in CO3D for storing intrinsics
                    rescale = half_image_size_wh_orig
                elif format_.lower() == "ndc_isotropic":
                    rescale = half_image_size_wh_orig.min()
                else:
                    raise ValueError(f"Unknown intrinsics format: {format}")

                principal_point_px = half_image_size_wh_orig - principal_point * rescale
                focal_length_px = focal_length * rescale

                # now, convert from pixels to PyTorch3D v0.5+ NDC convention
                out_size = list(reversed(data["image"]["size"]))

                half_image_size_output = torch.tensor(
                    out_size, dtype=torch.float) / 2.0
                half_min_image_size_output = half_image_size_output.min()

                # rescaled principal point and focal length in ndc
                principal_point = (
                    half_image_size_output - principal_point_px * scale
                ) / half_min_image_size_output
                focal_length = focal_length_px * scale / half_min_image_size_output

                camera = PerspectiveCameras(
                    focal_length=focal_length[None],
                    principal_point=principal_point[None],
                    R=torch.tensor(data["viewpoint"]["R"], dtype=torch.float)[None],
                    T=torch.tensor(data["viewpoint"]["T"], dtype=torch.float)[None],
                )

                img_size = torch.tensor(data["image"]["size"], dtype=torch.float)[None]
                R, t, intr_mat = opencv_from_cameras_projection(camera, img_size)
                FoVy = focal2fov(intr_mat[0, 1, 1], img_size[0, 0])
                FoVx = focal2fov(intr_mat[0, 0, 0], img_size[0, 1])

                return R[0].numpy(), t[0].numpy(), FoVx, FoVy, intr_mat[0].numpy()
            
            sequences = defaultdict(list)
            data_info = data_dir.split('/')
            seq_name = data_info[-1]
            category = data_info[-2]
            data_root = data_dir.split(category)[0]
            dataset = json.loads(gzip.GzipFile(os.path.join(data_root, category,
                                                            "frame_annotations.jgz"), "rb").read().decode("utf8"))
            for data in dataset:
                sequences[data["sequence_name"]].append(data)
            seq_data = sequences[seq_name]
            co3d_data = seq_data
            seq_len = len(co3d_data)


            camtoworlds = np.zeros((seq_len, 4, 4))
            for seq_data_i in seq_data:
                R, t, FoVx, FoVy, intr_mat = load_camera(seq_data_i)
                # index = seq_data["frame_index"]
                index = seq_data_i["frame_number"]-1
                camtoworlds[index][:3, :3] = R
                camtoworlds[index][:3, 3] = t
                camtoworlds[index][3, 3] = 1
            
            self.image_names = [f"frame{i:06d}.jpg" for i in range(1,seq_len+1)]
            self.image_paths = [os.path.join(data_dir,"images", f) for f in self.image_names]
            self.camtoworlds = camtoworlds
            self.camera_ids = [0] * seq_len
            self.Ks_dict = {0: intr_mat}
            self.camtype = "perspective"
            params = np.empty(0, dtype=np.float32) #default SIMPLE_PINHOLE for non-colmap dataset
            self.params_dict = {0: params}
            self.imsize_dict = {0: (self.Ks_dict[0][0, 2] * 2, self.Ks_dict[0][1, 2] * 2)}
            self.points = None
            self.points_err = None
            self.points_rgb = None
            self.point_indices = None
            self.transform = np.eye(4)
            # no distortion
        elif type == "custom":
            # if data_dir is not list:
            if not isinstance(data_dir, list):
                data_info = data_dir.split('/')
                seq_name = data_info[-1]
                # image_names = [name for name in sorted(os.listdir(os.path.join(data_dir)))]
                image_names = [os.path.basename(name)  for name in sorted(glob.glob(os.path.join(data_dir)+"*.png"))]
                seq_len = len(image_names)
                # find one image to get the size
                image = imageio.imread(os.path.join(data_dir,image_names[0]))[..., :3]
            else:
                data_info = data_dir[0].split('/')
                seq_name = data_info[-1]
                image_names = [name.split('/')[-1] for name in data_dir]
                seq_len = len(data_dir)
                image = imageio.imread(data_dir[0])[..., :3]
                data_dir = os.path.dirname(data_dir[0])

            # else:
            #     data_dir = os.path.dirname(data_dir[0])
            #     data_info = data_dir.split('/')
            #     seq_name = data_info[-1]
            #     image_names = [name.split('/')[-1] for name in image_names]
            #     seq_len = len(image_names)
            #     image = imageio.imread(data_dir[0])[..., :3]
            # image_names = [name for name in sorted(os.listdir(os.path.join(data_dir)),key = lambda x: int(x.split(".")[-2].split("_")[-1]))]

            height, width = image.shape[:2]
            # camera intrinsics
            if os.path.exists(os.path.join(data_dir, "camcalib","spec.pkl")):
                self.spec_data_path = os.path.join(data_dir, "spec_results","spec.pkl")
                self.spec_cam_path = os.path.join(data_dir, "camcalib","spec.pkl")
            if False:
                from pare.utils.geometry import batch_euler2matrix

                self.spec_data_path = os.path.join(data_dir, "spec_results","spec.pkl")
                self.spec_cam_path = os.path.join(data_dir, "camcalib","spec.pkl")
                print("!!!using spec camera")
                def read_cam_params(output_path, orig_shape):
                    # These are predicted camera parameters
                    # cite: SPEC: https://github.com/mkocabas/SPEC

                    pred_cam_params = joblib.load(output_path)

                    cam_pitch = pred_cam_params['pitch'].item()
                    cam_roll = pred_cam_params['roll'].item()

                    cam_vfov = pred_cam_params['vfov'].item()

                    cam_focal_length = pred_cam_params['f_pix']

                    cam_rotmat = batch_euler2matrix(torch.tensor([[cam_pitch, 0., cam_roll]]).float())[0]

                    pred_cam_int = torch.zeros(3, 3)

                    cx, cy = orig_shape[1] / 2, orig_shape[0] / 2

                    pred_cam_int[0, 0] = cam_focal_length
                    pred_cam_int[1, 1] = cam_focal_length

                    pred_cam_int[:-1, -1] = torch.tensor([cx, cy])

                    cam_int = pred_cam_int.float()

                    return cam_rotmat, cam_int, cam_vfov, cam_pitch, cam_roll, cam_focal_length
                # orig_height, orig_width = im.height, im.width
                cam_rotmat, cam_int, cam_vfov, cam_pitch, cam_roll, cam_focal_length = read_cam_params(os.path.join(data_dir, "camcalib","spec.pkl"), (height, width))
                print("parser: cam_int",cam_int)
                fx, fy, cx, cy = cam_int[0, 0], cam_int[1, 1], cam_int[0, 2], cam_int[1, 2]
                intr_mat = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
                # if factor > 1:
                #     intr_mat[:2, :] /= factor
                print('parser: intr_mat',intr_mat)

            else:
                # use some hardcoded values for custom data
                fov = 79.0
                # fov = 120.0
                FoVx = fov * math.pi / 180.0
                intr_mat = np.eye(3)
                intr_mat[0, 0] = fov2focal(FoVx, width)
                intr_mat[1, 1] = fov2focal(FoVx, width)
                intr_mat[0, 2] = width / 2
                intr_mat[1, 2] = height / 2
                print("intr_mat",intr_mat)


            # create camtoworlds, no ground truth so just identity matrix
            identity_matrix = np.eye(4)
            camtoworlds = np.repeat(identity_matrix[np.newaxis, :, :], seq_len, axis=0)
            
            self.image_names = image_names
            # self.image_paths = [os.path.join(data_dir,"images", f) for f in self.image_names]
            self.camtoworlds = camtoworlds
            self.camera_ids = [0] * seq_len
            self.Ks_dict = {0: intr_mat}
            self.camtype = "perspective"
            params = np.empty(0, dtype=np.float32) #default SIMPLE_PINHOLE for non-colmap dataset
            self.params_dict = {0: params}
            self.imsize_dict = {0: (self.Ks_dict[0][0, 2] * 2, self.Ks_dict[0][1, 2] * 2)}
            self.points = None
            self.points_err = None
            self.points_rgb = None
            self.point_indices = None
            self.transform = np.eye(4)

            self.semantic_path = os.path.join(monst3r_dir,"semantics_langsam")
            self.depth_path = os.path.join(monst3r_dir,"depths_marigold")
            if os.path.exists(os.path.join(data_dir,self.image_names[0])):
                self.image_paths = [os.path.join(data_dir, f) for f in self.image_names]
            else:
                self.image_paths = data_dir

        
        self.semantic_path = os.path.join(data_dir, "semantics") if self.semantic_path is None else self.semantic_path
        self.depth_path = os.path.join(data_dir, "depths") if self.depth_path is None else self.depth_path
        self.dino_path = os.path.join(data_dir, "dino_embeddings")
        os.makedirs(self.semantic_path, exist_ok=True)
        os.makedirs(self.depth_path, exist_ok=True)
        os.makedirs(self.dino_path, exist_ok=True)

        self.mapx_dict = dict()
        self.mapy_dict = dict()
        self.roi_undist_dict = dict()
        for camera_id in self.params_dict.keys():
            params = self.params_dict[camera_id]
            if len(params) == 0:
                continue  # no distortion
            assert camera_id in self.Ks_dict, f"Missing K for camera {camera_id}"
            assert (
                camera_id in self.params_dict
            ), f"Missing params for camera {camera_id}"
            K = self.Ks_dict[camera_id]
            width, height = self.imsize_dict[camera_id]
            K_undist, roi_undist = cv2.getOptimalNewCameraMatrix(
                K, params, (width, height), 0
            )
            print("K_undist",K_undist)
            mapx, mapy = cv2.initUndistortRectifyMap(
                K, params, None, K_undist, (width, height), cv2.CV_32FC1
            )
            self.Ks_dict[camera_id] = K_undist
            self.mapx_dict[camera_id] = mapx
            self.mapy_dict[camera_id] = mapy
            self.roi_undist_dict[camera_id] = roi_undist

        # size of the scene measured by cameras
        camera_locations = camtoworlds[:, :3, 3]
        scene_center = np.mean(camera_locations, axis=0)
        dists = np.linalg.norm(camera_locations - scene_center, axis=1)
        self.scene_scale = np.max(dists) 
        if self.scene_scale < 1e-6:
            self.scene_scale = 1.0
        self.factor = factor



class Dataset:
    """A simple dataset class."""

    def __init__(
        self,
        parser: Parser,
        split: str = "train",
        patch_size: Optional[int] = None,
        load_depths: bool = False,
        load_semantics: bool = False,
        load_canny: bool = False,
        # background_color = None
        datatype = "custom",
        dino = False,
        opt = None
    ):
        self.parser = parser
        self.split = split
        self.patch_size = patch_size
        self.load_depths = load_depths
        self.load_semantics = load_semantics
        self.load_canny = load_canny
        self.depth_type = "marigold"
        indices = np.arange(len(self.parser.image_names))
        self.camto_world = np.eye(4,4).astype(np.float32).reshape(1,4,4).repeat(len(indices),0)
        self.angle = np.zeros((len(indices),3))
        self.indices = indices
        self.gt_image_indices = copy.deepcopy(self.indices)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.depth_list = OrderedDict()
        self.depthmap_list = OrderedDict()
        self.camtoworlds_gt = None
        self.datatype = datatype
        self.dino = dino
        self.opt = opt
        # self.dino_emb_list = OrderedDict()
        # self.background_color = background_color

        # check whether depth map is already generated
        if os.path.exists(self.parser.depth_path) and len(os.listdir(self.parser.depth_path)) > 1:
            print("self.parser.depth_path",self.parser.depth_path)
            pass
        else:
            pipe: MarigoldPipeline = MarigoldPipeline.from_pretrained("prs-eth/marigold-lcm-v1-0", variant=None, torch_dtype=torch.float32)
            if self.depth_type == "marigold":
                self.pipe = pipe.to(self.device)
            else:
                midas = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid")
                midas.to(self.device)
                midas.eval()
                midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
                self.depth_transforms = midas_transforms.dpt_transform
                self.depth_model = midas
        if os.path.exists(self.parser.semantic_path) and len(os.listdir(self.parser.semantic_path)) > 0:
            pass
        else:
            breakpoint()
        if self.split == "val":
            self.indices = self.indices[:: self.parser.test_every]
        # if self.dino:
        #     if os.path.exists(self.parser.dino_path) and len(os.listdir(self.parser.dino_path)) > 0:
        #         pass
        #     else:
        #         # backbone_name= f"dinov2_vits14"
        #         # torch.hub.set_dir("/scratch/yy561/.cache/hub")
        #         # backbone_model = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model=backbone_name, )
        #         # backbone_model.eval()
        #         # backbone_model.to(self.device)
        #         # self.dino_model = backbone_model
                

    def __len__(self):
        return len(self.indices)
    
    def predict_semantics(self, image,text_prompt="dynamic"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        masks, boxes, phrases, logits = self.semantic_model.predict(Image.fromarray(image), text_prompt)
        print("masks",masks)
        if masks.shape[0]> 1:
            masks = masks[0].unsqueeze(0)
        mask_return = (~masks).long().to(device)
        return mask_return
    
    # def get_random_views(self, ):
    def set_camera_to_world(self, camtoworlds):
        self.camto_world = camtoworlds

    def make_normalize_transform(self,
        mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
        std: Sequence[float] = IMAGENET_DEFAULT_STD,
    ) -> transforms.Normalize:
        return transforms.Normalize(mean=mean, std=std)

    def make_classification_eval_transform(self,
        *,
        resize_size: int = 256,
        interpolation=transforms.InterpolationMode.BICUBIC,
        crop_size: int = 224,
        mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
        std: Sequence[float] = IMAGENET_DEFAULT_STD,
    ) -> transforms.Compose:
        transforms_list = [
            transforms.Resize(resize_size, interpolation=interpolation),
            transforms.CenterCrop(crop_size),
            self.make_normalize_transform(mean=mean, std=std),
        ]
        return transforms.Compose(transforms_list)
    def get_dino_features(self, image):
        transform = self.make_classification_eval_transform()
        test_image = torch.Tensor(image).permute(2, 0, 1).float() / 255.0
        test_image = transform(test_image)
        embeddings = self.dino_model(test_image.unsqueeze(0).cuda())
        return embeddings
    
    def minmax_norm(self,x):
        """Min-max normalization"""
        return (x - x.min(0).values) / (x.max(0).values - x.min(0).values)

    def get_dino_feature_map(self,image,H = 518,W = 518, rank = 6):
        backbone_name= f"dinov2_vits14"
        torch.hub.set_dir("/scratch/yy561/.cache/hub")
        backbone_model = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model=backbone_name, )
        backbone_model.eval()
        backbone_model.to(self.device)
        dino_model = backbone_model

        originalH, originalW = image.shape[-3], image.shape[-2]
        if len(image.shape) == 4:
            batch = image.shape[0]
        else:
            batch = 0
        if not isinstance(image, torch.Tensor):
            img = torch.Tensor(image).permute(2, 0, 1).float() / 255.0
        else:
            img = image.permute(0, 3, 1, 2).float() 
        img = resize(img, (H,W))
        # resize image to 224x224 with np
        # img = cv2.resize(image, (H,W))
        # normalize image with np
        # norm = (img-IMAGENET_DEFAULT_MEAN)/IMAGENET_DEFAULT_STD
        norm = Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)(img)
        
        I_norm = norm.unsqueeze(0).to(self.device) if len(norm.shape) == 3 else norm.to(self.device)
        features = dino_model.forward_features(I_norm)
        E_patch = features["x_norm_patchtokens"]
        E_patch_norm = rearrange(E_patch, "B L E -> (B L) E")
        _, _, V = torch.pca_lowrank(E_patch_norm, q=rank)
        E_pca = torch.matmul(E_patch_norm, V[:, :rank])
        E_pca = self.minmax_norm(E_pca)
        B, L, _ = E_patch.shape
        Z = B * L

        I_draw = E_pca
        I_draw = rearrange(I_draw, "(B L) C -> B L C", B=B)

        I_draw = rearrange(I_draw, "B (h w) C -> B h w C", h=H//14, w=W//14)

        if batch == 0:
            I_draw = I_draw.squeeze(0)
            image_1_pca = rearrange(I_draw, "H W C -> C H W")

            image_1_pca = resize(image_1_pca, (originalH,originalW))
            # image_1_pca = image_1_pca.permute(1, 2, 0).detach().cpu().numpy()
            image_1_pca = image_1_pca.permute(1, 2, 0)
        else:
            image_1_pca = rearrange(I_draw, "B H W C -> B C H W")
            image_1_pca = resize(image_1_pca, (originalH,originalW))
            image_1_pca = image_1_pca.permute(0, 2, 3, 1)
        del dino_model, backbone_model
        torch.cuda.empty_cache()
        
        return image_1_pca

    def __getitem__(self, item: int) -> Dict[str, Any]:
        if item not in self.gt_image_indices:
            return self.generated_get(item)
        else:
            index = self.indices[item]
            image = imageio.imread(self.parser.image_paths[index])[..., :3]
            # reshape to monst3r_size 
            # image = cv2.resize(image, (512, 384))
            # image = cv2.resize(image, (518, 228))
            
            camera_id = self.parser.camera_ids[index]
            # K = self.parser.Ks_dict[camera_id].cpu().numpy().copy()  # undistorted K
            K = self.parser.Ks_dict[camera_id].copy()  # undistorted K
            
            # resize large images according to factor
            if self.parser.factor > 1:
                image = cv2.resize(image, (image.shape[1] // self.parser.factor, image.shape[0] // self.parser.factor))
                # save resized image
                # breakpoint()

                # if isinstance(self.parser.image_paths[index], list):
                #     root_dir = os.path.dirname(self.parser.image_paths[index])
                # else:
                #     root_dir = self.parser.data_dir
                # if not os.path.exists(root_dir + f"/images_{self.parser.factor}"):
                #     os.makedirs(root_dir + f"/images_{self.parser.factor}")
                # imageio.imwrite(root_dir + f"/images_{self.parser.factor}/{self.parser.image_names[index]}", image)
                K[:2, :] /= self.parser.factor

            params = self.parser.params_dict[camera_id]
            camtoworlds_gt = self.parser.camtoworlds[index] 
            camtoworlds = self.camto_world[index] if self.datatype != 'monst3r' else self.parser.camtoworlds[index]
            
            if len(params) > 0:
                # Images are distorted. Undistort them.
                mapx, mapy = (
                    self.parser.mapx_dict[camera_id],
                    self.parser.mapy_dict[camera_id],
                )
                image = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)
                x, y, w, h = self.parser.roi_undist_dict[camera_id]
                image = image[y : y + h, x : x + w]

            if self.patch_size is not None:
                # Random crop.
                h, w = image.shape[:2]
                x = np.random.randint(0, max(w - self.patch_size, 1))
                y = np.random.randint(0, max(h - self.patch_size, 1))
                image = image[y : y + self.patch_size, x : x + self.patch_size]
                K[0, 2] -= x
                K[1, 2] -= y

            if self.load_canny:
                img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
                img_canny = cv2.Canny(img_blur, 50, 150)

            depth_filename = self.parser.depth_path + f"/{self.parser.image_names[index].split('.')[0]}.npy" if "tum" not in self.parser.depth_path else self.parser.depth_path + f"/{self.parser.image_names[index].split('.png')[0]}.npy"
            if index in self.depth_list:
                depth_map = self.depthmap_list[index]
                depth = self.depth_list[index]
            # elif os.path.exists(self.parser.depth_path + f"/{self.parser.image_names[index].split('.')[0]}.npy"):
            elif os.path.exists(depth_filename):
                print("load from : ",depth_filename)
                depth = np.load(depth_filename)

                if self.parser.factor > 1:
                    depth = cv2.resize(depth, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
                    assert depth.shape == depth.shape[:2], f"depth shape {depth.shape} does not match image shape {image.shape[:2]}"
                                    
                depth_map = torch.from_numpy(depth)
                pts = depth_to_3d(depth_map[None, None],
                                torch.from_numpy(K).float()[None],
                                normalize_points=False)
                depth = pts.permute(0, 2, 3, 1).squeeze(0).reshape(-1, 3)
                self.depth_list[index] = depth
                self.depthmap_list[index] = depth_map
            
            # elif os.path.exists(self.parser.depth_path + f"/{self.parser.image_names[index].split('.')[0]}_canonical.npy"):
            elif self.datatype == "monst3r":
                depth = np.load(self.parser.depth_path + f"/frame_{index:04d}.npy")
                depth_map = torch.from_numpy(depth)
                pts = depth_to_3d(depth_map[None, None],
                                torch.from_numpy(K).float()[None],
                                normalize_points=False)
                depth = pts.permute(0, 2, 3, 1).squeeze(0).reshape(-1, 3)
                self.depth_list[index] = depth
                self.depthmap_list[index] = depth_map
            else:
                input_image = torch.tensor(image).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
                if self.depth_type == "marigold":
                    try:
                        pipe_out = self.pipe(
                            input_image,
                            denoising_steps=None,
                            ensemble_size=5,
                            processing_res=768,
                            match_input_res=True,
                            batch_size=1,
                            color_map="Spectral",
                            show_progress_bar=False,
                            resample_method="bilinear",
                            generator=torch.Generator(self.device).manual_seed(42),
                        )
                    except:
                        pipe: MarigoldPipeline = MarigoldPipeline.from_pretrained("prs-eth/marigold-lcm-v1-0", variant=None, torch_dtype=torch.float32)
                        self.pipe = pipe.to(self.device)
                        pipe_out = self.pipe(
                            input_image,
                            denoising_steps=None,
                            ensemble_size=5,
                            processing_res=768,
                            match_input_res=True,
                            batch_size=1,
                            color_map="Spectral",
                            show_progress_bar=False,
                            resample_method="bilinear",
                            generator=torch.Generator(self.device).manual_seed(42),
                        )
                    prediction = pipe_out.depth_np
                    prediction = torch.from_numpy(pipe_out.depth_np)
                    # use image canny to mask out depth prediction

                    # # breakpoint()
                    prediction[prediction < 1e-5] = 1e-5

                    # depth to 3D 
                    intr_mat_tensor = torch.from_numpy(K).float()
                    # affine invariant depth so add scale and shift
                    marigold_scale = 1.1
                    marigold_t = 1.0
                    prediction = prediction*marigold_scale+marigold_t 

                    pts = depth_to_3d(prediction[None, None],
                                intr_mat_tensor[None],
                                normalize_points=False)
                    depth = pts.permute(0, 2, 3, 1).squeeze(0).reshape(-1, 3)

                    depth_map = prediction
                    self.depth_list[index] = depth
                    self.depthmap_list[index] = depth_map
                    prediction = prediction*10 # expand depth range to 0-10 
                    prediction[prediction<1] = 0 
                    pts_canonical = depth_to_3d(prediction[None, None],
                                intr_mat_tensor[None],
                                normalize_points=False)
                    depth_canonical = pts_canonical.permute(0, 2, 3, 1).squeeze(0).reshape(-1, 3)
                    print("marigold predicted ")

                else:
                    input_batch = self.depth_transforms(image).to(self.device)
                    prediction = self.depth_model(input_batch)

                    prediction = torch.nn.functional.interpolate(
                        prediction.unsqueeze(1),
                        size=input_image.shape[2:],
                        mode="bicubic",
                        align_corners=False,
                    ).squeeze()
                    prediction = prediction.cpu().detach()
                    prediction[prediction < 1e-8] = 1e-8
                    depth_map = prediction
                    # depth to 3D
                    intr_mat_tensor = torch.from_numpy(K).float().to(prediction.device)
                    pts = depth_to_3d(depth_map[None, None],
                                intr_mat_tensor[None],
                                normalize_points=False)
                    depth = pts.permute(0, 2, 3, 1).squeeze(0).reshape(-1, 3)
                    self.depth_list[index] = depth
                    self.depthmap_list[index] = depth_map

                    prediction = prediction*10 # expand depth range to 0-10
                    prediction[prediction<1] = 0
                    pts_canonical = depth_to_3d(prediction[None, None],
                                intr_mat_tensor[None],
                                normalize_points=False)
                    depth_canonical = pts_canonical.permute(0, 2, 3, 1).squeeze(0).reshape(-1, 3)

                # save the depth map
                # find image path and save depth map
                if not os.path.exists(self.parser.depth_path):
                    os.makedirs(self.parser.depth_path, exist_ok=True)
                np.save(f"{self.parser.depth_path}/{self.parser.image_names[index].split('.')[0]}.npy", depth_map.cpu().detach().numpy())


            if self.load_semantics and self.datatype == "custom":
                # assert os.path.exists(self.parser.semantic_path + f"/{self.parser.image_names[index].split('.')[0]}.npy") , f"Semantic mask not found for {self.parser.image_names[index].split('.')[0]}"
                if "tum" in self.parser.semantic_path:
                    mask_path = self.parser.semantic_path + f"/{self.parser.image_names[index].split('.png')[0]}.npy"
                else:
                    mask_path = self.parser.semantic_path + f"/{self.parser.image_names[index].split('.')[0]}.npy"
                assert os.path.exists(mask_path) , f"Semantic mask not found for {self.parser.image_names[index].split('.')[0]}"
                # mask = np.load(self.parser.semantic_path + f"/{self.parser.image_names[index].split('.')[0]}.npy")
                mask = np.load(mask_path)
                mask = torch.from_numpy(mask).bool()
                if self.parser.factor > 1:
                    mask = F.interpolate(mask.unsqueeze(0).float(), 
                                        size=(image.shape[0], image.shape[1]), mode='nearest').squeeze().bool()
                    assert mask.shape == image.shape[:2], f"Mask shape {mask.shape} does not match image shape {image.shape[:2]}"
                if mask.shape != image.shape[:2]:
                    mask = F.interpolate(mask.unsqueeze(0).float(),
                                        size=(image.shape[0], image.shape[1]), mode='nearest').squeeze().bool()
                mask = mask.view(image.shape[0], image.shape[1],1)
            elif self.load_semantics and self.datatype == "monst3r":
                assert os.path.exists(self.parser.semantic_path + f"/dynamic_mask_{index}.png")
                mask = imageio.imread(self.parser.semantic_path + f"/dynamic_mask_{index}.png") # 0 is non dynamic, 1 is dynamic
                mask = torch.from_numpy(mask/255).bool()
                if self.parser.factor > 1:
                    mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0).float(), 
                                        size=(image.shape[0], image.shape[1]), mode='nearest').squeeze().bool()
                    assert mask.shape == image.shape[:2], f"Mask shape {mask.shape} does not match image shape {image.shape[:2]}" 
                mask = ~mask.view(image.shape[0], image.shape[1],1)
            else:
                mask = None


            # dino feature 
            if self.dino:
                # if index not in self.dino_emb_list:
                dino_path = os.path.join(self.parser.dino_path, f"{self.parser.image_names[index].split('.')[0]}.npy")
                if not os.path.exists(dino_path):
                    # the minimum common multiple of 14 and image size 
                    embeddings = self.get_dino_feature_map(image, 518, 518, rank = self.opt.dino_dim if self.opt is not None else 3)
                    # assert embeddings.shape[:2] == image.shape[:2]
                    # self.dino_emb_list[index] = embeddings
                    np.save(dino_path, embeddings.detach().cpu().numpy())
                else:
                    embeddings = torch.from_numpy(np.load(dino_path))
                    if embeddings.shape[-1] != self.opt.dino_dim:
                        embeddings = self.get_dino_feature_map(image, 518, 518, rank = self.opt.dino_dim)
                        np.save(dino_path, embeddings.detach().cpu().numpy())
                    if embeddings.shape[:2] != image.shape[:2]:
                        embeddings = F.interpolate(embeddings.unsqueeze(0).permute(0,3,1,2), 
                                        size=(image.shape[0], image.shape[1]), mode='bilinear').squeeze().permute(1,2,0)
                    if self.parser.factor > 1:
                        embeddings = F.interpolate(embeddings.unsqueeze(0).permute(0,3,1,2), 
                                        size=(image.shape[0], image.shape[1]), mode='bilinear').squeeze().permute(1,2,0)
                    # self.dino_emb_list[index] = embeddings.to(self.device)

                # else:
                #     embeddings = self.dino_emb_list[index]

            else:
                embeddings = None
            data = {
                "K": torch.from_numpy(K).float().to(self.device),
                "camtoworld_gt": torch.from_numpy(camtoworlds_gt).float().to(self.device) if self.datatype != "monst3r" else torch.Tensor(camtoworlds_gt),
                # "camtoworld": torch.eye(4,4).float(),
                "camtoworld": torch.Tensor(camtoworlds).to(self.device),
                "image": torch.from_numpy(image).float().to(self.device) if not self.load_canny else torch.tensor(image*np.expand_dims(img_canny/255,axis =-1)).float(),
                # "image": input_image,
                "image_id": item,  # the index of the image in the dataset
                "depth": depth.to(self.device),
                "depth_map": depth_map.to(self.device),
                "mask": mask.to(self.device) if mask is not None else torch.zeros(image.shape[0], image.shape[1],1).bool().to(self.device),
                "generated" : False,
                "dino_embeddings": embeddings.to(self.device) if embeddings is not None else None
            }
            return data

    def generate_random_poses(self, phi_range, theta_range, radius_range, num_samples=2):
        phis = np.linspace(phi_range[0], phi_range[1], num_samples).reshape(-1)
        thetas = np.linspace(theta_range[0], theta_range[1], num_samples).reshape(-1)
        radii = np.linspace(radius_range[0], radius_range[1],2).reshape(-1)
        phis, thetas, radii = np.meshgrid(phis, thetas, radii)
        phis = phis.flatten()
        thetas = thetas.flatten()
        radii = radii.flatten()
        x = radii * np.sin(thetas) * np.cos(phis)
        y = radii * np.sin(thetas) * np.sin(phis)
        z = radii * np.cos(thetas)
        generated_cam_view = np.stack([x, y, z], axis=-1)
        # add the camera poses to the dataset
        for i in range(len(generated_cam_view)):
            self.indices = np.append(self.indices, len(self.indices)+1)
            cam_pose = np.eye(4)
            cam_pose[:3,3] = generated_cam_view[i]
            self.camto_world = np.append(self.camto_world, cam_pose.reshape(1,4,4), axis=0)

        import numpy as np

    # def generate_spherical_camera_poses(self, phi_range, theta_range, radius_range, target_position, num_samples=2):
    def generate_spherical_camera_poses(self, phi_range, theta_range, radius_range, target_position, num_samples=2):
        # Generate samples for phi, theta, and radius
        phis = np.linspace(phi_range[0], phi_range[1], num_samples)
        thetas = np.linspace(theta_range[0], theta_range[1], num_samples)
        radii = np.linspace(radius_range[0], radius_range[1], 2)
        # radii = np.linspace(radius_range[0], radius_range[0], 1)
        
        # phis = np.random.uniform(phi_range[0], phi_range[1], num_samples)
        # thetas = np.random.uniform(theta_range[0], theta_range[1], num_samples)
        # radii = np.random.uniform(radius_range[0], radius_range[1], num_samples)
    
        # # Convert phi and theta from degrees to radians
        # phis = np.deg2rad(phis)
        # thetas = np.deg2rad(thetas)
        
        # Create a grid of phi, theta, and radius combinations
        phis, thetas, radii = np.meshgrid(phis, thetas, radii)
        phis = phis.flatten()
        thetas = thetas.flatten()
        radii = radii.flatten()
        
        # Convert spherical coordinates to Cartesian coordinates
        x = radii * np.sin(thetas) * np.cos(phis)
        z = radii * np.sin(thetas) * np.sin(phis)
        y = radii * np.cos(thetas)


        generated_cam_positions = np.stack([x, y, z], axis=-1)  * 1

        print("generated_cam_positions",generated_cam_positions)
        # scale the position into the scale of original camto_world

        # Add the camera poses to the dataset
        for i in range(len(generated_cam_positions)):
            # Compute the position of the camera
            cam_position = generated_cam_positions[i]

            # Calculate the forward vector pointing from the camera to the target position
            forward =  -cam_position + target_position
            forward /= np.linalg.norm(forward)

            # Define a global up vector for Y-Up coordinate system
            up = np.array([0, 1, 0])

            # Recompute the right vector as the cross product of up and forward
            right = np.cross(up, forward)
            right /= np.linalg.norm(right)

            # Recalculate the up vector to ensure orthogonality
            up = np.cross(forward, right)
            up /= np.linalg.norm(up)

            # Construct the rotation matrix
            rotation_matrix = np.vstack([right, up, forward])

            # Create the 4x4 camera pose matrix
            cam_pose = np.eye(4)
            cam_pose[:3, :3] = rotation_matrix
            cam_pose[:3, 3] = cam_position

            # Append the camera pose to the dataset
            self.indices = np.append(self.indices, len(self.indices) + 1)
            self.camto_world = np.append(self.camto_world, cam_pose.reshape(1, 4, 4), axis=0)

    def normalize(self,v):
        return v / np.linalg.norm(v)

    def look_at_matrix(self,camera_position, target_point, up_vector=np.array([0, 1, 0])):  # Up vector is now y-axis
        # Compute forward vector (camera looks along the negative z-axis)
        forward = self.normalize(target_point - camera_position)
        
        # Compute the right vector (x-axis), which is perpendicular to both forward and up vectors
        right = self.normalize(np.cross(up_vector, forward))
        
        # Recompute the up vector to ensure orthogonality
        up = np.cross(forward, right)
        
        # Rotation matrix (camera space to world space)
        R = np.stack([right, up, forward], axis=1)
        
        # Build the 4x4 camera pose matrix
        pose_matrix = np.eye(4)
        pose_matrix[:3, :3] = R  # Insert rotation matrix
        pose_matrix[:3, 3] = camera_position  # Insert camera position (translation)
        
        return pose_matrix

    def generate_camera_poses(self,target_point, radius_range, theta_range, phi_range,nsamples=10):
        # create self.angels property in the dataset
        target = np.array(target_point)  # Target point in the point cloud
        poses = []
        
        # Loop through theta and phi ranges
        for r in np.linspace(radius_range[0], radius_range[-1], 1):  # Adjust step count as needed
            for theta in np.linspace(theta_range[0], theta_range[1], 1):  # Azimuthal angle
                for ind, phi in enumerate(np.linspace(phi_range[0], phi_range[-1], nsamples)):  # Polar angle
                    # Convert spherical to Cartesian coordinates for camera position
                    # change degree to radian
                    theta = np.deg2rad(theta)
                    phi = np.deg2rad(phi)
                    
                    x_c = target[0] + r * np.sin(phi) * np.cos(theta)  # X is width
                    y_c = target[1] + r * np.sin(phi) * np.sin(theta)  # Y is height
                    z_c = target[2] - r * np.cos(phi)  # Z is depth
                    # z_c = 0
                    camera_position = np.array([x_c, y_c, z_c])

                    
                    # Generate the 4x4 pose matrix
                    pose_matrix = self.look_at_matrix(camera_position, target)
                    self.indices = np.append(self.indices, len(self.indices) + 1)
                    self.camto_world = np.append(self.camto_world, pose_matrix.reshape(1, 4, 4), axis=0)
                    self.angle = np.append(self.angle, np.array([phi,theta,r]).reshape(1,3), axis=0)


    
    def generated_get(self,item:int) -> Dict[str, Any]:
        example_data = self.__getitem__(0)
        # K = self.parser.Ks_dict[0].copy()  # undistorted K
        K = example_data["K"].cpu().numpy()
        data ={
            "K": torch.from_numpy(K).float().to(self.device),
            "camtoworld_gt": torch.eye(4,4).float().to(self.device),
            "camtoworld": torch.from_numpy(self.camto_world[item]).float().to(self.device),
            "image": torch.zeros_like(example_data["image"]).float().to(self.device),
            # "image": None,
            "image_id": item % len(self.parser.image_names),  # the index of the image in the dataset
            "depth": torch.zeros_like(example_data["depth"]).float().to(self.device),
            # "depth": None,
            "depth_map": torch.zeros_like(example_data["depth_map"]).float().to(self.device),
            "mask": torch.zeros_like(example_data["mask"]).bool().to(self.device),
            "generated" : True,
            "angle": torch.from_numpy(self.angle[item]).float().to(self.device) 
            # "depth_map": None,
            # "mask": None

        }
        return data
    
    # def collate(self,batch):
    #     batch = [item for item in batch if item is not None]
    #     # Return the remaining items
    #     return torch.utils.data.dataloader.default_collate(batch)
    def collate(self,batch):
        collated_batch = {}
        
        # Get keys from the first item to form the structure of the batch
        keys = batch[0].keys()

        for key in keys:
            # For each key, collect all values and filter None values
            key_items = [item[key] for item in batch]
            
            if isinstance(key_items[0], torch.Tensor):
                # Use default collate if the values are tensors
                collated_batch[key] = torch.stack([x for x in key_items if x is not None])
            else:
                # Otherwise, just collect the values (could be None or something else)
                collated_batch[key] = key_items

if __name__ == "__main__":
    import argparse

    import imageio.v2 as imageio
    import tqdm

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/360_v2/garden")
    parser.add_argument("--factor", type=int, default=4)
    args = parser.parse_args()

    # Parse COLMAP data.
    parser = Parser(
        data_dir=args.data_dir, factor=args.factor, normalize=True, test_every=8
    )
    dataset = Dataset(parser, split="train", load_depths=True)
    print(f"Dataset: {len(dataset)} images.")

    writer = imageio.get_writer("results/points.mp4", fps=30)
    for data in tqdm.tqdm(dataset, desc="Plotting points"):
        image = data["image"].numpy().astype(np.uint8)
        points = data["points"].numpy()
        depths = data["depths"].numpy()
        for x, y in points:
            cv2.circle(image, (int(x), int(y)), 2, (255, 0, 0), -1)
        writer.append_data(image)
    writer.close()
