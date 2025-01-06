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
import math

from Marigold.marigold import MarigoldPipeline

from .normalize import (
    align_principle_axes,
    similarity_from_cameras,
    transform_cameras,
    transform_points,
)

from PIL import Image


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
    ):
        self.data_dir = data_dir
        self.factor = factor
        self.normalize = normalize
        self.test_every = test_every

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

                # camera intrinsics
                cam = manager.cameras[camera_id]
                fx, fy, cx, cy = cam.fx, cam.fy, cam.cx, cam.cy
                K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
                K[:2, :] /= factor
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
            data_info = data_dir.split('/')
            seq_name = data_info[-1]

            # image_names = [name for name in sorted(os.listdir(os.path.join(data_dir,"images")))]
            image_names = [name for name in sorted(os.listdir(os.path.join(data_dir,"images")),key = lambda x: int(x.split(".")[-2]))]
            seq_len = len(image_names)

            # find one image to get the size
            image = imageio.imread(os.path.join(data_dir,"images",image_names[0]))[..., :3]
            height, width = image.shape[:2]
            
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


        self.semantic_path = os.path.join(data_dir, "semantics")
        self.depth_path = os.path.join(data_dir, "depths")
        os.makedirs(self.semantic_path, exist_ok=True)
        os.makedirs(self.depth_path, exist_ok=True)
        
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
    ):
        self.parser = parser
        self.split = split
        self.patch_size = patch_size
        self.load_depths = load_depths
        self.load_semantics = load_semantics
        self.load_canny = load_canny
        self.depth_type = "marigold"
        indices = np.arange(len(self.parser.image_names))
        # self.camto_world = torch.eye(4,4).float().repeat(len(indices), 1, 1)
        self.camto_world = np.eye(4,4).astype(np.float32).reshape(1,4,4).repeat(len(indices),0)
        self.indices = indices
        pipe: MarigoldPipeline = MarigoldPipeline.from_pretrained("prs-eth/marigold-lcm-v1-0", variant=None, torch_dtype=torch.float32)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.depth_list = OrderedDict()
        self.depthmap_list = OrderedDict()
        if self.depth_type == "marigold":
            self.pipe = pipe.to(self.device)
        else:
            midas = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid")
            midas.to(self.device)
            midas.eval()
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            self.depth_transforms = midas_transforms.dpt_transform
            self.depth_model = midas

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



    def __getitem__(self, item: int) -> Dict[str, Any]:
        index = self.indices[item]
        image = imageio.imread(self.parser.image_paths[index])[..., :3]
        camera_id = self.parser.camera_ids[index]
        K = self.parser.Ks_dict[camera_id].copy()  # undistorted K
        
        # resize large images according to factor
        if self.parser.factor > 1:
            image = cv2.resize(image, (image.shape[1] // self.parser.factor, image.shape[0] // self.parser.factor))
            K[:2, :] /= self.parser.factor

        params = self.parser.params_dict[camera_id]
        camtoworlds_gt = self.parser.camtoworlds[index]
        camtoworlds = self.camto_world[index]

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

        
        if index in self.depth_list:
            depth_map = self.depthmap_list[index]
            depth = self.depth_list[index]
        elif os.path.exists(self.parser.depth_path + f"/{self.parser.image_names[index].split('.')[0]}.npy"):
            print("load from : ", self.parser.depth_path + f"/{self.parser.image_names[index].split('.')[0]}.npy")
            depth = np.load(self.parser.depth_path + f"/{self.parser.image_names[index].split('.')[0]}.npy")
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
                # # range 1-10
                
                # prediction = (prediction - prediction.min()) / (prediction.max() - prediction.min())
                # prediction = prediction*10 
                # # depth_scale = np.clip(depth_scale, 1, 10)
                # prediction = torch.clamp(prediction, 1, 10)
                
                # range 0-1


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
            np.save(f"{self.parser.depth_path}/{self.parser.image_names[index].split('.')[0]}_canonical.npy", prediction.cpu().detach().numpy())


        if self.load_semantics:
            assert os.path.exists(self.parser.semantic_path + f"/{self.parser.image_names[index].split('.')[0]}.npy") , f"Semantic mask not found for {self.parser.image_names[index].split('.')[0]}"

            mask = np.load(self.parser.semantic_path + f"/{self.parser.image_names[index].split('.')[0]}.npy")
            mask = torch.from_numpy(mask).bool()
            if self.parser.factor > 1:
                mask = F.interpolate(mask.unsqueeze(0).float(), 
                                     size=(image.shape[0], image.shape[1]), mode='nearest').squeeze().bool()
                assert mask.shape == image.shape[:2], f"Mask shape {mask.shape} does not match image shape {image.shape[:2]}"
            mask = mask.view(image.shape[0], image.shape[1],1)
        else:
            mask = None
        # breakpoint()
        data = {
            "K": torch.from_numpy(K).float(),
            "camtoworld_gt": torch.from_numpy(camtoworlds_gt).float(),
            # "camtoworld": torch.eye(4,4).float(),
            "camtoworld": torch.Tensor(camtoworlds),
            "image": torch.from_numpy(image).float() if not self.load_canny else torch.tensor(image*np.expand_dims(img_canny/255,axis =-1)).float(),
            # "image": input_image,
            "image_id": item,  # the index of the image in the dataset
            "depth": depth,
            "depth_map": depth_map,
            "mask": mask
        }
        return data
    
# class Dataset:
#     """A dataset class for loading images and camera parameters."""

#     def __init__(
#         self,
#         parser: Parser,
#         split: str = "train",
#         patch_size: Optional[int] = None,
#         load_depths: bool = False,
#     ):
#         self.parser = parser
#         self.split = split
#         self.patch_size = patch_size
#         self.load_depths = load_depths
#         indices = np.arange(len(self.parser.image_names))
#         if split == "train":
#             self.indices = indices[indices % self.parser.test_every != 0]
#         else:
#             self.indices = indices[indices % self.parser.test_every == 0]

#     def __len__(self):
#         return len(self.indices)

#     def __getitem__(self, item: int) -> Dict[str, Any]:
#         index = self.indices[item]
#         image = imageio.imread(self.parser.image_paths[index])[..., :3]
#         camera_id = self.parser.camera_ids[index]
#         K = self.parser.Ks_dict[camera_id].copy()  # undistorted K
#         params = self.parser.params_dict[camera_id]
#         camtoworlds = self.parser.camtoworlds[index]

#         if len(params) > 0:
#             # Images are distorted. Undistort them.
#             mapx, mapy = (
#                 self.parser.mapx_dict[camera_id],
#                 self.parser.mapy_dict[camera_id],
#             )
#             image = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)
#             x, y, w, h = self.parser.roi_undist_dict[camera_id]
#             image = image[y : y + h, x : x + w]

#         if self.patch_size is not None:
#             # Random crop.
#             h, w = image.shape[:2]
#             x = np.random.randint(0, max(w - self.patch_size, 1))
#             y = np.random.randint(0, max(h - self.patch_size, 1))
#             image = image[y : y + self.patch_size, x : x + self.patch_size]
#             K[0, 2] -= x
#             K[1, 2] -= y

#         data = {
#             "K": torch.from_numpy(K).float(),
#             "camtoworld": torch.from_numpy(camtoworlds).float(),
#             "image": torch.from_numpy(image).float(),
#             "image_id": item,  # the index of the image in the dataset
#         }

#         if self.load_depths:
#             # projected points to image plane to get depths
#             worldtocams = np.linalg.inv(camtoworlds)
#             image_name = self.parser.image_names[index]
#             point_indices = self.parser.point_indices[image_name]
#             points_world = self.parser.points[point_indices]
#             points_cam = (worldtocams[:3, :3] @ points_world.T + worldtocams[:3, 3:4]).T
#             points_proj = (K @ points_cam.T).T
#             points = points_proj[:, :2] / points_proj[:, 2:3]  # (M, 2)
#             depths = points_cam[:, 2]  # (M,)
#             if self.patch_size is not None:
#                 points[:, 0] -= x
#                 points[:, 1] -= y
#             # filter out points outside the image
#             selector = (
#                 (points[:, 0] >= 0)
#                 & (points[:, 0] < image.shape[1])
#                 & (points[:, 1] >= 0)
#                 & (points[:, 1] < image.shape[0])
#                 & (depths > 0)
#             )
#             points = points[selector]
#             depths = depths[selector]
#             data["points"] = torch.from_numpy(points).float()
#             data["depths"] = torch.from_numpy(depths).float()

#         return data


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
