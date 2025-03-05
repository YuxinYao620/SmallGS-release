"""A simple example to render a (large-scale) Gaussian Splats

```bash
python examples/simple_viewer.py --scene_grid 13
```
"""

import argparse
import math
import os
import time
from typing import Tuple

import imageio
import nerfview
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import viser
import viser.transforms as tf

from gsplat._helper import load_test_data
from gsplat.distributed import cli
from gsplat.rendering import rasterization

from scipy.spatial.transform import Rotation as R

import imageio.v3 as iio

def so3_to_quat(so3):
    w = torch.sqrt(1 + so3[0, 0] + so3[1, 1] + so3[2, 2]) / 2
    w4 = 4 * w
    x = (so3[2, 1] - so3[1, 2]) / w4
    y = (so3[0, 2] - so3[2, 0]) / w4
    z = (so3[1, 0] - so3[0, 1]) / w4

    # normalize
    quat = torch.tensor([w, x, y, z])
    quat /= torch.norm(quat)
    return quat

# def load_camera_pose(pose_path,device):
#     base_dir = os.path.dirname(pose_path)
#     result_path = os.path.dirname(pose_path)
#     poses = torch.load(pose_path)
#     poses_pred = torch.Tensor(poses['global_cam_poses_est'])
#     try: 
#         gt_pose = torch.from_numpy(np.load(base_dir+"/traj_est.npy"))
#         poses_gt_c2w = gt_pose
#         gt_quats = poses_gt_c2w[:, 3:]
#         gt_positions = poses_gt_c2w[:, :3]
#     except:
#         poses_gt_c2w = None
#         gt_quats = torch.eye(3).unsqueeze(0).repeat(poses_pred.shape[0],1,1)
#         gt_positions = torch.zeros(poses_pred.shape[0],3)
        
#     # rotation mat to quaternion
#     pred_quats = poses_pred[:, :3, :3].cpu()
#     pred_quats = torch.stack([so3_to_quat(gt_quat) for gt_quat in poses_pred])
#     pred_positions = poses_pred[:, :3, 3].cpu()

#     mean_point = poses['mean_point'] if 'mean_point' in poses else torch.zeros(3)


#     return gt_quats.to(device), gt_positions.to(device), pred_quats.to(device), pred_positions.to(device) , mean_point
def quat_to_rotmat(quat):
    r = R.from_quat(quat.cpu().numpy())
    return torch.tensor(r.as_matrix(), dtype=torch.float32)

def rotmat_to_quat(rotmat):
    r = R.from_matrix(rotmat.cpu().numpy())
    return torch.tensor(r.as_quat(), dtype=torch.float32)

def load_camera_pose(pose_path, device):
    from evo.core.trajectory import PoseTrajectory3D,PosePath3D

    # base_dir = "/home/yuxinyao/batch-cf3dgs/paper_example/tum_rgb_refine_pose_copy/rgbd_dataset_freiburg3_walking_static_seq_390_419"

    # pred_path = os.path.join(pose_path, "pred_traj_monst3r_for_refine.txt")
    pred_path = os.path.join(pose_path, "gs_pose.txt")

    # gt_path = os.path.join(pose_path, "cam_poses.npy")
    # gt_pose = torch.Tensor(np.load(gt_path))
    # gt_rot = gt_pose[:,:3,:3]


    with open(pred_path, 'r') as f:
        pred_pose = f.readlines()
        pred_pose = torch.Tensor(np.array([list(map(float,pose.split())) for pose in pred_pose]))
        #xyzw to wxyz 
        pred_pose[:,4:] = pred_pose[:,[7,4,5,6]]

    # align gt and pred
    # gt_path = os.path.join(pose_path, "cam_relative_traj.npy") 
    gt_path = os.path.join(pose_path, "cam_poses_traj.npy")
    gt_pose = torch.Tensor(np.load(gt_path))
    # gt_pose[:,[4,5,6]] = -gt_pose[:,[4,5,6]]

    # pose_gt = PosePath3D(poses_se3=gt_pose)
    pose_gt = PosePath3D( positions_xyz=gt_pose[:,0:3],
            orientations_quat_wxyz=gt_pose[:,[3,4,5,6]])
            # orientations_quat_wxyz=gt_pose[:,[4,5,6,3]])
            # orientations_quat_wxyz=gt_pose[:,[5,6,3,4]])
            # orientations_quat_wxyz=gt_pose[:,[6,3,4,5]])
    pose_pred = PosePath3D(
            positions_xyz=pred_pose[:,1:4],
            orientations_quat_wxyz=pred_pose[:,4:])
    # gt_quats = torch.stack([so3_to_quat(gt_quat) for gt_quat in gt_rot])
    # gt_positions = gt_pose[:, :3, 3].cpu()

    # pred_positions = pred_pose[:,1:4]
    # pred_quats =pred_pose[:,4:]

    # align 
    import copy
    traj_gt= copy.deepcopy(pose_gt)
    traj_gt.align(pose_pred, correct_scale=True,
                        correct_only_scale=False)

    
    gt_quats = torch.Tensor(traj_gt.orientations_quat_wxyz)
    gt_positions = torch.Tensor(traj_gt.positions_xyz)
    # gt_quats = gt_pose[:,3:]
    # gt_positions = gt_pose[:,0:3]

    pred_positions = pred_pose[:,1:4]
    pred_quats =pred_pose[:,4:]
    

    mean_point = np.zeros(3)
    # difference in quaternion to degree    
    r1 = R.from_quat(gt_quats.cpu().numpy())
    r2 = R.from_quat(pred_quats.cpu().numpy())
    relative_rot = r1.inv() * r2
    angle_rad = relative_rot.magnitude()
    angle_deg = np.degrees(angle_rad)
    print("relative rotation in degree", angle_deg)
    
    monst3r_path = os.path.join(pose_path, "pred_traj_monst3r_for_refine.txt")
    with open(monst3r_path, 'r') as f:
        monst3r_pose = f.readlines()
        monst3r_pose = torch.Tensor(np.array([list(map(float,pose.split())) for pose in monst3r_pose]))
        #xyzw to wxyz 
        # monst3r_pose[:,4:] = monst3r_pose[:,[7,4,5,6]]
    monst3r_traj = PosePath3D(
            positions_xyz=monst3r_pose[:,1:4],
            orientations_quat_wxyz=monst3r_pose[:,4:])
    # align
    # monst3r_traj.align(pose_pred, correct_scale=True,
    #                     correct_only_scale=False)

    monst3r_quats = monst3r_pose[:,4:]
    monst3r_positions = monst3r_pose[:,1:4]
    


    return pred_quats.to(device), gt_positions.to(device), pred_quats.to(device), gt_positions.to(device), mean_point
    # return pred_quats.to(device), pred_positions.to(device), pred_quats.to(device), pred_positions.to(device), mean_point 
    # return monst3r_quats.to(device), monst3r_positions.to(device), monst3r_quats.to(device), monst3r_positions.to(device), mean_point

def main(local_rank: int, world_rank, world_size: int, args):
    torch.manual_seed(42)
    device = torch.device("cuda", local_rank)

    if args.ckpt is None:
        (
            means,
            quats,
            scales,
            opacities,
            colors,
            viewmats,
            Ks,
            width,
            height,
        ) = load_test_data(device=device, scene_grid=args.scene_grid)

        assert world_size <= 2
        means = means[world_rank::world_size].contiguous()
        means.requires_grad = True
        quats = quats[world_rank::world_size].contiguous()
        quats.requires_grad = True
        scales = scales[world_rank::world_size].contiguous()
        scales.requires_grad = True
        opacities = opacities[world_rank::world_size].contiguous()
        opacities.requires_grad = True
        colors = colors[world_rank::world_size].contiguous()
        colors.requires_grad = True

        viewmats = viewmats[world_rank::world_size][:1].contiguous()
        Ks = Ks[world_rank::world_size][:1].contiguous()

        sh_degree = None
        C = len(viewmats)
        N = len(means)
        print("rank", world_rank, "Number of Gaussians:", N, "Number of Cameras:", C)

        # batched render
        for _ in tqdm.trange(1):
            render_colors, render_alphas, meta = rasterization(
                means,  # [N, 3]
                quats,  # [N, 4]
                scales,  # [N, 3]
                opacities,  # [N]
                colors,  # [N, 3]
                viewmats,  # [C, 4, 4]
                Ks,  # [C, 3, 3]
                width,
                height,
                render_mode="RGB+D",
                packed=False,
                distributed=world_size > 1,
            )
        C = render_colors.shape[0]
        assert render_colors.shape == (C, height, width, 4)
        assert render_alphas.shape == (C, height, width, 1)
        render_colors.sum().backward()

        render_rgbs = render_colors[..., 0:3]
        render_depths = render_colors[..., 3:4]
        render_depths = render_depths / render_depths.max()

        # dump batch images
        os.makedirs(args.output_dir, exist_ok=True)
        canvas = (
            torch.cat(
                [
                    render_rgbs.reshape(C * height, width, 3),
                    render_depths.reshape(C * height, width, 1).expand(-1, -1, 3),
                    render_alphas.reshape(C * height, width, 1).expand(-1, -1, 3),
                ],
                dim=1,
            )
            .detach()
            .cpu()
            .numpy()
        )
        imageio.imsave(
            f"{args.output_dir}/render_rank{world_rank}.png",
            (canvas * 255).astype(np.uint8),
        )
    else:
        means, quats, scales, opacities, sh0, shN = [], [], [], [], [], []
        # colors = []
        for ckpt_path in args.ckpt:
            ckpt = torch.load(ckpt_path, map_location=device)["splats"]
            means.append(ckpt["means"])
            quats.append(F.normalize(ckpt["quats"], p=2, dim=-1))
            scales.append(torch.exp(ckpt["scales"]))
            opacities.append(torch.sigmoid(ckpt["opacities"]))
            sh0.append(ckpt["sh0"])
            shN.append(ckpt["shN"])
            # colors.append(ckpt["colors"])
        means = torch.cat(means, dim=0)
        quats = torch.cat(quats, dim=0)
        scales = torch.cat(scales, dim=0)
        opacities = torch.cat(opacities, dim=0)
        sh0 = torch.cat(sh0, dim=0)
        shN = torch.cat(shN, dim=0)
        colors = torch.cat([sh0, shN], dim=-2)
        # colors = torch.cat(colors, dim=0)
        sh_degree = int(math.sqrt(colors.shape[-2]) - 1)

        print("Number of Gaussians:", len(means))


    # register and open viewer
    @torch.no_grad()
    def viewer_render_fn(camera_state: nerfview.CameraState, img_wh: Tuple[int, int]):
        width, height = img_wh
        c2w = camera_state.c2w
        K = camera_state.get_K(img_wh)
        c2w = torch.from_numpy(c2w).float().to(device)
        K = torch.from_numpy(K).float().to(device)
        viewmat = c2w.inverse()

        if args.backend == "gsplat":
            rasterization_fn = rasterization
        elif args.backend == "gsplat_legacy":
            from gsplat import rasterization_legacy_wrapper

            rasterization_fn = rasterization_legacy_wrapper
        elif args.backend == "inria":
            from gsplat import rasterization_inria_wrapper

            rasterization_fn = rasterization_inria_wrapper
        else:
            raise ValueError

        render_colors, render_alphas, meta = rasterization_fn(
            means,  # [N, 3]
            quats,  # [N, 4]
            scales,  # [N, 3]
            opacities,  # [N]
            colors,  # [N, 3]
            viewmat[None],  # [1, 4, 4]
            K[None],  # [1, 3, 3]
            width,
            height,
            sh_degree=sh_degree,
            render_mode="RGB",
            # this is to speedup large-scale rendering by skipping far-away Gaussians.
            radius_clip=3,
            backgrounds = torch.Tensor([255,255,255]).unsqueeze(0).to(device),
        )
        render_rgbs = render_colors[0, ..., 0:3].cpu().numpy()
        return render_rgbs



    server = viser.ViserServer(port=args.port, verbose=False)
    server.background_color = (0,0,0)
    image = np.ones((1000,1000,3),dtype=np.uint8)*255
    # save image 
    imageio.imwrite("background.png",image)


    server.scene.set_background_image(
    iio.imread("background.png"),
    format="png",
    )
    # generate a white background
    # Add main image.
    # server.scene.add_image(
    #     "/img",
    #     iio.imread("background.png"),
    #     2000.0,
    #     2000.0,
    #     format="png",
    #     wxyz=(1.0, 0.0, 0.0, 0.0),
    #     position=(0, 0,1050.0),
    # )


    ref_quats, ref_positions, pred_quats, pred_positions, mean_point  = load_camera_pose(args.pose_dir,device)

    
    pred_frame_list = []
    ref_frame_list = []


    @server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
        """For each client that connects, we create a set of random frames + a click handler for each frame.

        When a frame is clicked, we move the camera to the corresponding frame.
        """

        rng = np.random.default_rng(0)

        def make_frame(i: int,pred=True) -> None:
            # Sample a random orientation + position.
            # wxyz = rng.normal(size=4)
            # wxyz /= np.linalg.norm(wxyz) #quaternion normalization
            # position = rng.uniform(-3.0, 3.0, size=(3,)) #random position
            if pred:
                wxyz = pred_quats[i].cpu().numpy()
                position = pred_positions[i].cpu().numpy()
                color = (255,0,0)
            else:
                wxyz = ref_quats[i].cpu().numpy()
                position = ref_positions[i].cpu().numpy()
                color = (0,255,0)
            # Create a coordinate frame and label.
            # frame = client.scene.add_frame(f"/frame_{i}", wxyz=wxyz, position=position)
            # client.scene.add_label(f"/frame_{i}/label", text=f"Frame {i}")
            frame = client.scene.add_camera_frustum(name=f"/frame_{i}",fov=79, aspect=1.2,color=color, wxyz=wxyz, position=position, scale=0.003)
            # client.scene.add_label(f"/frame_{i}/label", text=f"Frame {i}")

            if pred:
                pred_frame_list.append(frame)
            else:
                ref_frame_list.append(frame)
            # Move the camera when we click a frame.
            @frame.on_click
            def _(_):
                T_world_current = tf.SE3.from_rotation_and_translation(
                    tf.SO3(client.camera.wxyz), client.camera.position
                )
                T_world_target = tf.SE3.from_rotation_and_translation(
                    tf.SO3(frame.wxyz), frame.position
                ) @ tf.SE3.from_translation(np.array([0.0, 0.0, -0.05]))

                T_current_target = T_world_current.inverse() @ T_world_target

                for j in range(20):
                    T_world_set = T_world_current @ tf.SE3.exp(
                        T_current_target.log() * j / 19.0
                    )

                    # We can atomically set the orientation and the position of the camera
                    # together to prevent jitter that might happen if one was set before the
                    # other.
                    with client.atomic():
                        client.camera.wxyz = T_world_set.rotation().wxyz
                        client.camera.position = T_world_set.translation()

                    client.flush()  # Optional!
                    time.sleep(1.0 / 60.0)

                # Mouse interactions should orbit around the frame origin.
                client.camera.look_at = frame.position


        for i in range(len(pred_quats)):
            make_frame(i)
            # make_frame(i,pred=False)
        

    button = server.gui.add_button("View all estimated camera pose")
    @button.on_click
    def _(event: viser.GuiEvent) -> None:
        client = event.client
        assert client is not None

        # view the scene from 20 different camera poses
        for i in range(len(pred_quats)):
            frame = pred_frame_list[i]
            T_world_current = tf.SE3.from_rotation_and_translation(
                    tf.SO3(client.camera.wxyz), client.camera.position
                )
            T_world_target = tf.SE3.from_rotation_and_translation(
                tf.SO3(frame.wxyz), frame.position
            ) @ tf.SE3.from_translation(np.array([0.0, 0.0, -0.005]))

            T_current_target = T_world_current.inverse() @ T_world_target

            T_world_set = T_world_current @ T_current_target
            with client.atomic():
                client.camera.wxyz = T_world_set.rotation().wxyz
                client.camera.position = T_world_set.translation()
                # client.flush()  # Optional!
            
            time.sleep(1.0)

            # Mouse interactions should orbit around the frame origin.
            client.camera.look_at = frame.position

    button2 = server.gui.add_button("View all droid camera pose")
    @button2.on_click
    def _(event: viser.GuiEvent) -> None:
        client = event.client
        assert client is not None
        # for i in range(len(pred_quats)):
        #     make_frame(i,pred=False)

        # view the scene from 20 different camera poses
        for i in range(len(ref_quats)):
            frame = ref_frame_list[i]
            T_world_current = tf.SE3.from_rotation_and_translation(
                    tf.SO3(client.camera.wxyz), client.camera.position
                )
            T_world_target = tf.SE3.from_rotation_and_translation(
                tf.SO3(frame.wxyz), frame.position
            ) @ tf.SE3.from_translation(np.array([0.0, 0.0, 0.0]))

            T_current_target = T_world_current.inverse() @ T_world_target

            T_world_set = T_world_current @ T_current_target
            with client.atomic():
                client.camera.wxyz = T_world_set.rotation().wxyz
                client.camera.position = T_world_set.translation()
                # client.flush()  # Optional!
            
            time.sleep(1.0)

            # Mouse interactions should orbit around the frame origin.
            client.camera.look_at = frame.position
            # Mouse interactions should orbit around the frame origin.

    button3 = server.gui.add_button("go to first frame")
    @button3.on_click
    def _(event: viser.GuiEvent) -> None:
        client = event.client
        assert client is not None
        # for i in range(len(pred_quats)):
        #     make_frame(i,pred=False)

        # view the scene from 20 different camera poses
        frame = pred_frame_list[0]
        T_world_current = tf.SE3.from_rotation_and_translation(
                    tf.SO3(client.camera.wxyz), client.camera.position
                )
        T_world_target = tf.SE3.from_rotation_and_translation(
            tf.SO3(frame.wxyz), frame.position
        ) @ tf.SE3.from_translation(np.array([0.0, 0.0, -0.05]))

        T_current_target = T_world_current.inverse() @ T_world_target

        T_world_set = T_world_current @ T_current_target
        with client.atomic():
            client.camera.wxyz = T_world_set.rotation().wxyz
            client.camera.position = T_world_set.translation()
            # client.flush()  # Optional!
        
        time.sleep(1.0)

        # Mouse interactions should orbit around the frame origin.
        client.camera.look_at = frame.position

        # Mouse interactions should orbit around the frame origin.
        # Mouse interactions should orbit around the frame origin.

    # highlight specific point in the scene
    gui_point_size = server.gui.add_slider(
    "Point size", min=0.01, max=0.1, step=0.001, initial_value=0.05
    )
    # breakpoint()
    # server.scene.add_point_cloud(name="/colmap/pcd",
    #     points=mean_point.reshape(1,3),
    #     colors=(0,255,0),
    #     point_size=gui_point_size.value,)

    # server.scene.add_point_cloud(name="origin",
    #                        points=np.array([0,0,0]).reshape(1,3),
    #                        colors=(255,0,0),
    #                        point_size = gui_point_size.value)
    

    viewer = nerfview.Viewer(
        server=server,
        render_fn=viewer_render_fn,
        mode="rendering",
    )

    print("Viewer running... Ctrl+C to exit.")
    time.sleep(100000)



if __name__ == "__main__":
    """
    # Use single GPU to view the scene
    CUDA_VISIBLE_DEVICES=0 python simple_viewer.py \
        --ckpt results/garden/ckpts/ckpt_3499_rank0.pt results/garden/ckpts/ckpt_3499_rank1.pt \
        --port 8081
    """
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir", type=str, default="results/", help="where to dump outputs"
    )
    parser.add_argument(
        "--scene_grid", type=int, default=1, help="repeat the scene into a grid of NxN"
    )
    parser.add_argument(
        "--ckpt", type=str, nargs="+", default=None, help="path to the .pt file"
    )
    parser.add_argument(
        "--port", type=int, default=8081, help="port for the viewer server"
    )
    parser.add_argument(
        "--backend", type=str, default="gsplat", help="gsplat, gsplat_legacy, inria"
    )
    parser.add_argument(
        "--pose_dir", type=str, default="results/traj", help="path to the pose data"
    )
    args = parser.parse_args()
    # args.pose_dir = "/home/yuxinyao/batch-cf3dgs/paper_example/tum_rgb_refine_pose_copy/rgbd_dataset_freiburg3_walking_static_seq_390_419/cam_poses.npy "
    print(args)
    assert args.scene_grid % 2 == 1, "scene_grid must be odd"

    cli(main, args, verbose=True)
