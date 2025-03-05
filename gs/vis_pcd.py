# visualize point cloud

import numpy as np
import open3d as o3d
import os
import torch
# # file = "/scratch/yy561/monst3r/paper_example/tum_rgb_refine_pose_copy/rgbd_dataset_freiburg3_walking_static_seq_390_419/gs_use_pts_29.ply"
# # splats = torch.load("/scratch/yy561/monst3r/paper_example/tum_rgb_refine_pose_copy/rgbd_dataset_freiburg3_walking_static_seq_390_419/ckpts/ckpt_16_to30_rank0.pt")
# load = o3d.io.read_point_cloud(file)
# pcd = o3d.geometry.PointCloud()
# pcd.points = load.points
# pcd.colors = load.colors
# # color = splats['splats']['sh0']
# # pcd = o3d.geometry.PointCloud()
# # pcd.points = o3d.utility.Vector3dVector(splats['splats']['means'].cpu().numpy())
# # turn to 0-255
# # sigmoid
# # color = torch.sigmoid(color)
# # color = color.cpu().numpy()
# # # color = (color * 255).astype(np.uint8)
# # color = color.astype(np.float64)

# # pcd.colors = o3d.utility.Vector3dVector(color)
# o3d.visualization.draw_geometries([pcd])
import pickle
file = "temp_pointmap.pkl"
with open(file, 'rb') as f:
    dicti = pickle.load(f)
position = dicti['position'].cpu().numpy().reshape(-1,3)
color = dicti['color'].cpu().numpy().reshape(-1,3)
import imageio
color = imageio.imread("/scratch/yy561/monst3r/paper_example/tum_rgb_refine_pose_copy/rgbd_dataset_freiburg3_walking_static_seq_390_419/frame_0000.png")
# map (-255,255) to (0,255)
color = color + 255
color = color / 2
breakpoint()
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(position)
pcd.colors = o3d.utility.Vector3dVector(color)
o3d.visualization.draw_geometries([pcd])