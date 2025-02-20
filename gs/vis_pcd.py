# visualize point cloud

import numpy as np
import open3d as o3d
import os
import torch
file = "/scratch/yy561/monst3r/paper_example/tum_rgb_refine_pose_copy/rgbd_dataset_freiburg3_walking_static_seq_390_419/point_cloud/16_0029.ply"
splats = torch.load("/scratch/yy561/monst3r/paper_example/tum_rgb_refine_pose_copy/rgbd_dataset_freiburg3_walking_static_seq_390_419/ckpts/ckpt_16_to30_rank0.pt")
# pcd = o3d.io.read_point_cloud(file)
color = splats['splats']['sh0']
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(splats['splats']['means'].cpu().numpy())
# turn to 0-255
# sigmoid
color = torch.sigmoid(color)
color = color.cpu().numpy()
# color = (color * 255).astype(np.uint8)
color = color.astype(np.float64)

pcd.colors = o3d.utility.Vector3dVector(color)
o3d.visualization.draw_geometries([pcd])

