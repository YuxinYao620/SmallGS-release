import imageio
import numpy as np

img_path = "/scratch/yy561/monst3r/tum_walking_static/tum_rgb_dino_refine_pose_16/rgbd_dataset_freiburg3_walking_static_seq_390_419/renders/val_1_0014.png"
mask_path = "/scratch/yy561/monst3r/tum_walking_static/tum_rgb_dino_refine_pose_16/rgbd_dataset_freiburg3_walking_static_seq_390_419/semantics/dynamic_mask_14.png"

img = imageio.imread(img_path)
img = img[:,:512,:]
mask = imageio.imread(mask_path)
mask = np.expand_dims(mask, axis=-1)  # Add a new axis to make the mask shape (384, 512, 1)
mask = np.repeat(mask, 3, axis=-1)

mask = ~mask
mask = (mask/255).astype(np.uint8)
masked = img*mask

imageio.imwrite("masked_render_dino_15.png", masked)

# import open3d as o3d
# # pcd = o3d.io.read_point_cloud("/scratch/yy561/monst3r/tum_walking_static/tum_rgb_dino_refine_pose_16/rgbd_dataset_freiburg3_walking_static_seq_390_419/pointclouds/val_1_0015.ply")
# file = "gs_use_pts_180.pkl"
# import pickle
# with open(file, 'rb') as f:
#     data = pickle.load(f)
# pts = data['points'].cpu().numpy().reshape(-1, 3)
# colors = data['color'].cpu().numpy().reshape(-1, 3)
# # -255 - 255 map to 0-255
# breakpoint()
# colors = (colors + 1) /2

# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(pts)
# pcd.colors = o3d.utility.Vector3dVector(colors)

# o3d.visualization.draw_geometries([pcd])
