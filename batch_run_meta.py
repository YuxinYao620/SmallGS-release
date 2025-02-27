import subprocess
import pickle
import numpy as np
import os 
import time
import argparse

# #比较refine 和gs_pose， 用monst3r的intrinsic & depth， indicate refine 可能对修正的效果有限。 
# cmd = "python demo_copy.py --input_dir data/tum/tum_cam_poses_all_0.07.pkl  --output_dir tum_all/tum_rgb_refine_pose/ --gs_pose --gs_refine"
# # subprocess.run(cmd, shell=True)
# os.system(cmd)

#gs pose 和 gs_pose_own,  monst3r的depth & sem 不是很work? 不是很完全。所以使用了lang-sem的mask

# # run after go
# cmd = "python demo_copy.py --input_dir data/tum/tum_cam_poses_all_0.07.pkl  --output_dir test/tum_rgb_gs_own_pose/ --data_factor 1 --gs_pose"
# subprocess.run(cmd, shell=True)

# # 比较raw image， 和用monster intermeidate points. 体现monst3r的pair是有效的？ refine的情况会ok?
# cmd = "python demo_copy.py --input_dir data/tum/tum_cam_poses_all_0.07.pkl  --output_dir tum_all/tum_rgb_refine_pose_intermediate/ --gs_pose --gs_refine"
# # subprocess.run(cmd, shell=True)
# os.system(cmd)

# # # dino image, compare with 3 dim and 6 dim, compare to gs_pose_own, as they don't relie on the intermediate output of monst3r

# cmd = "python demo_copy.py --input_dir data/tum/tum_cam_poses_all_0.07.pkl  --output_dir tum_all/tum_rgb_dino_refine_pose_6/ --data_factor 1 --gs_refine --gs_pose --dino "
# # subprocess.run(cmd, shell=True)
# os.system(cmd)

# cmd = "python demo_copy.py --input_dir data/tum/tum_cam_poses_all_0.07.pkl  --output_dir tum_all/tum_rgb_dino_refine_pose_3/ --data_factor 1 --gs_refine --gs_pose --dino --dino_dim 3"
# # subprocess.run(cmd, shell=True)
# os.system(cmd)

# # gs_refine & gs pose, lower batch size  ssim= 0
# cmd = "CUDA_VISIBLE_DEVICES=1 python demo_copy.py --input_dir data/tum/tum_cam_poses_static_gt.pkl  --output_dir tum_0ssim/tum_rgb_refine_pose/ --gs_pose --gs_refine"
# os.system(cmd)

# os.system('cp -r tum_0ssim/tum_rgb_refine_pose tum_0ssim/tum_rgb_refine_pose_dino16/')
# cmd = "CUDA_VISIBLE_DEVICES=1 python demo_copy.py --input_dir data/tum/tum_cam_poses_static_gt.pkl  --output_dir tum_0ssim/tum_rgb_refine_pose_dino16/ --gs_pose --gs_refine --dino --dino_dim 16"

cmd = "CUDA_VISIBLE_DEVICES=1 python demo_copy.py --input_dir data/tum/tum_cam_poses_static_gt.pkl  --output_dir tum_02ssim/tum_rgb_refine_pose/ --gs_pose --gs_refine"

os.system(cmd)

cmd = "CUDA_VISIBLE_DEVICES=1 python demo_copy.py --input_dir data/tum/tum_cam_poses_static_gt.pkl  --output_dir tum_02ssim/tum_rgb_refine_pose_dino16/ --gs_pose --gs_refine --dino --dino_dim 16"
os.system(cmd)