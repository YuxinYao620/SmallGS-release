import subprocess
import os

# os.system("mkdir tum_02ssim/tum_rgb_refine_pose_intermediate")
# os.system("cp -r tum_paper_test/tum_rgb_refine_pose_intermediate/* tum_02ssim/tum_rgb_refine_pose_intermediate/")

# cmd = "python demo_copy.py --input_dir data/tum/tum_cam_poses_static_gt.pkl --output_dir tum_paper_test/tum_rgb_refine_pose_intermediate/ --gs_pose --gs_refine --use_monst3r_intermediate"
# os.system(cmd)

# cmd = "python demo_copy.py --input_dir data/tum/tum_cam_poses_static_gt.pkl  --output_dir tum_paper_test/tum_rgb_refine_pose_dino16/ --gs_pose --gs_refine --dino --dino_dim 16"
# os.system(cmd)

# cmd = "python demo_copy.py --input_dir data/tum/tum_cam_poses_static_gt.pkl --output_dir tum_02ssim/tum_rgb_refine_pose_intermediate/ --gs_pose --gs_refine --use_monst3r_intermediate"
# os.system(cmd)

cmd = "python demo_copy.py --input_dir data/tum/tum_cam_poses_static_gt_3.pkl  --output_dir tum_02ssim3/tum_rgb_refine_pose_dino16/ --gs_pose --gs_refine --dino --dino_dim 16 --ssim_lambda 0.2 --camera_smoothness_lambda 1"
os.system(cmd)

# cmd = "python demo_copy.py --input_dir data/tum/tum_cam_poses_static_gt_3.pkl  --output_dir tum_02ssim3/tum_rgb_refine_pose --gs_pose --gs_refine --ssim_lambda 0.2 --camera_smoothness_lambda 1"
# os.system(cmd)

