import subprocess
import os

# #比较refine 和gs_pose， 用monst3r的intrinsic & depth， indicate refine 可能对修正的效果有限。 
# # cmd = "python demo_copy.py --input_dir data/tum/tum_cam_poses_all_0.07_2test.pkl  --output_dir tum_walking_static/tum_rgb_refine_pose/ --gs_pose --gs_refine"
# # os.system(cmd)

# #gs pose 和 gs_pose_own,  monst3r的depth & sem 不是很work? 不是很完全。所以使用了lang-sem的mask

# # # run after go
# # cmd = "python demo_copy.py --input_dir data/tum/tum_cam_poses_all_0.07.pkl  --output_dir test/tum_rgb_gs_own_pose/ --data_factor 1 --gs_pose"
# # subprocess.run(cmd, shell=True)
# #fix error
# cmd = "cp -r tum_walking_static/tum_rgb_refine_pose_intermediate/ tum_walking_static/tum_rgb_refine_pose_intermediate2/"
# os.system(cmd)
# cmd = "python demo_copy.py --input_dir data/tum/tum_cam_poses_all_0.07_2test.pkl --output_dir tum_walking_static/tum_rgb_refine_pose_intermediate2/ --gs_pose --gs_refine --use_monst3r_intermediate"
# os.system(cmd)
# # # 比较raw image， 和用monster intermeidate points. 体现monst3r的pair是有效的？ refine的情况会ok?
# cmd = "python demo_copy.py --input_dir data/tum/tum_cam_poses_all_0.07_rest_except_static_walking.pkl --output_dir tum_walking_static/tum_rgb_refine_pose_intermediate/ --gs_pose --gs_refine --use_monst3r_intermediate"
# os.system(cmd)

# # dino image, compare with 3 dim and 6 dim, compare to gs_pose_own, as they don't relie on the intermediate output of monst3r

# cmd = "python demo_copy.py --input_dir data/tum/tum_cam_poses_all_0.07_rest_except_static_walking.pkl --output_dir tum_walking_static/tum_rgb_dino_refine_pose_6/ --data_factor 1 --gs_refine --gs_pose --dino "
# os.system(cmd)

# cmd = "python demo_copy.py --input_dir data/tum/tum_cam_poses_all_0.07_rest_except_static_walking.pkl --output_dir tum_walking_static/tum_rgb_dino_refine_pose_3/ --data_factor 1 --gs_refine --gs_pose --dino --dino_dim 3"
# os.system(cmd)


# cf3dgs 
# python run_cf3dgs.py -s data/Tanks/Francis --mode monst3r --data_type monst3r # hardcode the paths 

# cmd = "cp -r /scratch/yy561/monst3r/pointodyssey/monst3r/* pointodyssey/refine_pose/"
# os.system(cmd)
cmd = "python demo_copy.py --input_dir data/pointodyssey.pkl --output_dir pointodyssey/refine_pose/ --gs_pose --gs_refine"
os.system(cmd)

cmd = "cp -r /scratch/yy561/monst3r/pointodyssey/refine_pose/* pointodyssey/refine_pose_intermediate/"
os.system(cmd)
cmd = "python demo_copy.py --input_dir  data/pointodyssey.pkl --output_dir pointodyssey/refine_pose_intermediate/ --gs_pose --gs_refine --use_monst3r_intermediate"
os.system(cmd)


# dino image, compare with 3 dim and 6 dim, compare to gs_pose_own, as they don't relie on the intermediate output of monst3r
cmd = "cp -r /scratch/yy561/monst3r/pointodyssey/refine_pose/* pointodyssey/dino_refine_pose_3/"
os.system(cmd)
cmd = "python demo_copy.py --input_dir  data/pointodyssey.pkl --output_dir pointodyssey/refine_pose_dino_3/ --gs_pose --gs_refine --dino --dino_dim 3"
os.system(cmd)
cmd = "cp -r /scratch/yy561/monst3r/pointodyssey/refine_pose/* pointodyssey/dino_refine_pose_16/"
os.system(cmd)
cmd = "python demo_copy.py --input_dir  data/pointodyssey.pkl --output_dir pointodyssey/refine_pose_dino_16/ --gs_pose --gs_refine --dino --dino_dim 16"
os.system(cmd)