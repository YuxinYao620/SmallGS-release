# from evo.core.trajectory import PoseTrajectory3D,PosePath3D

# from evo.tools import plot
import copy
import torch
import numpy as np
import os
import pickle
import glob
import subprocess
# def eval_pose_monst3r(self, pose_path,  interpolate=True,disable_plot=False):
#     """Evaluate camera pose."""
#     # pose_path = os.path.join(result_path, 'ep00_init.pth')
#     base_dir = os.path.dirname(pose_path)
#     result_path = os.path.dirname(pose_path)
#     poses = torch.load(pose_path)
#     with open(base_dir+"/monst3r_pred_traj_correct.txt") as f:
#         gt_pose = f.readlines()
#         gt_pose = np.array([list(map(float,pose.split())) for pose in gt_pose])
#     gt_pose = torch.from_numpy(gt_pose)
#     poses_pred = torch.Tensor(poses['global_cam_poses_est'])
#     # poses_gt_c2w = gt_pose
#     # poses_gt = poses_gt_c2w[:len(poses_pred)].clone()
#     output_path = os.path.join(result_path, 'pose_vis_traj.png')
#     # gt_pose = gt_pose[:20,:]
#     # poses_pred = poses_pred[:20,:,:]

#     from evo.core.trajectory import PoseTrajectory3D,PosePath3D

#     from evo.tools import plot
#     import copy
#     from evo.core.metrics import PoseRelation
#     traj_ref = PosePath3D(
#     positions_xyz=gt_pose[:,1:4],
#     orientations_quat_wxyz=gt_pose[:,4:])
    
#     traj_est = PosePath3D(poses_se3=poses_pred)
#     traj_est_aligned = copy.deepcopy(traj_est)
#     traj_est_aligned.align(traj_ref, correct_scale=True,
#                         correct_only_scale=False)
#     traj_est_aligned.align(traj_ref, correct_scale=True,
#                         correct_only_scale=False)
#     fig = plt.figure()
#     traj_by_label = {
#         # "estimate (not aligned)": traj_est,
#         "Ours (aligned)": traj_est_aligned,
#         # "Ground-truth": traj_ref
#         "monst3r": traj_ref
#     }
#     plot_mode = plot.PlotMode.xyz
#     # ax = plot.prepare_axis(fig, plot_mode, 111)
#     ax = fig.add_subplot(111, projection="3d")
#     ax.xaxis.set_tick_params(labelbottom=True)
#     ax.yaxis.set_tick_params(labelleft=True)
#     ax.zaxis.set_tick_params(labelleft=True)
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z')
#     colors = ['r', 'b']
#     styles = ['-', '--']


#     for idx, (label, traj) in enumerate(traj_by_label.items()):
#         plot.traj(ax, plot_mode, traj,
#                 styles[idx], colors[idx], label)
#         # break
#     # plot.trajectories(fig, traj_by_label, plot.PlotMode.xyz)
#     ax.view_init(elev=10., azim=45)
#     plt.tight_layout()
#     # pose_vis_path = os.path.join(os.path.dirname(output_path), 'pose_vis.png')
#     pose_vis_path = output_path 
#     fig.savefig(pose_vis_path)

#     dic = {
#         'traj_est_aligned': traj_est_aligned.positions_xyz,
#         'traj_ref': traj_ref.positions_xyz,
#         'traj_est_aligned_wxyz': traj_est_aligned.orientations_quat_wxyz,
#         'traj_ref_wxyz': traj_ref.orientations_quat_wxyz
#     }
#     # breakpoint()
#     torch.save(dic, os.path.join(self.ckpt_dir, "aligned_pose.pt"))
#     print("save to ", os.path.join(self.ckpt_dir, "aligned_pose.pt"))
if __name__ == '__main__':
    # datasets = ['cnb_dlab_0215_ego2','cnb_dlab_0225_ego2','egobody_3rd']
    datasets =['scene_d78_ego1', 'r1_new_f', 'ani1_new_f', 'r1_new_', 'ani10_new_f', 'ani14_new_f', 'scene_d78_ego2', 'scene_d78_3rd', 'ani16_new_', 'ani3_new_', 'ani1_new_']
    for dataset_name in datasets:
        print("Processing dataset: ", dataset_name)
        with open('/scratch/yy561/pointodyssey/point_odyssey/cam_poses_{}.pkl'.format(dataset_name), 'rb') as f:
            save = pickle.load(f)

        # ls all output dirs under seq 
        output_dir = '/scratch/yy561/monst3r/batch_test_results/{}'.format(dataset_name)
        seq_dir = glob.glob(output_dir + '/*')

        result_camera_dir = '/scratch/yy561/monst3r/batch_test_results/{}/cameras'.format(dataset_name)
        if not os.path.exists(result_camera_dir):
            os.makedirs(result_camera_dir, exist_ok=True)

        for seq in seq_dir:
            # download camera gt
            gt_camera = seq + '/cam_poses.npy'
            pred_camera = seq + '/pred_traj.txt'

            seq_name = seq.split('/')[-1]
            seq_cam_dir = os.path.join(result_camera_dir, seq_name)
            os.makedirs(seq_cam_dir, exist_ok=True)
            cmd = 'cp {} {}/'.format(gt_camera, seq_cam_dir)
            subprocess.run(cmd, shell=True)
            cmd = 'cp {} {}/'.format(pred_camera, seq_cam_dir)
            subprocess.run(cmd, shell=True)
            print(cmd)
            
