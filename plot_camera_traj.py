from evo.core.trajectory import PoseTrajectory3D,PosePath3D

from evo.tools import plot
import copy
from evo.core.metrics import PoseRelation
from batch_eval_camera import eval_monst3r_gs_poses
import matplotlib.pyplot as plt
import os
import numpy as np
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter


def argparse():
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate the camera poses')
    parser.add_argument('--file_name', type=str, help='Path to the file containing the camera poses')


    return parser.parse_args()

def argparse():
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate the camera poses')
    parser.add_argument('--output_dir', type=str, help='Path to the file containing the camera poses of monst3r, refined, gs, cf3dgs, droid')
    parser.add_argument('--gs_dir', type=str, help='Path to the file containing the camera poses of Smallgs')
    parser.add_argument('--refined_dir', type=str, help='Path to the file containing the camera poses of refine(smallGS)')
    return parser.parse_args()

if __name__ == "__main__":
    args = argparse()
    import pickle
    with open("data/tum/tum_cam_poses_static_gt.pkl", 'rb') as f:
        save = pickle.load(f)
    seq_ind = save['selected_frames']
    gt_cam_poses = save['cam_poses']
    dataset_name = save['dataset']
    output_dir = args.output_dir
    refined_dir = args.refined_dir
    gs_dir = args.gs_dir

    for ind in range(len(seq_ind)):
        seq = '{}_seq_{}_{}'.format(dataset_name[ind],seq_ind[ind][0], seq_ind[ind][-1])
        monst3r_output_dir = output_dir
        gt_pose_dir = output_dir
        cf3dgs_output_dir = output_dir
        droid_pose = output_dir
        refined_dir = args.refined_dir
        gs_dir = args.gs_dir

        monst3r_traj_path  = f'{output_dir}/{seq}/pred_traj.txt'
        refined_traj_path = f'{refined_dir}/{seq}/refined_pose.txt'
        gs_traj_path = f'{gs_dir}/{seq}/gs_pose.txt'
        cf3dgs_traj_path = f'{output_dir}/{seq}/cf3dgs.txt'
        droid_traj_path = f'{output_dir}/{seq}/droid_traj.npy'
        traj_monster = None
        traj_refined = None
        traj_gs = None
        traj_cf3dgs = None
        traj_droid = None

        gt_path = f'{output_dir}/{seq}/cam_poses_traj.npy'
        gt_pose = np.load(gt_path) 
        timestamps_mat = np.arange(gt_pose.shape[0]).astype(float)

        # monst3r pose
        if os.path.exists(monst3r_traj_path):
            with open(monst3r_traj_path, 'r') as f:
                monster_pose = f.readlines()
                monster_pose = np.array([list(map(float,pose.split())) for pose in monster_pose])
            
            pose_monster = PosePath3D(
                positions_xyz=monster_pose[:,1:4],
                orientations_quat_wxyz=monster_pose[:,4:])
            traj_monster = PoseTrajectory3D(poses_se3=pose_monster.poses_se3, timestamps=timestamps_mat)
        # refined pose
        if os.path.exists(refined_traj_path):
            with open(refined_traj_path, 'r') as f:
                refined_pose = f.readlines()
                refined_pose = np.array([list(map(float,pose.split())) for pose in refined_pose])
            pose_refined = PosePath3D(
                positions_xyz=refined_pose[:,1:4],
                orientations_quat_wxyz=refined_pose[:,4:])
            traj_refined = PoseTrajectory3D(poses_se3=pose_refined.poses_se3, timestamps=timestamps_mat)
        
        if os.path.exists(gs_traj_path):
            with open(gs_traj_path, 'r') as f:
                gs_pose = f.readlines()
                gs_pose = np.array([list(map(float,pose.split())) for pose in gs_pose])
            pose_gs = PosePath3D(
                positions_xyz=gs_pose[:,1:4],
                orientations_quat_wxyz=gs_pose[:,4:])
            traj_gs = PoseTrajectory3D(poses_se3=pose_gs.poses_se3, timestamps=timestamps_mat)
        if os.path.exists(cf3dgs_traj_path):
            with open(cf3dgs_traj_path, 'r') as f:
                cf3dgs_pose = f.readlines()
                cf3dgs_pose = np.array([list(map(float,pose.split())) for pose in cf3dgs_pose])
            pose_cf3dgs = PosePath3D(
                positions_xyz=cf3dgs_pose[:,1:4],
                orientations_quat_wxyz=cf3dgs_pose[:,4:])
            traj_cf3dgs = PoseTrajectory3D(poses_se3=pose_cf3dgs.poses_se3, timestamps=timestamps_mat)
        if os.path.exists(droid_traj_path):
            droid_pose = np.load(droid_traj_path)
            traj_droid = PoseTrajectory3D(
                positions_xyz=droid_pose[:,:3],
                orientations_quat_wxyz=droid_pose[:,3:],
                timestamps=np.array(timestamps_mat))
        
            
        # pose_gt = PosePath3D(poses_se3=gt_pose)
        pose_gt = PoseTrajectory3D(
            positions_xyz=gt_pose[:,:3],
            orientations_quat_wxyz=gt_pose[:,3:],
            timestamps=np.array(timestamps_mat))
        traj_gt = PoseTrajectory3D(poses_se3=pose_gt.poses_se3, timestamps=timestamps_mat)

        traj_monster.align(traj_gt, correct_scale=True)
        traj_refined.align(traj_gt, correct_scale=True)
        traj_gs.align(traj_gt, correct_scale=True)
        traj_cf3dgs.align(traj_gt, correct_scale=True)
        traj_droid.align(traj_gt, correct_scale=True)

        labels = ["Ground-truth", "Monst3r", "Refined", "SmallGS", "CF3DGS", "Droid"]
        styles = [':', '--', '-.', '-', '-.', '--']
        colors = ['r', 'b', 'g', 'c', 'm', 'y']

        colors = {'Ground-truth': 'r', 'Monst3r': 'c', 'Refined': 'g', 'SmallGS': 'b', 'CF3DGS': 'm', 'Droid': 'y'}
        styles = {'Ground-truth': ':', 'Monst3r': '--', 'Refined': '-.', 'SmallGS': '-', 'CF3DGS': '-.', 'Droid': '--'}

        for timestamp in timestamps_mat:
            timestamp = int(timestamp)+1
            plot_gt = PoseTrajectory3D(
                positions_xyz=traj_gt.positions_xyz[:timestamp],
                orientations_quat_wxyz=traj_gt.orientations_quat_wxyz[:timestamp],
                timestamps=np.array(timestamps_mat[:timestamp]))
            plot_monster = PoseTrajectory3D(
                positions_xyz=traj_monster.positions_xyz[:timestamp],
                orientations_quat_wxyz=traj_monster.orientations_quat_wxyz[:timestamp],
                timestamps=np.array(timestamps_mat[:timestamp]))
            plot_refined = PoseTrajectory3D(
                positions_xyz=traj_refined.positions_xyz[:timestamp],
                orientations_quat_wxyz=traj_refined.orientations_quat_wxyz[:timestamp],
                timestamps=np.array(timestamps_mat[:timestamp]))
            plot_gs = PoseTrajectory3D(
                positions_xyz=traj_gs.positions_xyz[:timestamp],
                orientations_quat_wxyz=traj_gs.orientations_quat_wxyz[:timestamp],
                timestamps=np.array(timestamps_mat[:timestamp]))
            plot_cf3dgs = PoseTrajectory3D(
                positions_xyz=traj_cf3dgs.positions_xyz[:timestamp],
                orientations_quat_wxyz=traj_cf3dgs.orientations_quat_wxyz[:timestamp],
                timestamps=np.array(timestamps_mat[:timestamp]))
            plot_droid = PoseTrajectory3D(
                positions_xyz=traj_droid.positions_xyz[:timestamp],
                orientations_quat_wxyz=traj_droid.orientations_quat_wxyz[:timestamp],
                timestamps=np.array(timestamps_mat[:timestamp]))

            traj_by_label = {
            "Ground-truth": plot_gt,
            "Monst3r": plot_monster,
            # "Refined": traj_refined,
            # "CF3DGS": traj_cf3dgs,
            # "Droid": traj_droid,
            "SmallGS": plot_gs,
            }
            plot_mode = plot.PlotMode.xyz
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            ax.xaxis.set_tick_params(labelbottom=False)
            ax.yaxis.set_tick_params(labelleft=False)
            ax.zaxis.set_tick_params(labelleft=False)
            for idx, (label, traj) in enumerate(traj_by_label.items()):
                print(f"Plotting {label}")
                # try:
                plot.traj(ax, plot_mode, traj,
                        styles[label], colors[label], label)
            ax.legend(fontsize=16)
            ax.view_init(elev=10., azim=45)
            plt.tight_layout()
            output_path = os.path.join(output_dir, "paper_monst3r_gs_final2","{}_{}".format(seq_ind[ind][0], seq_ind[ind][-1]), "{}_{}.png".format(seq,timestamp))

            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            fig.savefig(output_path)
