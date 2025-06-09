import subprocess
import pickle
import numpy as np
import os 
import time
import json
import glob
from scipy.spatial.transform import Rotation as R
framerate = 30
def velocity_between_traj(traj_ref, traj_est,timestamps):
    # align 
    traj_est.align(traj_ref, correct_scale=True)

    pred_positions = traj_est.positions_xyz
    gt_positions = traj_ref.positions_xyz

    # calculate the velocity difference 
    # get the velocity of the ground truth
    dt = np.diff(timestamps)[:, None]  # Time differences (Nx1)
    pred_velocity = np.diff(pred_positions, axis=0) / dt  # (N-1, 3)
    gt_velocity = np.diff(gt_positions, axis=0) / dt  # (N-1, 3)

    # calculate the velocity difference

    avg_velocity_pred = np.mean(np.linalg.norm(pred_velocity, axis=1))
    avg_velocity_gt = np.mean(np.linalg.norm(gt_velocity, axis=1))

    velocity_diff = np.abs(avg_velocity_gt - avg_velocity_pred)

    return velocity_diff*framerate

def eval_monst3r_gs_poses(seq, result_dir, gt_pose=None,gt_quats=None):
    from dust3r.utils.vo_eval import load_traj, eval_metrics, plot_trajectory, save_trajectory_tum_format, process_directory, calculate_averages
    from evo.core.trajectory import PoseTrajectory3D,PosePath3D

    monst3r_traj_path  = f'{result_dir}/{seq}/pred_traj.txt'
    refined_traj_path = f'{result_dir}/{seq}/refined_pose.txt'
    gs_traj_path = f'{result_dir}/{seq}/gs_pose.txt'
    cf3dgs_traj_path = f'{result_dir}/{seq}/cf3dgs_sem.txt'
    droid_traj_path = f'{result_dir}/{seq}/droid_traj.npy'
    traj_monster = None
    traj_refined = None
    traj_gs = None
    traj_cf3dgs = None
    traj_droid = None
    
    timestamps_mat = np.arange(30).astype(float)

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
            # turn tu wxyz
            refined_pose[:,4:] = refined_pose[:,[7,4,5,6]]
        pose_refined = PosePath3D(
            positions_xyz=refined_pose[:,1:4],
            orientations_quat_wxyz=refined_pose[:,4:])
        traj_refined = PoseTrajectory3D(poses_se3=pose_refined.poses_se3, timestamps=timestamps_mat)
    
    if os.path.exists(gs_traj_path):
        with open(gs_traj_path, 'r') as f:
            gs_pose = f.readlines()
            gs_pose = np.array([list(map(float,pose.split())) for pose in gs_pose])
            gs_pose[:,4:] = gs_pose[:,[7,4,5,6]]
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
    
    if gt_quats is not None: 
        gt_quats= np.array(gt_quats)
        # save traj
        gt_quats_path = f'{result_dir}/{seq}/cam_poses_traj.npy'
        np.save(gt_quats_path, gt_quats)
        traj_gt = PoseTrajectory3D(
            positions_xyz=gt_quats[:,:3],
            orientations_quat_wxyz=gt_quats[:,3:],
            timestamps=np.array(timestamps_mat))
    else:
        pose_gt = PosePath3D(poses_se3=gt_pose)
        traj_gt = PoseTrajectory3D(poses_se3=pose_gt.poses_se3, timestamps=timestamps_mat)

    if os.path.exists(monst3r_traj_path):
        monst3r_velocity_diff = velocity_between_traj(traj_gt, traj_monster, timestamps_mat)
    if os.path.exists(refined_traj_path):
        refined_velocity_diff = velocity_between_traj(traj_gt, traj_refined, timestamps_mat)

    if os.path.exists(gs_traj_path):
        gs_velocity_diff = velocity_between_traj(traj_gt, traj_gs, timestamps_mat)
    if os.path.exists(cf3dgs_traj_path):
        cf3dgs_velocity_diff = velocity_between_traj(traj_gt, traj_cf3dgs, timestamps_mat)
    if os.path.exists(droid_traj_path):
        droid_velocity_diff = velocity_between_traj(traj_gt, traj_droid, timestamps_mat)
    eval_results = {
        "monst3r_velocity_diff": monst3r_velocity_diff if os.path.exists(monst3r_traj_path) else None,
        "refined_velocity_diff": refined_velocity_diff if os.path.exists(refined_traj_path) else None,
        "gs_velocity_diff": gs_velocity_diff if os.path.exists(gs_traj_path) else None,
        "droid_velocity_diff": droid_velocity_diff if os.path.exists(droid_traj_path) else None,
        "cf3dgs_velocity_diff": cf3dgs_velocity_diff if os.path.exists(cf3dgs_traj_path) else None,
    }
    json.dump(eval_results, open(f'{result_dir}/{seq}/eval_velocity_cf3dgs_sem.json', 'w'))

def argparse():
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate the camera poses')
    parser.add_argument('--seq', type=str, default=None, help='sequence name')
    parser.add_argument('--result_dir', type=str, default='batch_test_results', help='output directory')
    parser.add_argument('--meta_dir', type=str, default='batch_test', help='save directory')
    return parser.parse_args()

if __name__ == '__main__':
    args = argparse()
    monst3r_velocity_diff_overall = []
    refine_velocity_diff_overall = []
    gs_velocity_diff_overall = []
    droid_velocity_diff_overall = []
    cf3dgs_velocity_diff_overall = []

    monst3r_rper_velocity_diff_overall = []
    refine_rper_velocity_diff_overall = []
    gs_rper_velocity_diff_overall = []
    droid_rper_velocity_diff_overall = []
    cf3dgs_rper_velocity_diff_overall = []

    with open(args.meta_dir, 'rb') as f:
        save = pickle.load(f)
        
    seq_ind = save['selected_frames']
    gt_cam_poses = save['cam_poses']
    dataset_name = save['dataset']
    gt_cam_quats = save['cam_quats']
    if args.result_dir is not None:
        result_dir = args.result_dir
        num_seq = len(glob.glob(os.path.join(result_dir, 'seq_*')))
    for ind in range(len(seq_ind)):
        seq_dir = os.path.join(result_dir, '{}_seq_{}_{}'.format(dataset_name[ind],seq_ind[ind][0], seq_ind[ind][-1]))
        print("Processing sequence: ",seq_dir)
        if not os.path.exists(seq_dir):
            continue # skip if the sequence not exit yet 

        # check whether the sequence is already processed
        if not os.path.exists(os.path.join(result_dir, '{}_seq_{}_{}'.format(dataset_name[ind],seq_ind[ind][0], seq_ind[ind][-1]), "scene.glb")) and \
            not os.path.exists(os.path.join(result_dir, '{}_seq_{}_{}'.format(dataset_name[ind],seq_ind[ind][0], seq_ind[ind][-1]), "gs_pose.txt")):
            continue
        # evaluate the camera poses
        try:
            eval_monst3r_gs_poses('{}_seq_{}_{}'.format(dataset_name[ind],seq_ind[ind][0], seq_ind[ind][-1]), 
                        result_dir, gt_pose=gt_cam_poses[ind], gt_quats = gt_cam_quats[ind])
        except Exception as e:
            eval_monst3r_gs_poses('{}_seq_{}_{}'.format(dataset_name[ind],seq_ind[ind][0], seq_ind[ind][-1]), 
                        result_dir, gt_pose=gt_cam_poses[ind], gt_quats = gt_cam_quats[ind]) 

        with open(os.path.join(result_dir, '{}_seq_{}_{}'.format(dataset_name[ind],seq_ind[ind][0], seq_ind[ind][-1]), "eval_velocity_cf3dgs_sem.json"), 'r') as f:
            data = json.load(f)
            monst3r_velocity_diff_overall.append(data['monst3r_velocity_diff'])
            refine_velocity_diff_overall.append(data['refined_velocity_diff'])
            gs_velocity_diff_overall.append(data['gs_velocity_diff'])
            droid_velocity_diff_overall.append(data['droid_velocity_diff'])
            cf3dgs_velocity_diff_overall.append(data['cf3dgs_velocity_diff'])


    file_name_overall_ate = "overall_velocity.json"
    print('saving to ', os.path.join(result_dir, file_name_overall_ate))
    # remove all the None in the list 
    monst3r_velocity_diff_overall = [x for x in monst3r_velocity_diff_overall if x is not None]
    refine_velocity_diff_overall = [x for x in refine_velocity_diff_overall if x is not None]
    gs_velocity_diff_overall = [x for x in gs_velocity_diff_overall if x is not None]
    droid_velocity_diff_overall = [x for x in droid_velocity_diff_overall if x is not None]
    cf3dgs_velocity_diff_overall = [x for x in cf3dgs_velocity_diff_overall if x is not None]
    with open(os.path.join(result_dir, file_name_overall_ate), 'w') as f:
        json.dump({
            "monst3r_velocity_diff": np.format_float_positional(np.mean(monst3r_velocity_diff_overall),precision=4, unique=False, fractional=False, trim='k'),
            "refined_velocity_diff": np.format_float_positional(np.mean(refine_velocity_diff_overall),precision=4, unique=False, fractional=False, trim='k'),
            "gs_velocity_diff": np.format_float_positional(np.mean(gs_velocity_diff_overall),precision=4, unique=False, fractional=False, trim='k'),
            "droid_velocity_diff": np.format_float_positional(np.mean(droid_velocity_diff_overall),precision=4, unique=False, fractional=False, trim='k'),
            "cf3dgs_velocity_diff": np.format_float_positional(np.mean(cf3dgs_velocity_diff_overall),precision=4, unique=False, fractional=False, trim='k'),        
        }, f,indent=4)