import subprocess
import pickle
import numpy as np
import os 
import time
import json
import glob
from scipy.spatial.transform import Rotation as R
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

    avg_velocity_pred = np.mean(np.linalg.norm(pred_velocity, axis=1))
    avg_velocity_gt = np.mean(np.linalg.norm(gt_velocity, axis=1))

    velocity_diff = np.abs(avg_velocity_gt - avg_velocity_pred)

    # angular velocity difference 
    # get the orientation of the ground truth
    pred_quats = traj_est.orientations_quat_wxyz
    gt_quats = traj_ref.orientations_quat_wxyz
    # get the angular velocity
    rotations_pred = R.from_quat(pred_quats,scalar_first=True)
    rot_diff_pred = rotations_pred[:-1].inv() * rotations_pred[1:]
    ang_vel_pred = 2 * rot_diff_pred.as_rotvec() / dt  # (N-1, 3)

    rotations_gt = R.from_quat(gt_quats,scalar_first=True)
    rot_diff_gt = rotations_gt[:-1].inv() * rotations_gt[1:]
    ang_vel_gt = 2 * rot_diff_gt.as_rotvec() / dt  # (N-1, 3)

    avg_ang_velocity_pred = np.mean(np.linalg.norm(ang_vel_pred, axis=1))
    avg_ang_velocity_gt = np.mean(np.linalg.norm(ang_vel_gt, axis=1))

    angular_velocity_diff = np.abs(avg_ang_velocity_gt - avg_ang_velocity_pred)



    return velocity_diff*framerate  , angular_velocity_diff*framerate


def eval_monst3r_gs_poses(seq, output_dir, gt_pose=None,gt_quats=None):
    from dust3r.utils.vo_eval import load_traj, eval_metrics, plot_trajectory, save_trajectory_tum_format, process_directory, calculate_averages
    from evo.core.trajectory import PoseTrajectory3D,PosePath3D

    monst3r_traj_path  = f'{output_dir}/{seq}/pred_traj.txt'
    refined_traj_path = f'{output_dir}/{seq}/refined_pose.txt'
    gs_traj_path = f'{output_dir}/{seq}/gs_pose.txt'
    cf3dgs_traj_path = f'{output_dir}/{seq}/cf3dgs_sem.txt'
    droid_traj_path = f'{output_dir}/{seq}/droid_traj.npy'
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
        gt_quats_path = f'{output_dir}/{seq}/cam_poses_traj.npy'
        np.save(gt_quats_path, gt_quats)
        traj_gt = PoseTrajectory3D(
            positions_xyz=gt_quats[:,:3],
            orientations_quat_wxyz=gt_quats[:,3:],
            timestamps=np.array(timestamps_mat))
    else:
        pose_gt = PosePath3D(poses_se3=gt_pose)
        traj_gt = PoseTrajectory3D(poses_se3=pose_gt.poses_se3, timestamps=timestamps_mat)

    if os.path.exists(monst3r_traj_path):
        monst3r_ate, monst3r_rpe_trans, monst3r_rpe_rot = eval_metrics(
                traj_monster, traj_gt, seq=seq, filename=f'{output_dir}/{seq}/monst3r_eval_metric.txt'
            )
    if os.path.exists(refined_traj_path):
        refined_ate, refined_rpe_trans, refined_rpe_rot = eval_metrics(
                    traj_refined, traj_gt, seq=seq, filename=f'{output_dir}/{seq}/refined_eval_metric.txt'
                )
    if os.path.exists(gs_traj_path):
        gs_ate, gs_rpe_trans, gs_rpe_rot = eval_metrics(
                    traj_gs, traj_gt, seq=seq, filename=f'{output_dir}/{seq}/gs_eval_metric.txt'
                )
    if os.path.exists(cf3dgs_traj_path):
        cf_ate, cf_rpe_trans, cf_rpe_rot = eval_metrics(
                    traj_cf3dgs, traj_gt, seq=seq, filename=f'{output_dir}/{seq}/cf3dgs_eval_metric.txt'
        )   
    if os.path.exists(droid_traj_path):
        droid_ate, droid_rpe, droid_rpe_rot = eval_metrics(
                    traj_droid, traj_gt, seq=seq, filename=f'{output_dir}/{seq}/droid_eval_metric.txt')
    
    # calculate the velocity difference
    if os.path.exists(monst3r_traj_path):
        monst3r_velocity_diff, monst3r_angvel_diff = velocity_between_traj(traj_gt, traj_monster, timestamps_mat)
    if os.path.exists(refined_traj_path):
        refined_velocity_diff, refined_angvel_diff = velocity_between_traj(traj_gt, traj_refined, timestamps_mat)
    if os.path.exists(gs_traj_path):
        gs_velocity_diff, gs_angvel_diff = velocity_between_traj(traj_gt, traj_gs, timestamps_mat)
    if os.path.exists(cf3dgs_traj_path):
        cf_velocity_diff, cf_angvel_diff = velocity_between_traj(traj_gt, traj_cf3dgs, timestamps_mat)
    if os.path.exists(droid_traj_path):
        droid_velocity_diff, droid_angvel_diff = velocity_between_traj(traj_gt, traj_droid, timestamps_mat)

    # record the evaluation results in json
    eval_results = {
        "monst3r_ate": monst3r_ate if os.path.exists(monst3r_traj_path) else None, 
        "monst3r_rpe_trans": monst3r_rpe_trans if os.path.exists(monst3r_traj_path) else None,
        "monst3r_rpe_rot": monst3r_rpe_rot if os.path.exists(monst3r_traj_path) else None,
        "refined_ate": refined_ate if os.path.exists(refined_traj_path) else None,
        "refined_rpe_trans": refined_rpe_trans if os.path.exists(refined_traj_path) else None,
        "refined_rpe_rot": refined_rpe_rot if os.path.exists(refined_traj_path) else None,
        "gs_ate": gs_ate if os.path.exists(gs_traj_path) else None,
        "gs_rpe_trans": gs_rpe_trans if os.path.exists(gs_traj_path) else None,
        "gs_rpe_rot": gs_rpe_rot if os.path.exists(gs_traj_path) else None,
        "cf_ate": cf_ate if os.path.exists(cf3dgs_traj_path) else None,
        "cf_rpe_trans" : cf_rpe_trans if os.path.exists(cf3dgs_traj_path) else None,
        "cf_rpe_rot": cf_rpe_rot if os.path.exists(cf3dgs_traj_path) else None,
        "droid_ate": droid_ate if os.path.exists(droid_traj_path) else None,
        "droid_rpe_trans": droid_rpe if os.path.exists(droid_traj_path) else None,
        "droid_rpe_rot": droid_rpe_rot if os.path.exists(droid_traj_path) else None,
        # velocity difference
        "monst3r_velocity_diff": monst3r_velocity_diff if os.path.exists(monst3r_traj_path) else None,
        "refined_velocity_diff": refined_velocity_diff if os.path.exists(refined_traj_path) else None,
        "gs_velocity_diff": gs_velocity_diff if os.path.exists(gs_traj_path) else None,
        "cf_velocity_diff": cf_velocity_diff if os.path.exists(cf3dgs_traj_path) else None,
        "droid_velocity_diff": droid_velocity_diff if os.path.exists(droid_traj_path) else None,
    }
    json.dump(eval_results, open(f'{output_dir}/{seq}/eval_results_gs_monst3r_cf3dgs_sem_debug.json', 'w'))

    return traj_gt, traj_monster, traj_refined, traj_gs, traj_cf3dgs, traj_droid

def argparse():
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate the camera poses')
    parser.add_argument('--output_dir', type=str, default='/scratch/yy561/monst3r/batch_test_results', help='output directory')
    parser.add_argument('--save_dir', type=str, default='/scratch/yy561/monst3r/batch_test', help='save directory')
    return parser.parse_args()

if __name__ == '__main__':
    args = argparse()
    framerate = 30 # for TUM dynamics
    ate_overall_monst3r = []
    ate_overall_refined = []
    ate_overall_gs = []
    ate_overall_cf3dgs = []
    rpe_rot_monst3r = []
    rpe_rot_refined = []
    rpe_rot_gs = []
    rpe_rot_cf3dgs = []
    rpe_trans_monst3r = []
    rpe_trans_refined = []
    rpe_trans_gs = []
    rpe_trans_cf3dgs = []
    ate_overall_droid = []
    rpe_rot_droid = []
    rpe_trans_droid = []

    monst3r_velocity_diff_overall = []
    refine_velocity_diff_overall = []
    gs_velocity_diff_overall = []
    droid_velocity_diff_overall = []
    cf3dgs_velocity_diff_overall = []

    with open(args.save_dir, 'rb') as f:
        save = pickle.load(f)


    seq_ind = save['selected_frames']
    gt_cam_poses = save['cam_poses']
    dataset_name = save['dataset']
    output_dir = args.output_dir
    num_seq = len(glob.glob(os.path.join(output_dir, 'seq_*')))
    for ind in range(len(seq_ind)):
        seq_dir = os.path.join(output_dir, '{}_seq_{}_{}'.format(dataset_name[ind],seq_ind[ind][0], seq_ind[ind][-1]))
        print("Processing sequence: ",seq_dir)
        if not os.path.exists(seq_dir):
            continue # skip if the sequence not exit yet 

        # check whether the sequence is already processed
        if not os.path.exists(os.path.join(output_dir, '{}_seq_{}_{}'.format(dataset_name[ind],seq_ind[ind][0], seq_ind[ind][-1]), "scene.glb")) and \
            not os.path.exists(os.path.join(output_dir, '{}_seq_{}_{}'.format(dataset_name[ind],seq_ind[ind][0], seq_ind[ind][-1]), "gs_pose.txt")):
            continue
        if not "static" in dataset_name[ind]:
                # only evaluate static sequences
                continue
        
        # evaluate the camera poses
        eval_monst3r_gs_poses('{}_seq_{}_{}'.format(dataset_name[ind],seq_ind[ind][0], seq_ind[ind][-1]), 
                    output_dir, gt_pose=gt_cam_poses[ind])

        # collect the overall ATE
        with open(os.path.join(output_dir, '{}_seq_{}_{}'.format(dataset_name[ind],seq_ind[ind][0], seq_ind[ind][-1]), "eval_results_gs_monst3r_cf3dgs_sem_debug.json"), 'r') as f:
            data = json.load(f)
            ate_overall_monst3r.append(data['monst3r_ate'])
            ate_overall_refined.append(data['refined_ate'])
            ate_overall_gs.append(data['gs_ate'])
            ate_overall_cf3dgs.append(data['cf_ate'])
            rpe_rot_monst3r.append(data['monst3r_rpe_rot'])
            rpe_rot_refined.append(data['refined_rpe_rot'])
            rpe_rot_gs.append(data['gs_rpe_rot'])
            rpe_rot_cf3dgs.append(data['cf_rpe_rot'])
            rpe_trans_monst3r.append(data['monst3r_rpe_trans']),
            rpe_trans_refined.append(data['refined_rpe_trans']),
            rpe_trans_gs.append(data['gs_rpe_trans'])
            rpe_trans_cf3dgs.append(data['cf_rpe_trans'])
            ate_overall_droid.append(data['droid_ate'])
            rpe_rot_droid.append(data['droid_rpe_rot'])
            rpe_trans_droid.append(data['droid_rpe_trans'])
            # velocity difference
            monst3r_velocity_diff_overall.append(data['monst3r_velocity_diff'])
            refine_velocity_diff_overall.append(data['refined_velocity_diff'])
            gs_velocity_diff_overall.append(data['gs_velocity_diff'])
            droid_velocity_diff_overall.append(data['droid_velocity_diff'])
            cf3dgs_velocity_diff_overall.append(data['cf_velocity_diff'])

    file_name_overall_ate = "static_ate_debug.json"
    print('saving to ', os.path.join(output_dir, file_name_overall_ate))
    with open(os.path.join(output_dir, file_name_overall_ate), 'w') as f:
        results = {}
        if not None in ate_overall_monst3r:
            results['monst3r_ate'] = np.format_float_positional(np.mean(ate_overall_monst3r), precision=4, unique=False, fractional=False, trim='k')
            results['monst3r_rpe_rot'] = np.format_float_positional(np.mean(rpe_rot_monst3r), precision=4, unique=False, fractional=False, trim='k')
            results['monst3r_rpe_trans'] = np.format_float_positional(np.mean(rpe_trans_monst3r), precision=4, unique=False, fractional=False, trim='k')
            results['monst3r_velocity_diff'] = np.format_float_positional(np.mean(monst3r_velocity_diff_overall), precision=4, unique=False, fractional=False, trim='k')
        if not None in ate_overall_refined:
            results['refined_ate'] = np.format_float_positional(np.mean(ate_overall_refined), precision=4, unique=False, fractional=False, trim='k')
            results['refined_rpe_rot'] = np.format_float_positional(np.mean(rpe_rot_refined), precision=4, unique=False, fractional=False, trim='k')
            results['refined_rpe_trans'] = np.format_float_positional(np.mean(rpe_trans_refined), precision=4, unique=False, fractional=False, trim='k')
            results['refined_velocity_diff'] = np.format_float_positional(np.mean(refine_velocity_diff_overall), precision=4, unique=False, fractional=False, trim='k')
        if not None in ate_overall_gs:
            results['gs_ate'] = np.format_float_positional(np.mean(ate_overall_gs), precision=4, unique=False, fractional=False, trim='k')
            results['gs_rpe_rot'] = np.format_float_positional(np.mean(rpe_rot_gs), precision=4, unique=False, fractional=False, trim='k')
            results['gs_rpe_trans'] = np.format_float_positional(np.mean(rpe_trans_gs), precision=4, unique=False, fractional=False, trim='k')
            results['gs_velocity_diff'] = np.format_float_positional(np.mean(gs_velocity_diff_overall), precision=4, unique=False, fractional=False, trim='k')
        if not None in ate_overall_cf3dgs:
            results['cf3dgs_ate'] = np.format_float_positional(np.mean(ate_overall_cf3dgs), precision=4, unique=False, fractional=False, trim='k')
            results['cf3dgs_rpe_rot'] = np.format_float_positional(np.mean(rpe_rot_cf3dgs), precision=4, unique=False, fractional=False, trim='k')
            results['cf3dgs_rpe_trans'] = np.format_float_positional(np.mean(rpe_trans_cf3dgs), precision=4, unique=False, fractional=False, trim='k')
            results['cf3dgs_velocity_diff'] = np.format_float_positional(np.mean(cf3dgs_velocity_diff_overall), precision=4, unique=False, fractional=False, trim='k')
        if not None in ate_overall_droid:
            results['droid_ate'] = np.format_float_positional(np.mean(ate_overall_droid), precision=4, unique=False, fractional=False, trim='k')
            results['droid_rpe_rot'] = np.format_float_positional(np.mean(rpe_rot_droid), precision=4, unique=False, fractional=False, trim='k')
            results['droid_rpe_trans'] = np.format_float_positional(np.mean(rpe_trans_droid), precision=4, unique=False, fractional=False, trim='k')
            results['droid_velocity_diff'] = np.format_float_positional(np.mean(droid_velocity_diff_overall), precision=4, unique=False, fractional=False, trim='k')
        json.dump(results, f, indent=4)