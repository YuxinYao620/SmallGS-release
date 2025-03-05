import os
import numpy as np
from dust3r.utils.vo_eval import load_traj, eval_metrics, plot_trajectory, save_trajectory_tum_format, process_directory, calculate_averages
from evo.core.trajectory import PoseTrajectory3D,PosePath3D
def eval_monst3r_gs_poses(seq, output_dir, gt_pose=None,gt_quats=None,results_path=None):

    monst3r_traj_path  = f'{output_dir}/{seq}/pred_traj.txt'
    refined_traj_path = f'{output_dir}/{seq}/refined_pose.txt'
    gs_traj_path = f'{output_dir}/{seq}/gs_pose2.txt'
    cf3dgs_traj_path = f'{output_dir}/{seq}/cf3dgs.txt'
    droid_traj_path = f'{output_dir}/{seq}/droid_traj.npy'
    traj_monster = None
    traj_refined = None
    traj_gs = None
    traj_cf3dgs = None
    traj_droid = None
    # load the ground truth poses
    # if gt_pose is None:
    #     gt_path = f'{output_dir}/{seq}/cam_poses.npy'
    #     gt_pose = np.load(gt_path)
    # else:
    #     gt_pose = np.array(gt_pose)
    #     gt_path = f'{output_dir}/{seq}/cam_poses.npy'
    #     np.save(gt_path, gt_pose)
    
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
        "droid_rpe_rot": droid_rpe_rot if os.path.exists(droid_traj_path) else None
    }
    import json
    json.dump(eval_results, open(results_path, 'w'))

    return traj_gt, traj_monster, traj_refined, traj_gs, traj_cf3dgs, traj_droid

output_dir = "/scratch/yy561/monst3r/tum_02ssim/test/"
seq = "rgbd_dataset_freiburg3_walking_static_seq_0_29"
gt_pose = np.load(f'{output_dir}/{seq}/cam_poses.npy')
gt_quats = np.load(f'{output_dir}/{seq}/cam_poses_traj.npy')
results_path = "/scratch/yy561/monst3r/tum_02ssim/test/rgbd_dataset_freiburg3_walking_static_seq_0_29/eval_results_gs_monst3r2.json"
traj_gt, traj_monster, traj_refined, traj_gs, traj_cf3dgs, traj_droid = eval_monst3r_gs_poses(seq, output_dir, gt_pose, gt_quats, results_path =results_path )