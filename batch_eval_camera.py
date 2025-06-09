import subprocess
import pickle
import numpy as np
import os 
import time
import json
import glob

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
        monst3r_ate, monst3r_rpe_trans, monst3r_rpe_rot = eval_metrics(
                traj_monster, traj_gt, seq=seq, filename=f'{result_dir}/{seq}/monst3r_eval_metric.txt'
            )
    if os.path.exists(refined_traj_path):
        refined_ate, refined_rpe_trans, refined_rpe_rot = eval_metrics(
                    traj_refined, traj_gt, seq=seq, filename=f'{result_dir}/{seq}/refined_eval_metric.txt'
                )
    if os.path.exists(gs_traj_path):
        gs_ate, gs_rpe_trans, gs_rpe_rot = eval_metrics(
                    traj_gs, traj_gt, seq=seq, filename=f'{result_dir}/{seq}/gs_eval_metric.txt'
                )
    if os.path.exists(cf3dgs_traj_path):
        cf_ate, cf_rpe_trans, cf_rpe_rot = eval_metrics(
                    traj_cf3dgs, traj_gt, seq=seq, filename=f'{result_dir}/{seq}/cf3dgs_eval_metric.txt'
        )   
    if os.path.exists(droid_traj_path):
        droid_ate, droid_rpe, droid_rpe_rot = eval_metrics(
                    traj_droid, traj_gt, seq=seq, filename=f'{result_dir}/{seq}/droid_eval_metric.txt')
    
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
    json.dump(eval_results, open(f'{result_dir}/{seq}/eval_results_gs_monst3r_cf3dgs_sem.json', 'w'))

    return traj_gt, traj_monster, traj_refined, traj_gs, traj_cf3dgs, traj_droid

def argparse():
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate the camera poses')
    parser.add_argument('--seq', type=str, default=None, help='sequence name')
    parser.add_argument('--result_dir', type=str, default='batch_test_results', help='result directory')
    parser.add_argument('--meta_dir', type=str, default='batch_test', help='meta data directory')
    parser.add_argument('--static', action='store_true', help='static sequence')
    parser.add_argument('--xyz', action='store_true', help='xyz only')
    return parser.parse_args()

if __name__ == '__main__':
    args = argparse()

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

    with open(args.meta_dir, 'rb') as f:
        save = pickle.load(f)


    seq_ind = save['selected_frames']
    gt_cam_poses = save['cam_poses']
    dataset_name = save['dataset']
    if args.result_dir is not None:
        result_dir = args.result_dir
        num_seq = len(glob.glob(os.path.join(result_dir, 'seq_*')))
    for ind in range(len(seq_ind)):
        seq_dir = os.path.join(result_dir, '{}_seq_{}_{}'.format(dataset_name[ind],seq_ind[ind][0], seq_ind[ind][-1]))
        print("Processing sequence: ",seq_dir)
        if not os.path.exists(seq_dir):
            continue # skip if the sequence not exit yet 

        if not os.path.exists(os.path.join(result_dir, '{}_seq_{}_{}'.format(dataset_name[ind],seq_ind[ind][0], seq_ind[ind][-1]), "scene.glb")) and \
            not os.path.exists(os.path.join(result_dir, '{}_seq_{}_{}'.format(dataset_name[ind],seq_ind[ind][0], seq_ind[ind][-1]), "gs_pose.txt")):
            continue
        if args.static:
            if not "static" in dataset_name[ind]:
                # if not "xyz" in dataset_name[ind]:
                    continue
            

        # evaluate the camera poses
        try:
            eval_monst3r_gs_poses('{}_seq_{}_{}'.format(dataset_name[ind],seq_ind[ind][0], seq_ind[ind][-1]), 
                        result_dir, gt_pose=gt_cam_poses[ind])
        except Exception as e:
            eval_monst3r_gs_poses('{}_seq_{}_{}'.format(dataset_name[ind],seq_ind[ind][0], seq_ind[ind][-1]), 
                        result_dir, gt_pose=gt_cam_poses[ind])

        
        # collect the overall ATE

        with open(os.path.join(result_dir, '{}_seq_{}_{}'.format(dataset_name[ind],seq_ind[ind][0], seq_ind[ind][-1]), "eval_results_gs_monst3r_cf3dgs_sem.json"), 'r') as f:
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

    # average the overall ATE
    if not None in ate_overall_monst3r:
        print("monst3r ate: ", np.mean(ate_overall_monst3r))
    if not None in ate_overall_refined:
        print("refined ate: ", np.mean(ate_overall_refined))
    if not None in ate_overall_gs:
        print("gs ate: ", np.mean(ate_overall_gs))
    if not None in ate_overall_droid:
        print("droid ate: ", np.mean(ate_overall_droid))
    print("len(ate_overall_monst3r): ", len(ate_overall_monst3r))

    # remove all the None in the list 
    ate_overall_monst3r = [x for x in ate_overall_monst3r if x is not None]
    ate_overall_refined = [x for x in ate_overall_refined if x is not None]
    ate_overall_gs = [x for x in ate_overall_gs if x is not None]
    ate_overall_cf3dgs = [x for x in ate_overall_cf3dgs if x is not None]
    ate_overall_droid = [x for x in ate_overall_droid if x is not None]
    rpe_rot_monst3r = [x for x in rpe_rot_monst3r if x is not None]
    rpe_rot_refined = [x for x in rpe_rot_refined if x is not None]
    rpe_rot_gs = [x for x in rpe_rot_gs if x is not None]
    rpe_rot_cf3dgs = [x for x in rpe_rot_cf3dgs if x is not None]
    rpe_rot_droid = [x for x in rpe_rot_droid if x is not None]
    rpe_trans_monst3r = [x for x in rpe_trans_monst3r if x is not None]
    rpe_trans_refined = [x for x in rpe_trans_refined if x is not None]
    rpe_trans_gs = [x for x in rpe_trans_gs if x is not None]
    rpe_trans_cf3dgs = [x for x in rpe_trans_cf3dgs if x is not None]
    rpe_trans_droid = [x for x in rpe_trans_droid if x is not None]

    file_name_overall_ate = "metric.json"
    print('saving to ', os.path.join(result_dir, file_name_overall_ate))
    with open(os.path.join(result_dir, file_name_overall_ate), 'w') as f:
        json.dump({
            "monst3r_ate": np.format_float_positional(np.mean(ate_overall_monst3r) if not None in ate_overall_monst3r else None, precision=4, unique=False, fractional=False, trim='k'),
            "monst3r_rpe_rot": np.format_float_positional(np.mean(rpe_rot_monst3r) if not None in rpe_rot_monst3r else None, precision=4, unique=False, fractional=False, trim='k'),
            "monst3r_rpe_trans": np.format_float_positional(np.mean(rpe_trans_monst3r) if not None in rpe_trans_monst3r else None, precision=4, unique=False, fractional=False, trim='k'),

            "refined_ate": np.format_float_positional(np.mean(ate_overall_refined) if not None in ate_overall_refined else None, precision=4, unique=False, fractional=False, trim='k'),
            "refined_rpe_rot": np.format_float_positional(np.mean(rpe_rot_refined) if not None in rpe_rot_refined else None, precision=4, unique=False, fractional=False, trim='k'),
            "refined_rpe_trans": np.format_float_positional(np.mean(rpe_trans_refined) if not None in rpe_trans_refined else None, precision=4, unique=False, fractional=False, trim='k'),

            "gs_ate": np.format_float_positional(np.mean(ate_overall_gs) if not None in ate_overall_gs else None, precision=4, unique=False, fractional=False, trim='k'),
            "gs_rpe_rot": np.format_float_positional(np.mean(rpe_rot_gs) if not None in rpe_rot_gs else None, precision=4, unique=False, fractional=False, trim='k'),
            "gs_rpe_trans": np.format_float_positional(np.mean(rpe_trans_gs) if not None in rpe_trans_gs else None, precision=4, unique=False, fractional=False, trim='k'),
        
            "cf3dgs_ate": np.format_float_positional(np.mean(ate_overall_cf3dgs) if not None in ate_overall_cf3dgs else None, precision=4, unique=False, fractional=False, trim='k'),
            "cf3dgs_rpe_rot": np.format_float_positional(np.mean(rpe_rot_cf3dgs) if not None in rpe_rot_cf3dgs else None, precision=4, unique=False, fractional=False, trim='k'),
            "cf3dgs_rpe_trans": np.format_float_positional(np.mean(rpe_trans_cf3dgs) if not None in rpe_trans_cf3dgs else None, precision=4, unique=False, fractional=False, trim='k'),

            "droid_ate": np.format_float_positional(np.mean(ate_overall_droid) if not None in ate_overall_droid else None, precision=4, unique=False, fractional=False, trim='k'),
            "droid_rpe_rot": np.format_float_positional(np.mean(rpe_rot_droid) if not None in rpe_rot_droid else None, precision=4, unique=False, fractional=False, trim='k'),
            "droid_rpe_trans": np.format_float_positional(np.mean(rpe_trans_droid) if not None in rpe_trans_droid else None, precision=4, unique=False, fractional=False, trim='k')
        
        }, f,indent=4)
