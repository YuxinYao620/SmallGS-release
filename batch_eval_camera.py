import subprocess
import pickle
import numpy as np
import os 
import time
import json

def eval_monst3r_gs_poses(seq, output_dir):
    from dust3r.utils.vo_eval import load_traj, eval_metrics, plot_trajectory, save_trajectory_tum_format, process_directory, calculate_averages
    from evo.core.trajectory import PoseTrajectory3D,PosePath3D

    monst3r_traj_path  = f'{output_dir}/{seq}/pred_traj.txt'
    refined_traj_path = f'{output_dir}/{seq}/refined_pose.txt'
    gs_traj_path = f'{output_dir}/{seq}/gs_pose.txt'
    gt_path = f'{output_dir}/{seq}/cam_poses.npy'
    gt_pose = np.load(gt_path)
    timestamps_mat = np.arange(gt_pose.shape[0]).astype(float)

    # monst3r pose
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

    
    pose_gt = PosePath3D(poses_se3=gt_pose)
    traj_gt = PoseTrajectory3D(poses_se3=pose_gt.poses_se3, timestamps=timestamps_mat)

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
    
    # record the evaluation results in json
    eval_results = {
        "monst3r_ate": monst3r_ate,
        "monst3r_rpe_trans": monst3r_rpe_trans,
        "monst3r_rpe_rot": monst3r_rpe_rot,
        "refined_ate": refined_ate if os.path.exists(refined_traj_path) else None,
        "refined_rpe_trans": refined_rpe_trans if os.path.exists(refined_traj_path) else None,
        "refined_rpe_rot": refined_rpe_rot if os.path.exists(refined_traj_path) else None,
        "gs_ate": gs_ate if os.path.exists(gs_traj_path) else None,
        "gs_rpe_trans": gs_rpe_trans if os.path.exists(gs_traj_path) else None,
        "gs_rpe_rot": gs_rpe_rot if os.path.exists(gs_traj_path) else None
    }
    json.dump(eval_results, open(f'{output_dir}/{seq}/eval_results_gs_monst3r.json', 'w'))

if __name__ == '__main__':
    # Load the data
    # datasets =['scene_d78_ego1', 'r1_new_f', 'ani1_new_f', 'r1_new_', 'ani10_new_f', 'ani14_new_f', 'scene_d78_ego2', 'scene_d78_3rd', 'ani16_new_', 'ani3_new_', 'ani1_new_']
    datasets = [l for l in os.listdir('/scratch/yy561/pointodyssey/val/') if os.path.isdir(os.path.join('/scratch/yy561/pointodyssey/val/', l))]
    datasets = ['scene_d78_ego1']
    ate_overall_monst3r = []
    ate_overall_refined = []
    ate_overall_gs = []
    for dataset_name in datasets:
        with open('/scratch/yy561/pointodyssey/point_odyssey/cam_poses_{}_30.pkl'.format(dataset_name), 'rb') as f:
            save = pickle.load(f)

        seq_ind = save['selected_frames']
        gt_cam_poses = save['cam_poses']
        #     save['dataset'] = dataset_name
        # save['annotation'] = '{}/{}/anno.npz'.format(data_path, dataset_name)
        # dataset_name = save['dataset']
        annotation = np.load(save['annotation'])
        num_seq = len(seq_ind)
        output_dir = '/scratch/yy561/monst3r/batch_test_results_0.07/{}'.format(dataset_name)
        for ind in range(num_seq):
            
            # seq dir 
            seq_dir = os.path.join(output_dir, 'seq_{}_{}_gs'.format(seq_ind[ind][0], seq_ind[ind][-1]))
            print("Processing sequence: ",seq_dir)
            if not os.path.exists(seq_dir):
                # print("skip ", seq_dir, "as not exist")
                continue # skip if the sequence not exit yet 

            # check whether the sequence is already processed
            if not os.path.exists(os.path.join(output_dir, 'seq_{}_{}_gs'.format(seq_ind[ind][0], seq_ind[ind][-1]), "scene.glb")):
                # print("skip ", output_dir, seq_ind[ind][0], seq_ind[ind][-1], "as not processed")
                continue

            # if not os.path.exists(os.path.join(output_dir, 'seq_{}_{}_gs'.format(seq_ind[ind][0], seq_ind[ind][-1]), "eval_results_gs_monst3r.json")):
            #     try:
            #         print("evaluating ", output_dir, seq_ind[ind][0], seq_ind[ind][-1])
            #         eval_monst3r_gs_poses('seq_{}_{}_gs'.format(seq_ind[ind][0], seq_ind[ind][-1]), 
            #                         output_dir)
            #     except Exception as e:
            #         print("skip ", output_dir, seq_ind[ind][0], seq_ind[ind][-1], "as ", e)
            #         continue
            # else:
            #     print("skip ", output_dir, seq_ind[ind][0], seq_ind[ind][-1])
            try:
                eval_monst3r_gs_poses('seq_{}_{}_gs'.format(seq_ind[ind][0], seq_ind[ind][-1]), 
                            output_dir)
            except Exception as e:
                np.save(os.path.join(output_dir, 'seq_{}_{}_gs'.format(seq_ind[ind][0], seq_ind[ind][-1]), 'cam_poses.npy'), np.array(gt_cam_poses[ind]))
                eval_monst3r_gs_poses('seq_{}_{}_gs'.format(seq_ind[ind][0], seq_ind[ind][-1]), 
                            output_dir) 

            
            # collect the overall ATE

            with open(os.path.join(output_dir, 'seq_{}_{}_gs'.format(seq_ind[ind][0], seq_ind[ind][-1]), "eval_results_gs_monst3r.json"), 'r') as f:
                data = json.load(f)
                ate_overall_monst3r.append(data['monst3r_ate'])
                ate_overall_refined.append(data['refined_ate'])
                ate_overall_gs.append(data['gs_ate'])
            assert ate_overall_monst3r[-1] is not None, breakpoint()

    # average the overall ATE
    print("monst3r ate: ", np.mean(ate_overall_monst3r))
    if not None in ate_overall_refined:
        print("refined ate: ", np.mean(ate_overall_refined))
    if not None in ate_overall_gs:
        print("gs ate: ", np.mean(ate_overall_gs))

    # if np.mean(ate_overall_monst3r) is not None and np.mean(ate_overall_refined) is not None:
    with open(os.path.join(os.path.dirname(output_dir), "overall_ate.json"), 'w') as f:
        json.dump({
            "monst3r_ate": np.mean(ate_overall_monst3r),
            "refined_ate": np.mean(ate_overall_refined) if not None in ate_overall_refined else None,
            "gs_ate": np.mean(ate_overall_gs) if not None in ate_overall_gs else None
        }, f)
    breakpoint()
    #find where is None
    np.where(np.array(ate_overall_monst3r) == None)
    # monst3r ate:  0.008998474463947742
    # refined ate:  0.010698046352635882
