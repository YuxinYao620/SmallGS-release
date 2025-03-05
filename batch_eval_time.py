import subprocess
import pickle
import numpy as np
import os 
import time
import json
import glob
import re
def eval_time(seq, output_dir, gt_pose=None,gt_quats=None, transformed_quats_poses=None):
    from dust3r.utils.vo_eval import load_traj, eval_metrics, plot_trajectory, save_trajectory_tum_format, process_directory, calculate_averages
    from evo.core.trajectory import PoseTrajectory3D,PosePath3D

    gs_time_path = os.path.join(output_dir, seq, 'cf3dgs_time.txt')
    try:
        with open(gs_time_path, "r") as file:
            lines = file.readlines()
        last_gs_time = None
        for line in lines:
            match = re.search(r"Duration:\s*([\d\.]+)", line)
            if match:
                last_gs_time = float(match.group(1))  # Keep updating with the latest match

        # Print the last extracted value
        if last_gs_time is not None:
            print(last_gs_time)
        gs_time.append(last_gs_time)
    except:
        print("GS time not found")


def argparse():
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate the camera poses')
    parser.add_argument('--seq', type=str, default=None, help='sequence name')
    parser.add_argument('--output_dir', type=str, default='/scratch/yy561/monst3r/batch_test_results', help='output directory')
    parser.add_argument('--save_dir', type=str, default='/scratch/yy561/monst3r/batch_test', help='save directory')
    parser.add_argument('--static', action='store_true', help='static sequence')
    parser.add_argument('--xyz', action='store_true', help='xyz only')
    return parser.parse_args()

if __name__ == '__main__':
    args = argparse()

    # Load the data
    # datasets =['scene_d78_ego1', 'r1_new_f', 'ani1_new_f', 'r1_new_', 'ani10_new_f', 'ani14_new_f', 'scene_d78_ego2', 'scene_d78_3rd', 'ani16_new_', 'ani3_new_', 'ani1_new_']
    # datasets = [l for l in os.listdir('/scratch/yy561/pointodyssey/val/') if os.path.isdir(os.path.join('/scratch/yy561/pointodyssey/val/', l))]
    # datasets = ['scene_d78_ego1']
    gs_time = [] 
    # for dataset_name in datasets:

    # if "pkl" in args.save_dir:   # input_dir is a metadata file
    #     import pickle
    #     # find parent dir 
    #     parent_dir = os.path.dirname(args.save_dir)
    #     pkl_paths = [os.path.join(parent_dir, f) for f in os.listdir(parent_dir) if f.endswith('.pkl')]

    # for pkl_path in pkl_paths:
    if True:

        # with open('/scratch/yy561/pointodyssey/point_odyssey/cam_poses_{}_30.pkl'.format(dataset_name), 'rb') as f:
        #     save = pickle.load(f)
        with open(args.save_dir, 'rb') as f:
            save = pickle.load(f)
        # if args.save_dir is not None:
        #     with open(os.path.join(args.save_dir), 'rb') as f:
        #         save = pickle.load(f)


        seq_ind = save['selected_frames']
        gt_cam_poses = save['cam_poses']
        dataset_name = save['dataset']
        gt_cam_quats = save['cam_quats']
        gt_transformed_quats_poses = save['transformed_quats_poses'] if 'transformed_quats_poses' in save else None

        # num_seq = len(os.listdir('/scratch/yy561/monst3r/batch_test_results_0.07_puregs/{}'.format(dataset_name)))
        # output_dir = '/scratch/yy561/monst3r/batch_test_results_0.07_puregs/{}'.format(dataset_name)
        # num_seq = len(os.listdir('/scratch/yy561/monst3r/tum_result_test'))-1
        # output_dir = '/scratch/yy561/monst3r/tum_result_test'
        if args.output_dir is not None:
            output_dir = args.output_dir
            # num_seq = len(os.listdir(output_dir))
            num_seq = len(glob.glob(os.path.join(output_dir, 'seq_*')))
        # for ind in range(num_seq):
        for ind in range(len(seq_ind)):
            # seq dir 
            # breakpoint()
            seq_dir = os.path.join(output_dir, '{}_seq_{}_{}'.format(dataset_name[ind],seq_ind[ind][0], seq_ind[ind][-1]))
            print("Processing sequence: ",seq_dir)
            if not os.path.exists(seq_dir):
                continue # skip if the sequence not exit yet 
            # check whether the sequence is already processed
            if not os.path.exists(os.path.join(output_dir, '{}_seq_{}_{}'.format(dataset_name[ind],seq_ind[ind][0], seq_ind[ind][-1]), "scene.glb")) and \
                not os.path.exists(os.path.join(output_dir, '{}_seq_{}_{}'.format(dataset_name[ind],seq_ind[ind][0], seq_ind[ind][-1]), "gs_pose.txt")):
                continue
            if args.static:
                if not "static" in dataset_name[ind]:
                    # if not "xyz" in dataset_name[ind]:
                        continue


            # evaluate the camera poses
            # try:
            eval_time('{}_seq_{}_{}'.format(dataset_name[ind],seq_ind[ind][0], seq_ind[ind][-1]), 
                        output_dir, gt_pose=gt_cam_poses[ind], gt_quats = gt_cam_quats[ind])
            
            # except Exception as e:
            #     eval_monst3r_gs_poses('{}_seq_{}_{}'.format(dataset_name[ind],seq_ind[ind][0], seq_ind[ind][-1]), 
            #                 output_dir, gt_pose=gt_cam_poses[ind], gt_quats = gt_cam_quats[ind])

            
            # collect the overall ATE
    #average time 
    print("Average time: ", np.mean(gs_time))




