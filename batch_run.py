import subprocess
import pickle
import numpy as np
import os 
import time
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser_url = parser.add_mutually_exclusive_group()
    parser.add_argument("--gs_pose", action='store_true', default=False,
                            help="use gs estimated camera poses")

    parser.add_argument("--gs_refine", action='store_true', default=False,
                            help="use gs refined camera poses")
    parser.add_argument("--save_dir", type=str, default='/scratch/yy561/monst3r/batch_test')
    parser.add_argument("--output_dir", type=str, default='/scratch/yy561/monst3r/batch_test_results')
    args = parser.parse_args()
    # Load the data
    datasets =['scene_d78_ego1']
    # , 'r1_new_f', 'ani1_new_f', 'r1_new_', 'ani10_new_f', 'ani14_new_f', 'scene_d78_ego2', 'scene_d78_3rd', 'ani16_new_', 'ani3_new_', 'ani1_new_']
    # datasets = [l for l in os.listdir('/scratch/yy561/pointodyssey/val/') if os.path.isdir(os.path.join('/scratch/yy561/pointodyssey/val/', l))]

    for dataset_name in datasets:
        with open('/scratch/yy561/pointodyssey/point_odyssey/cam_poses_{}.pkl'.format(dataset_name), 'rb') as f:
        # with open('/scratch/yy561/pointodyssey/point_odyssey/cam_poses_{}_45_theta0.07.pkl'.format(dataset_name), 'rb') as f:
            save = pickle.load(f)
        seq_ind = save['selected_frames']
        gt_cam_poses = save['cam_poses']
        #     save['dataset'] = dataset_name
        # save['annotation'] = '{}/{}/anno.npz'.format(data_path, dataset_name)
        # dataset_name = save['dataset']
        annotation = np.load(save['annotation'])
        num_seq = len(seq_ind)

        data_dir = '/scratch/yy561/pointodyssey/val/{}'.format(dataset_name)
        save_dir = '{}/{}'.format(args.save_dir, dataset_name) 
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        image_dir = os.path.join(save_dir, 'images')
        if not os.path.exists(image_dir):
            os.makedirs(image_dir, exist_ok=True)

        output_dir = '{}/{}'.format(args.output_dir, dataset_name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        for ind in range(len(seq_ind)):

            # seq dir 
            seq_dir = os.path.join(image_dir, 'seq_{}_{}'.format(seq_ind[ind][0], seq_ind[ind][-1]))
            print("Processing sequence: ",seq_dir)
            if not os.path.exists(seq_dir):
                os.makedirs(seq_dir, exist_ok=False)
            

            if len(os.listdir(seq_dir)) == 0: 
                for frame_ind in seq_ind[ind]:
                    # cp images to the seq dir
                    frame_path = os.path.join(data_dir, 'rgbs', 'rgb_{:05d}.jpg'.format(frame_ind))

                    # if os.path.exists(seq_dir):
                    #     cmd = ''
                    #     continue
                    # else:
                    cmd = 'cp {} {}'.format(frame_path, seq_dir)
                    subprocess.run(cmd, shell=True)


            # Construct the command
            cmd = 'python demo_copy.py'
            cmd += ' --input {}'.format(seq_dir)
            cmd += ' --output_dir {}'.format(output_dir)
            cmd += ' --seq_name {}'.format('seq_{}_{}_gs'.format(seq_ind[ind][0], seq_ind[ind][-1]))
            cmd += ' --gs_pose' if args.gs_pose else ''
            cmd += ' --gs_refine' if args.gs_refine else ''
            cmd += ' --camera_smoothness_lambda 1'
            print(cmd)

            # check whether exist dir, if is, skip 
            # Execute the command
            # time it 
            start_time = time.time()
            subprocess.run(cmd, shell=True)
            elapsed_time = time.time() - start_time
            print("Elapsed time: ", elapsed_time)
            #record the time and save it to a file
            try:
                with open(os.path.join(output_dir, 'seq_{}_{}_gs'.format(seq_ind[ind][0], seq_ind[ind][-1]), 'time.txt'), 'a') as f:
                    f.write('elapsed_time: {}'.format(elapsed_time))
            except FileNotFoundError:
                os.makedirs(os.path.join(output_dir, 'seq_{}_{}_gs'.format(seq_ind[ind][0], seq_ind[ind][-1])), exist_ok=True)
                with open(os.path.join(output_dir, 'seq_{}_{}_gs'.format(seq_ind[ind][0], seq_ind[ind][-1]), 'time.txt'), 'a') as f:
                    f.write('elapsed_time: {}'.format(elapsed_time))
            # save the camera poses
            np.save(os.path.join(output_dir, 'seq_{}_{}_gs'.format(seq_ind[ind][0], seq_ind[ind][-1]), 'cam_poses.npy'), np.array(gt_cam_poses[ind]))
            print("finished processing sequence: ", seq_dir)
