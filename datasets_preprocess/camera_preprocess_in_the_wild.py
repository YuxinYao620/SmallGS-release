import glob
import os
import shutil
import numpy as np
from scipy.spatial.transform import Rotation as R
from llff_data_utils import batch_parse_llff_poses
from llff_data_utils import load_llff_data
def read_file_list(filename):
    """
    Reads a trajectory from a text file. 
    
    File format:
    The file format is "stamp d1 d2 d3 ...", where stamp denotes the time stamp (to be matched)
    and "d1 d2 d3.." is arbitary data (e.g., a 3D position and 3D orientation) associated to this timestamp. 
    
    Input:
    filename -- File name
    
    Output:
    dict -- dictionary of (stamp,data) tuples
    
    """
    file = open(filename)
    data = file.read()
    lines = data.replace(","," ").replace("\t"," ").split("\n") 
    list = [[v.strip() for v in line.split(" ") if v.strip()!=""] for line in lines if len(line)>0 and line[0]!="#"]
    list = [(float(l[0]),l[1:]) for l in list if len(l)>1]
    return dict(list)

def associate(first_list, second_list, offset, max_difference):
    """
    Associate two dictionaries of (stamp, data). As the time stamps never match exactly, we aim 
    to find the closest match for every input tuple.
    
    Input:
    first_list -- first dictionary of (stamp, data) tuples
    second_list -- second dictionary of (stamp, data) tuples
    offset -- time offset between both dictionaries (e.g., to model the delay between the sensors)
    max_difference -- search radius for candidate generation

    Output:
    matches -- list of matched tuples ((stamp1, data1), (stamp2, data2))
    """
    # Convert keys to sets for efficient removal
    first_keys = set(first_list.keys())
    second_keys = set(second_list.keys())
    
    potential_matches = [(abs(a - (b + offset)), a, b) 
                         for a in first_keys 
                         for b in second_keys 
                         if abs(a - (b + offset)) < max_difference]
    potential_matches.sort()
    matches = []
    for diff, a, b in potential_matches:
        if a in first_keys and b in second_keys:
            first_keys.remove(a)
            second_keys.remove(b)
            matches.append((a, b))
    
    matches.sort()
    return matches

dirs = glob.glob("/scratch/yy561/monst3r/nvidia_data/Nvidia_long/*/")
dirs = sorted(dirs)
root_dir = "/scratch/yy561/monst3r/"


# extract frames
selected_frames = []
selected_frames_path = []
cam_poses = []
save = {"selected_frames":[], "cam_poses":[], "image_paths":[], 'scene_name':[], 'dataset_path':[], "dataset":[]}
for dir in dirs:
    if "Depths" in dir:
        continue
    name = dir.split("/")[-2]
    # dir = dir + name + "/dense/"
    dir = os.path.join(dir, name, "dense")
    dataset_name = dir.split('/')[-3]
    sliding_window = 30
    continue_flag = True

    # extract matched frames
    frames = []
    gt = []

    _, poses, bds, _, i_test, rgb_files, _ = load_llff_data(
        dir,
        height=288,
        num_avg_imgs=sliding_window,
        load_imgs=False,
    )
    intrinsics, c2w_mats = batch_parse_llff_poses(poses)
    h, w = poses[0][:2, -1]
    render_intrinsics, render_c2w_mats = (
        intrinsics,
        c2w_mats,
    )

    frames = rgb_files 
    cam_exts = render_c2w_mats
    # turn quaternions into rotation matrices
    # cam_exts = []
    # for i in range(len(gt)):
    #     # tx ty tz qx qy qz qw
    #     q = np.array([float(x) for x in gt[i][4:]])
    #     t = np.array([float(x) for x in gt[i][1:4]])
    #     cam_ext_i = np.eye(4)
    #     cam_ext_i[:3, :3] = R.from_quat(q, scalar_first = False).as_matrix()
    #     cam_ext_i[:3, 3] = t
    #     cam_exts.append(cam_ext_i)
    num_frames = len(cam_exts)
    # select camera constrained videos
    while continue_flag:
        # find consecutive frames within camera poses as long as possible
        selected_frames = []
        selected_frames_path = []
        cam_poses = []
        indices = range(0, num_frames, 1)
        reset_ind = False

        for index in indices:
            if reset_ind:
                reset_ind = False
                break
            if index + sliding_window//2 >= num_frames:
                break
            if index - sliding_window//2 < 0:
                continue
            cam_middle = cam_exts[index]
            accept_flag = True
            for i in range(index - sliding_window//2, index + sliding_window//2):
                for ls in selected_frames:
                    if i in ls:
                        accept_flag = False
                        break
                if accept_flag == False:
                    break
                if i < 0 or i >= num_frames:
                    break
                else:
                    # find the difference between the middle frame and the current frame
                    rotation_current = cam_exts[i][:3, :3]
                    rotation_middle = cam_middle[:3, :3]
                    # turn the rotation matrix into quaternion
                    q_current = R.from_matrix(rotation_current).as_quat()
                    q_middle = R.from_matrix(rotation_middle).as_quat()
                    # find the difference between the two quaternions
                    theta = 2*np.arccos(np.abs(np.dot(q_current,q_middle.conjugate())))
                    if theta > 0.07:
                        accept_flag = False
                    # check the translation difference, reject large baseline
                    # breakpoint()
                    t_current = cam_exts[i][:3, 3]
                    t_middle = cam_middle[:3, 3]
                    if np.linalg.norm(t_current - t_middle) > 0.01:
                        accept_flag = False
            if accept_flag:
                max_ind = min(num_frames-1, index + sliding_window//2)
                if len(selected_frames) < 1:
                    selected_frames = [[i for i in range(index - sliding_window//2, max_ind)]]
                    cam_poses = [[cam_exts[i] for i in range(index - sliding_window//2, max_ind)]]
                    selected_frames_path = [[frames[i].replace("../","") for i in range(index - sliding_window//2, max_ind)]]
                else:
                    selected_frames.append([i for i in range(index - sliding_window//2, max_ind)]) 
                    cam_poses.append([cam_exts[i] for i in range(index - sliding_window//2, max_ind)])
                    selected_frames_path.append([frames[i].replace("../","") for i in range(index - sliding_window//2, max_ind)])
                # index = max_ind
                # remove indices form list 
                indices = [i for i in indices if i not in range(index - sliding_window//2, max_ind)]
                # print("Number of frames: ", num_frames, "Sliding window: ", sliding_window, "Number of frames left: ", len(indices))
            else:
                accept_flag = True
                # sliding_window -= 5
                # if sliding_window < 15:
                #     continue_flag = False
                #     break
        if len(selected_frames) > 0:
            continue_flag = False
            # save = {}
            # save['selected_frames'] = selected_frames
            # save['cam_poses'] = cam_poses
            # save['dataset'] = dataset_name
            # save['dataset_path'] = dir
            # save['image_paths'] = selected_frames_path
            # save['dataset_type'] = 'tum'
            # print(dir, "\n", "Save Number of frames: ", num_frames, "Sliding window: ", sliding_window)
            # import pickle
            # with open(f"/scratch/yy561/monst3r/data/tum/tum_cam_poses_{dataset_name}_{sliding_window}_theta0.07.pkl", "wb") as f:
            #     pickle.dump(save, f)
            save['dataset'] += [dataset_name for i in range(len(selected_frames))]
            save['dataset_path'] += [dir for i in range(len(selected_frames))]
            save['selected_frames'] += selected_frames
            save['cam_poses'] += cam_poses
            save['image_paths'] += selected_frames_path
            breakpoint()
            print(dir, "\n", "Save Number of frames: ", num_frames, "Sliding window: ", sliding_window, "sequence length: ", len(selected_frames))
            
        
        sliding_window -= 5
        if sliding_window < 10:
            continue_flag = False
            import pickle
            # if len(selected_frames) > 0:
            # #     save = {}
            # #     save['selected_frames'] = selected_frames
            # #     save['cam_poses'] = cam_poses
            # #     save['dataset'] = dataset_name
            # #     save['dataset_type'] = 'tum'
            # #     save['dataset_path'] = dir
            # #     save['image_paths'] = selected_frames_path
            #     # save['dataset'] = dataset_name
            #     save['selected_frames'] += selected_frames
            #     save['cam_poses'] += cam_poses
            #     save['image_paths'] += selected_frames_path


            #     with open(f"tum_cam_poses_{dataset_name}_max.pkl", "wb") as f:
            #             pickle.dump(save, f)
            break

# save = {}
# save['selected_frames'] = selected_frames
# save['cam_poses'] = cam_poses
# save['dataset'] = dataset_name
# save['dataset_path'] = dir
# save['image_paths'] = selected_frames_path
# save['dataset_type'] = 'tum'
print(dir, "\n", "Save Number of frames: ", num_frames, "Sliding window: ", sliding_window)
import pickle
with open(f"/scratch/yy561/monst3r/data/in_the_wild_0.07.pkl", "wb") as f:
    pickle.dump(save, f)
