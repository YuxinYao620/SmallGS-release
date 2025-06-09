import glob
import os
import shutil
import numpy as np
from scipy.spatial.transform import Rotation as R

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

dirs = glob.glob("data/tum/*/")
dirs = sorted(dirs)

# extract frames
selected_frames = []
selected_frames_path = []
cam_poses = []
save = {"selected_frames":[], "cam_poses":[], "image_paths":[], 'scene_name':[], 'dataset_path':[], "dataset":[], "cam_quats":[]}
for dir in dirs:
    if "static" not in dir:
        continue
    dataset_name = dir.split('/')[-2]
    sliding_window = 30
    continue_flag = True

    # extract matched frames
    frames = []
    gt = []
    first_file = dir + 'rgb.txt'
    second_file = dir + 'groundtruth.txt'

    first_list = read_file_list(first_file)
    second_list = read_file_list(second_file)
    matches = associate(first_list, second_list, 0.0, 0.02)

    for a,b in matches:
        frames.append(dir + first_list[a][0])
        gt.append([b]+second_list[b])

    gt_traj_func= lambda img_path, anno_path, seq: os.path.join(img_path, seq, 'groundtruth_90.txt'),
    
    # turn quaternions into rotation matrices
    cam_exts = []
    cam_quats = []
    for i in range(len(gt)):
        # tx ty tz qx qy qz qw
        q = np.array([float(x) for x in gt[i][4:]])
        q = np.array([q[3], q[0], q[1], q[2]])

        t = np.array([float(x) for x in gt[i][1:4]])
        cam_ext_i = np.eye(4)
        cam_ext_i[:3, :3] = R.from_quat(q, scalar_first = True).as_matrix()
        cam_ext_i[:3, 3] = t
        cam_exts.append(cam_ext_i)
        cam_quats.append(np.array(np.concatenate([t, q])))

        

    num_frames = len(gt)
    # select camera constrained videos
    while continue_flag:
        # find consecutive frames within camera poses as long as possible
        selected_frames = []
        selected_frames_path = []
        cam_poses = []
        cam_quats_pose = []
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
                    t_current = cam_exts[i][:3, 3]
                    t_middle = cam_middle[:3, 3]
                    if np.linalg.norm(t_current - t_middle) > 0.1:
                        accept_flag = False
            if accept_flag:
                max_ind = min(num_frames-1, index + sliding_window//2)
                if len(selected_frames) < 1:
                    selected_frames = [[i for i in range(index - sliding_window//2, max_ind)]]
                    cam_poses = [[cam_exts[i] for i in range(index - sliding_window//2, max_ind)]]
                    cam_quats_pose = [[cam_quats[i] for i in range(index - sliding_window//2, max_ind)]]
                    selected_frames_path = [[frames[i].replace("../","") for i in range(index - sliding_window//2, max_ind)]]
                else:
                    selected_frames.append([i for i in range(index - sliding_window//2, max_ind)]) 
                    cam_poses.append([cam_exts[i] for i in range(index - sliding_window//2, max_ind)])
                    cam_quats_pose.append([cam_quats[i] for i in range(index - sliding_window//2, max_ind)])
                    selected_frames_path.append([frames[i].replace("../","") for i in range(index - sliding_window//2, max_ind)])
                # remove indices form list 
                indices = [i for i in indices if i not in range(index - sliding_window//2, max_ind)]
            else:
                accept_flag = True
        if len(selected_frames) > 0:
            continue_flag = False
            save['dataset'] += [dataset_name for i in range(len(selected_frames))]
            save['dataset_path'] += [dir for i in range(len(selected_frames))]
            save['selected_frames'] += selected_frames
            save['cam_poses'] += cam_poses
            save['image_paths'] += selected_frames_path
            save['cam_quats'] += cam_quats_pose

            print(dir, "\n", "Save Number of frames: ", num_frames, "Sliding window: ", sliding_window, "sequence length: ", len(selected_frames))
            
        
        sliding_window -= 5
        if sliding_window < 10:
            continue_flag = False
            break

print(dir, "\n", "Save Number of frames: ", num_frames, "Sliding window: ", sliding_window)
import pickle
print(len(save['selected_frames']))
with open(f"data/tum/tum_meta.pkl", "wb") as f:
    pickle.dump(save, f)
