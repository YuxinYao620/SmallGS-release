import glob
import os
import shutil
import numpy as np



from scipy.spatial.transform import Rotation

def save_to_ply(data, filename):
    with open(filename, 'w') as ply:
        # PLY header
        ply.write("ply\n")
        ply.write("format ascii 1.0\n")
        ply.write("element vertex {}\n".format(data.shape[0]))  # only non-zero values
        ply.write("comment vertices\n")
        ply.write("property float x\n")
        ply.write("property float y\n")
        ply.write("property float z\n")
        # ply.write("property float value\n")  # or use "uchar red", "uchar green", "uchar blue" for RGB colors
        # Adding properties for RGB colors

        ply.write("end_header\n")

        # PLY data
        for (x, y, z) in data:
            ply.write("{} {} {} \n".format(x, y, z))


def reprojection(points, K, RT):
    v = np.concatenate((points, np.ones((points.shape[0], 1))), axis=1)
    XYZ = (RT @ v.T).T[:, :3]
    Z = XYZ[:, 2:]
    XYZ = XYZ / XYZ[:, 2:]
    xyz = (K @ XYZ.T).T
    uv = xyz[:, :2]
    return uv, Z

def inverse_projection(depth, K, RT):
    h, w = depth.shape

    v, u = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')

    # u = w - u - 1 # for v1.0 dataset


    uv_homogeneous = np.vstack((u.flatten(), v.flatten(), np.ones_like(u.flatten())))

    K_inv = np.linalg.inv(K)

    # use max depth as 10m for visualization
    depth = depth.flatten()
    mask = depth < 10

    XYZ = K_inv @ uv_homogeneous * depth

    XYZ = np.vstack((XYZ, np.ones(XYZ.shape[1])))
    world_coordinates = np.linalg.inv(RT) @ XYZ
    world_coordinates = world_coordinates[:3, :].T
    world_coordinates = world_coordinates[mask]

    return world_coordinates

def get_consecutive_frame(frames, window_size):
    
    results = []
    # split the frames into consecutive pieces
    diff = frames[1:] - frames[0:-1]
    diff_ind = np.where(diff != 1)[0]
    start_indices = np.concatenate(([0], diff_ind[:-1])) + 1
    end_indices = diff_ind
    
    # breakpoint()
    for start, end in zip(start_indices, end_indices):
        breakpoint()
        if frames[end] - frames[start]  >= window_size:
            for i in range(start, end - window_size + 1):
                results.append(frames[start:start + window_size])
    return results

import tqdm
select_constant = True

data_path = '/scratch/yy561/pointodyssey/val/'
# dataset_name = 'cnb_dlab_0215_ego2'
exist = []

ls = os.listdir(data_path)
datasets = [l for l in ls if os.path.isdir(os.path.join(data_path, l)) and l not in exist]
# datasets = ['scene_d78_ego1','cnb_dlab_0225_ego2','cnb_dlab_0215_ego2']
# datasets = ['cnb_dlab_0225_ego2','cnb_dlab_0215_ego2']
datasets = ['cnb_dlab_0225_ego2']

selected_frames = []
selected_frames_path = []
cam_poses = []
save = {"selected_frames":[], "cam_poses":[], "image_paths":[], 'scene_name':[], 'dataset_path':[], "dataset":[]}

for dataset_name in datasets:
    annotations = np.load('{}/{}/anno.npz'.format(data_path, dataset_name))
    trajs_3d = annotations['trajs_3d'].astype(np.float32)
    cam_ints = annotations['intrinsics'].astype(np.float32)
    cam_exts = annotations['extrinsics'].astype(np.float32)

    num_frames = cam_exts.shape[0]
    # dataset_name = dir.split('/')[-2]
    sliding_window = 30
    continue_flag = True

    # load all camera poses
    # for image_file in os.listdir(os.path.join(sintel_meta['img_path'], seq_name)):
    # frames = sorted(glob.glob(os.path.join(sintel_meta['dir_path_func'](img_path, dataset_name), "*.png")))
    # frames = [f.replace("../","") for f in frames]
    frames = sorted(glob.glob(os.path.join(data_path, dataset_name, "rgbs", "*.jpg")))
    frames = [f.replace("../","") for f in frames]
    # turn quaternions into rotation matrices
    # cam_exts = []
    # for i in range(len(traj_tum)):
    #     # tx ty tz qx qy qz qw
    #     q = np.array([float(x) for x in traj_tum[i][3:]])
    #     t = np.array([float(x) for x in traj_tum[i][0:3]])
    #     cam_ext_i = np.eye(4)
    #     cam_ext_i[:3, :3] = Rotation.from_quat(q, scalar_first = False).as_matrix()
    #     cam_ext_i[:3, 3] = t
    #     cam_exts.append(cam_ext_i)
    num_frames = len(cam_exts)
    # select camera constrained videos
    while continue_flag:
        # find consecutive frames within camera poses as long as possible
        # selected_frames = []
        # selected_frames_path = []
        # cam_poses = []
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
                    q_current = Rotation.from_matrix(rotation_current).as_quat()
                    q_middle = Rotation.from_matrix(rotation_middle).as_quat()
                    # find the difference between the two quaternions
                    theta = 2*np.arccos(np.abs(np.dot(q_current,q_middle.conjugate())))
                    if theta > 0.07:
                        accept_flag = False
                    # check the translation difference, reject large baseline
                    # breakpoint()
                    t_current = cam_exts[i][:3, 3]
                    t_middle = cam_middle[:3, 3]
                    # if np.linalg.norm(t_current - t_middle) > 0.01:
                    #     accept_flag = False
                    t_current = cam_exts[i][:3, 3]
                    t_middle = cam_middle[:3, 3]
                    if np.linalg.norm(t_current - t_middle) > 0.1:
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
            save['selected_frames'] += selected_frames
            save['cam_poses'] += cam_poses
            save['dataset'] += [dataset_name for i in range(len(selected_frames))]
            save['dataset_path'] += [dir for i in range(len(selected_frames))]
            save['image_paths'] = selected_frames_path
            print(dir, "\n", "Save Number of frames: ", num_frames, "Sliding window: ", sliding_window)
            # import pickle
            # with open(f"/scratch/yy561/monst3r/data/sintel/cam_poses_{dataset_name}_{sliding_window}_theta0.1.pkl", "wb") as f:
            #     pickle.dump(save, f)
            
        
        sliding_window -= 5
        if sliding_window < 10:
            continue_flag = False
            import pickle
            # if len(selected_frames) > 0:
            #     save = {}
            #     save['selected_frames'] = selected_frames
            #     save['cam_poses'] = cam_poses
            #     save['dataset'] = dataset_name
            #     with open(f"tum_cam_poses_{dataset_name}_max.pkl", "wb") as f:
            #             pickle.dump(save, f)
            break

print(dir, "\n", "Save Number of frames: ", num_frames, "Sliding window: ", sliding_window)
import pickle
print(len(save['selected_frames']))
breakpoint()
with open(f"/scratch/yy561/monst3r/data/pointodyssey_0225.pkl", "wb") as f:
    pickle.dump(save, f)
