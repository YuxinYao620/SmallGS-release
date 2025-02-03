import glob
import os
import shutil
import numpy as np
from scipy.spatial.transform import Rotation
sintel_meta ={
        'img_path': "/scratch/yy561/monst3r/data/sintel/training/final",
        'anno_path': "/scratch/yy561/monst3r/data/sintel/training/camdata_left",
        'mask_path': None,
        'dir_path_func': lambda img_path, seq: os.path.join(img_path, seq),
        'gt_traj_func': lambda img_path, anno_path, seq: os.path.join(anno_path, seq),
        'traj_format': None,
        'seq_list': ["alley_2", "ambush_4", "ambush_5", "ambush_6", "cave_2", "cave_4", "market_2",
                     "market_5", "market_6", "shaman_3", "sleeping_1", "sleeping_2", "temple_2", "temple_3"],
        'full_seq': False,
        'mask_path_seq_func': lambda mask_path, seq: None,
        'skip_condition': None,
}


def sintel_cam_read(filename):
    """Read camera data, return (M,N) tuple.

    M is the intrinsic matrix, N is the extrinsic matrix, so that

    x = M*N*X,
    with x being a point in homogeneous image pixel coordinates, X being a
    point in homogeneous world coordinates.
    """
    TAG_FLOAT = 202021.25

    f = open(filename, "rb")
    check = np.fromfile(f, dtype=np.float32, count=1)[0]
    assert (
        check == TAG_FLOAT
    ), " cam_read:: Wrong tag in flow file (should be: {0}, is: {1}). Big-endian machine? ".format(
        TAG_FLOAT, check
    )
    M = np.fromfile(f, dtype="float64", count=9).reshape((3, 3))
    N = np.fromfile(f, dtype="float64", count=12).reshape((3, 4))
    return M, N

def load_sintel_traj(gt_file): # './data/sintel/training/camdata_left/alley_2'
    # Refer to ParticleSfM
    gt_pose_lists = sorted(os.listdir(gt_file))
    gt_pose_lists = [os.path.join(gt_file, x) for x in gt_pose_lists if x.endswith(".cam")]
    tstamps = [float(x.split("/")[-1][:-4].split("_")[-1]) for x in gt_pose_lists]
    gt_poses = [sintel_cam_read(f)[1] for f in gt_pose_lists] # [1] means get the extrinsic
    breakpoint()
    xyzs, wxyzs = [], []
    tum_gt_poses = []
    for gt_pose in gt_poses:
        gt_pose = np.concatenate([gt_pose, np.array([[0, 0, 0, 1]])], 0)
        gt_pose_inv = np.linalg.inv(gt_pose)  # world2cam -> cam2world
        xyz = gt_pose_inv[:3, -1]
        xyzs.append(xyz)
        R = Rotation.from_matrix(gt_pose_inv[:3, :3])
        xyzw = R.as_quat()  # scalar-last for scipy
        wxyz = np.array([xyzw[-1], xyzw[0], xyzw[1], xyzw[2]])
        wxyzs.append(wxyz)
        tum_gt_pose = np.concatenate([xyz, wxyz], 0) #TODO: check if this is correct
        tum_gt_poses.append(tum_gt_pose)

    tum_gt_poses = np.stack(tum_gt_poses, 0)
    tum_gt_poses[:, :3] = tum_gt_poses[:, :3] - np.mean(
        tum_gt_poses[:, :3], 0, keepdims=True
    )
    tt = np.expand_dims(np.stack(tstamps, 0), -1)
    return tum_gt_poses, tt

def load_sintel_traj_se3(gt_file):
    gt_pose_lists = sorted(os.listdir(gt_file))
    gt_pose_lists = [os.path.join(gt_file, x) for x in gt_pose_lists if x.endswith(".cam")]
    tstamps = [float(x.split("/")[-1][:-4].split("_")[-1]) for x in gt_pose_lists]
    gt_poses = [sintel_cam_read(f)[1] for f in gt_pose_lists] # [1] means get the extrinsic
    xyzs, wxyzs = [], []
    tum_gt_poses = []
    for gt_pose in gt_poses:
        gt_pose = np.concatenate([gt_pose, np.array([[0, 0, 0, 1]])], 0)
        tum_gt_poses.append(gt_pose)

    tum_gt_poses = np.stack(tum_gt_poses, 0)
    tum_gt_poses[:, 3, :3] = tum_gt_poses[:, 3, :3] - np.mean(
        tum_gt_poses[:, 3, :3], 0, keepdims=True
    )
    tt = np.expand_dims(np.stack(tstamps, 0), -1)
    return tum_gt_poses, tt
# dirs = glob.glob("../data/tum/*/")
# dirs = sorted(dirs)

# root_dir = "/scratch/yy561/monst3r/"

img_path = sintel_meta['img_path']
mask_path = sintel_meta['mask_path']

# extract frames
# for dir in dirs:
for dataset_name in sintel_meta['seq_list']:

    # dataset_name = dir.split('/')[-2]
    sliding_window = 50
    continue_flag = True

    # load all camera poses
    traj_path = os.path.join(sintel_meta['gt_traj_func'](img_path, sintel_meta['anno_path'], dataset_name))
    cam_exts, timestamps_mat = load_sintel_traj_se3(traj_path)
    # for image_file in os.listdir(sintel_meta['dir_path_func'](img_path, seq_name)):
    # for image_file in os.listdir(os.path.join(sintel_meta['img_path'], seq_name)):
    frames = sorted(glob.glob(os.path.join(sintel_meta['dir_path_func'](img_path, dataset_name), "*.png")))
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
                    q_current = Rotation.from_matrix(rotation_current).as_quat()
                    q_middle = Rotation.from_matrix(rotation_middle).as_quat()
                    # find the difference between the two quaternions
                    theta = 2*np.arccos(np.abs(np.dot(q_current,q_middle.conjugate())))
                    if theta > 0.1:
                        accept_flag = False
                    # check the translation difference, reject large baseline
                    # breakpoint()
                    t_current = cam_exts[i][:3, 3]
                    t_middle = cam_middle[:3, 3]
                    # if np.linalg.norm(t_current - t_middle) > 0.01:
                    #     accept_flag = False
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
            save = {}
            save['selected_frames'] = selected_frames
            save['cam_poses'] = cam_poses
            save['dataset'] = dataset_name
            save['dataset_path'] = dir
            save['image_paths'] = selected_frames_path
            print(dir, "\n", "Save Number of frames: ", num_frames, "Sliding window: ", sliding_window)
            import pickle
            with open(f"/scratch/yy561/monst3r/data/sintel/cam_poses_{dataset_name}_{sliding_window}_theta0.1.pkl", "wb") as f:
                pickle.dump(save, f)
            
        
        sliding_window -= 5
        if sliding_window < 15:
            continue_flag = False
            import pickle
            if len(selected_frames) > 0:
                save = {}
                save['selected_frames'] = selected_frames
                save['cam_poses'] = cam_poses
                save['dataset'] = dataset_name
                with open(f"tum_cam_poses_{dataset_name}_max.pkl", "wb") as f:
                        pickle.dump(save, f)
            break
    