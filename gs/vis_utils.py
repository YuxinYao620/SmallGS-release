
import os
import matplotlib
import matplotlib.pyplot as plt
# from matplotlib import tight_layout
import copy
from evo.core.trajectory import PosePath3D, PoseTrajectory3D
from evo.main_ape import ape
# from evo.tools import plot
from evo.core import sync
from evo.tools import file_interface
from evo.core import metrics
import evo
import torch
import numpy as np
from scipy.spatial.transform import Slerp
from scipy.spatial.transform import Rotation as R
import scipy.interpolate as si
from scipy.spatial.transform import Rotation as RotLib
import math
import numpy

import gs.utils_poses.ATE.transformations as tfs
import gs.utils_poses.ATE.align_trajectory as align



def plot_pose(ref_poses, est_poses, output_path, vid=False):
    print("plot_pose")
    ref_poses = [pose for pose in ref_poses]
    if isinstance(est_poses, dict):
        est_poses = [pose for k, pose in est_poses.items()]
    else:
        est_poses = [pose for pose in est_poses]
    traj_ref = PosePath3D(poses_se3=ref_poses)
    traj_est = PosePath3D(poses_se3=est_poses)
    traj_est_aligned = copy.deepcopy(traj_est)
    traj_est_aligned.align(traj_ref, correct_scale=True,
                           correct_only_scale=False)
    if vid:
        for p_idx in range(len(ref_poses)):
            fig = plt.figure()
            current_est_aligned = traj_est_aligned.poses_se3[:p_idx+1]
            current_ref = traj_ref.poses_se3[:p_idx+1]
            current_est_aligned = PosePath3D(poses_se3=current_est_aligned)
            current_ref = PosePath3D(poses_se3=current_ref)
            traj_by_label = {
                # "estimate (not aligned)": traj_est,
                "Ours (aligned)": current_est_aligned,
                # "Ground-truth": current_ref
                "Ground Truth": current_ref
            }
            plot_mode = plot.PlotMode.xyz
            # ax = plot.prepare_axis(fig, plot_mode, 111)
            ax = fig.add_subplot(111, projection="3d")
            ax.xaxis.set_tick_params(labelbottom=False)
            ax.yaxis.set_tick_params(labelleft=False)
            ax.zaxis.set_tick_params(labelleft=False)
            colors = ['r', 'b']
            styles = ['-', '--']
            for idx, (label, traj) in enumerate(traj_by_label.items()):
                plot.traj(ax, plot_mode, traj,
                          styles[idx], colors[idx], label)
                # break
            # plot.trajectories(fig, traj_by_label, plot.PlotMode.xyz)
            ax.view_init(elev=10., azim=45)
            plt.tight_layout()
            os.makedirs(os.path.join(os.path.dirname(
                output_path), 'pose_vid'), exist_ok=True)
            pose_vis_path = os.path.join(os.path.dirname(
                output_path), 'pose_vid', 'pose_vis_{:03d}.png'.format(p_idx))
            print(pose_vis_path)
            fig.savefig(pose_vis_path)

    # else:

    fig = plt.figure()
    traj_by_label = {
        # "estimate (not aligned)": traj_est,
        "Ours (aligned)": traj_est_aligned,
        # "Ground-truth": traj_ref
        "Droid": traj_ref
    }
    plot_mode = plot.PlotMode.xyz
    # ax = plot.prepare_axis(fig, plot_mode, 111)
    ax = fig.add_subplot(111, projection="3d")
    ax.xaxis.set_tick_params(labelbottom=True)
    ax.yaxis.set_tick_params(labelleft=True)
    ax.zaxis.set_tick_params(labelleft=True)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    colors = ['r', 'b']
    styles = ['-', '--']

    for idx, (label, traj) in enumerate(traj_by_label.items()):
        plot.traj(ax, plot_mode, traj,
                  styles[idx], colors[idx], label)
        # break
    # plot.trajectories(fig, traj_by_label, plot.PlotMode.xyz)
    ax.view_init(elev=10., azim=45)
    plt.tight_layout()
    pose_vis_path = os.path.join(os.path.dirname(output_path), 'pose_vis.png')
    fig.savefig(pose_vis_path)



