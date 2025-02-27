# --------------------------------------------------------
# gradio demo
# --------------------------------------------------------

import argparse
import math
import gradio
import os
import torch
import numpy as np
import tempfile
import functools
import copy

from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.image_pairs import make_pairs
from dust3r.utils.image import load_images, rgb, enlarge_seg_masks
from dust3r.utils.device import to_numpy
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from dust3r.utils.viz_demo import convert_scene_output_to_glb, get_dynamic_mask_from_pairviewer
import matplotlib.pyplot as pl
import time
from gs.gs_diff_copy import Runner
# from gs.gs_diff import Runner
from pathlib import Path
from evo.core.trajectory import PoseTrajectory3D,PosePath3D
import copy
import gc

import pickle

from scipy.spatial.transform import Rotation
pl.ion()

torch.backends.cuda.matmul.allow_tf32 = True  # for gpu >= Ampere and pytorch >= 1.12
batch_size = 1


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser_url = parser.add_mutually_exclusive_group()
    parser_url.add_argument("--local_network", action='store_true', default=False,
                            help="make app accessible on local network: address will be set to 0.0.0.0")
    parser_url.add_argument("--server_name", type=str, default=None, help="server url, default is 127.0.0.1")
    parser.add_argument("--image_size", type=int, default=512, choices=[512, 224], help="image size")
    parser.add_argument("--server_port", type=int, help=("will start gradio app on this port (if available). "
                                                         "If None, will search for an available port starting at 7860."),
                        default=None)
    parser.add_argument("--weights", type=str, help="path to the model weights", default='checkpoints/MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt.pth')
    parser.add_argument("--model_name", type=str, default='Junyi42/MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt', help="model name")
    parser.add_argument("--device", type=str, default='cuda', help="pytorch device")
    parser.add_argument("--output_dir", type=str, default='./demo_tmp', help="value for tempfile.tempdir")
    parser.add_argument("--silent", action='store_true', default=False,
                        help="silence logs")
    parser.add_argument("--input_dir", type=str, help="Path to input images directory", default=None)
    parser.add_argument("--seq_name", type=str, help="Sequence name for evaluation", default='NULL')
    parser.add_argument('--use_gt_davis_masks', action='store_true', default=False, help='Use ground truth masks for DAVIS')
    parser.add_argument('--fps', type=int, default=0, help='FPS for video processing')
    parser.add_argument('--num_frames', type=int, default=200, help='Maximum number of frames for video processing')

    # Gaussian splatting args
    parser.add_argument('--ckpt', type=str, default=None, help="Path to the .pt file. If provided, it will skip training and render a video")
    parser.add_argument('--pose_ckpt', type=str, default=None, help="Path to the pose checkpoint file")

    parser.add_argument('--data_dir', type=str, default="data/360_v2/garden", help="Path to the Mip-NeRF 360 dataset")
    parser.add_argument('--data_factor', type=int, default=1, help="Downsample factor for the dataset")
    parser.add_argument('--result_dir', type=str, default="results/garden", help="Directory to save results")
    
    parser.add_argument('--global_scale', type=float, default=1.0, help="A global scaler that applies to the scene size related parameters")
    parser.add_argument('--batch_size', type=int, default=15, help="Batch size for training. Learning rates are scaled automatically")
    parser.add_argument('--steps_scaler', type=float, default=1.0, help="A global factor to scale the number of training steps")
    parser.add_argument('--patch_size', type=int, default=None, help="Patch size for training")

    parser.add_argument('--max_steps', type=int, default=500, help="Number of training steps")
    parser.add_argument('--eval_steps', type=int, nargs='*', default=[7000, 30000], help="Steps to evaluate the model")
    parser.add_argument('--save_steps', type=int, nargs='*', default=[7000, 30000], help="Steps to save the model")
    parser.add_argument('--test_every', type=int, default=15, help="Reset the optimizer every this steps")

    parser.add_argument('--init_type', type=str, default="predict", help="Initialization strategy")
    parser.add_argument('--init_num_pts', type=int, default=100000, help="Initial number of GSs. Ignored if using sfm")
    parser.add_argument('--init_extent', type=float, default=3.0, help="Initial extent of GSs as a multiple of the camera extent. Ignored if using sfm")
    parser.add_argument('--sh_degree', type=int, default=3, help="Degree of spherical harmonics")
    parser.add_argument('--sh_degree_interval', type=int, default=1000, help="Turn on another SH degree every this steps")
    parser.add_argument('--refine_start_iter', type=int, default=500, help="Start refining GSs after this iteration")
    parser.add_argument('--refine_stop_iter', type=int, default=15000, help="Stop refining GSs after this iteration")
    parser.add_argument('--reset_every', type=int, default=3000, help="Reset opacities every this steps")
    parser.add_argument('--refine_every', type=int, default=100, help="Refine GSs every this steps")

    parser.add_argument('--init_opa', type=float, default=0.1, help="Initial opacity of GS")
    parser.add_argument('--init_scale', type=float, default=1.0, help="Initial scale of GS")
    parser.add_argument('--ssim_lambda', type=float, default=0.0, help="Weight for SSIM loss")
    parser.add_argument('--near_plane', type=float, default=0.01, help="Near plane clipping distance")
    parser.add_argument('--far_plane', type=float, default=1e10, help="Far plane clipping distance")
    parser.add_argument('--prune_opa', type=float, default=0.005, help="GSs with opacity below this value will be pruned")
    parser.add_argument('--grow_grad2d', type=float, default=0.0002, help="GSs with image plane gradient above this value will be split/duplicated")
    parser.add_argument('--grow_scale3d', type=float, default=0.01, help="GSs with scale below this value will be duplicated. Above will be split")
    parser.add_argument('--prune_scale3d', type=float, default=0.1, help="GSs with scale above this value will be pruned")

    parser.add_argument('--packed', action='store_true', help="Use packed mode for rasterization, this leads to less memory usage but slightly slower.")
    parser.add_argument('--sparse_grad', action='store_true', help="Use sparse gradients for optimization. (experimental)")
    parser.add_argument('--absgrad', action='store_true', help="Use absolute gradient for pruning. This typically requires larger --grow_grad2d, e.g., 0.0008 or 0.0006")
    parser.add_argument('--antialiased', action='store_true', help="Anti-aliasing in rasterization. Might slightly hurt quantitative metrics.")
    parser.add_argument('--revised_opacity', action='store_true', help="Whether to use revised opacity heuristic from arXiv:2404.06109 (experimental)")
    parser.add_argument('--random_bkgd', action='store_true', help="Use random background for training to discourage transparency")
    parser.add_argument('--pose_opt', action='store_true', help="Enable camera optimization.")
    parser.add_argument('--pose_opt_lr', type=float, default=1e-2, help="Learning rate for camera optimization")
    parser.add_argument('--pose_opt_reg', type=float, default=1e-5, help="Regularization for camera optimization as weight decay")
    parser.add_argument('--pose_noise', type=float, default=0.0, help="Add noise to camera extrinsics. This is only to test the camera pose optimization.")
    parser.add_argument('--cf3dgs_position_lr_init', type=float, default=0.00016, help="Initial learning rate for cf3dgs position")
    parser.add_argument('--cf3dgs_position_lr_final', type=float, default=0.0000016, help="Final learning rate for cf3dgs position")
    parser.add_argument('--cf3dgs_position_lr_delay_mult', type=float, default=0.01, help="Delay multiplier for cf3dgs position learning rate")
    parser.add_argument('--cf3dgs_position_lr_max_steps', type=int, default=30000, help="Max steps for cf3dgs position learning rate")
    parser.add_argument('--app_opt', action='store_true', help="Enable appearance optimization. (experimental)")
    parser.add_argument('--app_embed_dim', type=int, default=16, help="Appearance embedding dimension")
    parser.add_argument('--app_opt_lr', type=float, default=1e-3, help="Learning rate for appearance optimization")
    parser.add_argument('--app_opt_reg', type=float, default=1e-6, help="Regularization for appearance optimization as weight decay")
    parser.add_argument('--depth_loss', action='store_true', default="True", help="Enable depth loss. (experimental)")
    parser.add_argument('--depth_lambda', type=float, default=1e-2, help="Weight for depth loss")

    parser.add_argument('--rotation_lr', type=float, default=1e-3, help="Learning rate for rotation")
    parser.add_argument('--datatype', type=str, default="monst3r", help="Datatype")
    parser.add_argument('--doma', action='store_true', help="Enable doma")
    parser.add_argument('--motion_model_name', type=str, default="transfield4d", help="Motion model name")
    parser.add_argument('--motion_color_name', type=str, default="transfield4d", help="Motion color name")
    parser.add_argument('--elastic_loss_weight', type=float, default=0.0, help="Elastic loss weight")
    parser.add_argument('--homo_loss_weight', type=float, default=0.1, help="Homo loss weight")
    parser.add_argument('--semantics_opt', default=True, action='store_true', help="Enable semantics optimization")
    parser.add_argument('--save_sem_mask', action='store_true', help="Save semantic mask")
    parser.add_argument('--eval_overall_gs', action='store_true', help="Evaluate overall GS")
    parser.add_argument('--eval_pose', action='store_true', help="Evaluate pose")
    parser.add_argument('--doma_max_step', type=int, default= 1000, help="Max iteration for doma and guidance")
    parser.add_argument('--doma_max_epoch', type=int, default= 100 , help="Max epochs for doma and guidance")
    parser.add_argument('--initial_train_step', type= int, default= 5000, help="Max iteration for intial doma and gs training")
    parser.add_argument('--only_increase_gs', action='store_true')
    parser.add_argument('--use_all_human_depth', action='store_true')

    #splat fields args
    # parser.add_argument('--radius', type=float, default=1.1)
    parser.add_argument('--position_lr_init', type=float, default=1e-4, help="Initial learning rate for cf3dgs position")
    parser.add_argument('--position_lr_final', type=float, default=1e-6, help="Final learning rate for cf3dgs position")
    parser.add_argument('--position_lr_delay_mult', type=float, default=0.01, help="Delay multiplier for cf3dgs position learning rate")
    parser.add_argument('--position_lr_max_steps', type=int, default=5000, help="Max steps for cf3dgs position learning rate")
    parser.add_argument('--deform_lr_max_steps', type=int, default=5000, help="Max steps for cf3dgs deformation learning rate")

    parser.add_argument('--canny_opt', action='store_true', help="Enable canny optimization")
    parser.add_argument('--camera_smoothness_loss', action='store_true', default=True,help="Enable camera smoothness loss")
    parser.add_argument('--camera_smoothness_lambda', type=float, default=0.5, help="Weight for camera smoothness loss")

    parser.add_argument('--local_rank',type=int, default=0)
    parser.add_argument('--world_rank',type=int, default=0)
    parser.add_argument('--world_size',type=int, default=1)

    parser.add_argument('--test_gs', action='store_true', help="Test GS")
    parser.add_argument('--fix_strategy', action='store_true', help="Fix gaussian number ")
    parser.add_argument('--use_spec_points', action = 'store_true', help="Add SMPL vertices to the intialization for gaussian splatting")
    parser.add_argument('--background_color', type=float, nargs='*', default=None, help="Background color for the renderer")
    parser.add_argument('--draw_frame', action='store_true', help="Draw frame") 
    
    # Add "share" argument if you want to make the demo accessible on the public internet
    parser.add_argument("--share", action='store_true', default=False, help="Share the demo")

    # gs options
    parser.add_argument("--gs_camera", type=str, help="camera pose estimated by gaussian splatting")
    parser.add_argument("--gs_refine", action='store_true', default=False, help="whether use gaussian splatting to refine camera poses")
    parser.add_argument("--gs_pose", action='store_true', help="purely use gs for pose estimation")
    parser.add_argument("--monst3r_camera", action='store_true', help="use monst3r to estimate camera poses")
    parser.add_argument("--use_monst3r_intermediate", action='store_true', help="monst3r intermediate pointmaps")
    #eval options
    parser.add_argument("--eval_only", action='store_true', default=False, help="only evaluate the model")

    parser.add_argument("--dino", action='store_true', default=False, help="use dino model")
    parser.add_argument("--ft_loss_lambda", type=float, default=1, help="feature loss lambda")
    parser.add_argument("--dino_dim", type=int, default=6, help="use feature loss")
    
    parser.add_argument("--doma_eval", action='store_true', default=False, help="evaluate doma")
    return parser

def get_3D_model_from_scene(outdir, silent, scene, min_conf_thr=3, as_pointcloud=False, mask_sky=False,
                            clean_depth=False, transparent_cams=False, cam_size=0.05, show_cam=True, save_name=None, thr_for_init_conf=True):
    """
    extract 3D_model (glb file) from a reconstructed scene
    """
    if scene is None:
        return None
    # post processes
    if clean_depth:
        scene = scene.clean_pointcloud()
    if mask_sky:
        scene = scene.mask_sky()

    # get optimized values from scene
    rgbimg = scene.imgs
    focals = scene.get_focals().cpu()
    cams2world = scene.get_im_poses().cpu()
    # 3D pointcloud from depthmap, poses and intrinsics
    pts3d = to_numpy(scene.get_pts3d(raw_pts=True))
    scene.min_conf_thr = min_conf_thr
    scene.thr_for_init_conf = thr_for_init_conf
    msk = to_numpy(scene.get_masks())
    cmap = pl.get_cmap('viridis')
    cam_color = [cmap(i/len(rgbimg))[:3] for i in range(len(rgbimg))]
    cam_color = [(255*c[0], 255*c[1], 255*c[2]) for c in cam_color]
    return convert_scene_output_to_glb(outdir, rgbimg, pts3d, msk, focals, cams2world, as_pointcloud=as_pointcloud,
                                        transparent_cams=transparent_cams, cam_size=cam_size, show_cam=show_cam, silent=silent, save_name=save_name,
                                        cam_color=cam_color)

def intermediate_pcds(pairs, output, imgs, args):
    """
    get intermediate point clouds from the output
    """
    start_frame = 1
    end_frame = len(imgs)
    batch_ind = []
    data_block = []
    for current_frame in range(start_frame, end_frame, args.batch_size):
        block_end = min(current_frame + args.batch_size, end_frame)
        for i in range(current_frame-1, block_end):
            data_block.append(i)
        batch_ind.append([data_block[0], data_block[-1]])
        data_block = []
    gs_use_pts = {}
    gs_use_colors = {}
    batch_ind = np.array(batch_ind)
    # extract all pairs of ind
    pair_ind = []
    for i, (view1, view2) in enumerate(pairs):
        pair_ind.append([i, view1['idx'], view2['idx']])
    
    for ind in batch_ind[:,0]:
        max_ind = ind
        mark_i = 0
        for i, ind1,ind2 in pair_ind:
            if ind1 == ind:
                if abs(ind2-ind) > abs(max_ind - ind):
                    max_ind = ind2
                    mark_i = i
            if ind2 == ind:
                if abs(ind1-ind) > abs(max_ind - ind):
                    max_ind = ind1
                    mark_i = i
        pred1 = output['pred1']['pts3d'][mark_i]
        gs_use_pts[ind] = pred1
        gs_use_colors[ind] = output['view1']['img'][mark_i]
    return gs_use_pts, gs_use_colors
def get_reconstructed_scene(args, outdir, model, device, silent, image_size, filelist, schedule, niter, min_conf_thr,
                            as_pointcloud, mask_sky, clean_depth, transparent_cams, cam_size, show_cam, scenegraph_type, winsize, refid, 
                            seq_name, new_model_weights, temporal_smoothing_weight, translation_weight, shared_focal, 
                            flow_loss_weight, flow_loss_start_iter, flow_loss_threshold, use_gt_mask, fps, num_frames):
    """
    from a list of images, run dust3r inference, global aligner.
    then run get_3D_model_from_scene
    """
    monst3r_time = 0
    gs_refine_time = 0
    start_time = time.time()
    # if True:
    translation_weight = float(translation_weight)
    if new_model_weights != args.weights:
        model = AsymmetricCroCo3DStereo.from_pretrained(new_model_weights).to(device)
    model.eval()
    if seq_name != "NULL":
        dynamic_mask_path = f'data/davis/DAVIS/masked_images/480p/{seq_name}'
    else:
        dynamic_mask_path = None
    imgs = load_images(filelist, size=image_size, verbose=not silent, dynamic_mask_root=dynamic_mask_path, fps=fps, num_frames=num_frames)
    if len(imgs) == 1:
        imgs = [imgs[0], copy.deepcopy(imgs[0])]
        imgs[1]['idx'] = 1
    if scenegraph_type == "swin" or scenegraph_type == "swinstride" or scenegraph_type == "swin2stride":
        scenegraph_type = scenegraph_type + "-" + str(winsize) + "-noncyclic"
    elif scenegraph_type == "oneref":
        scenegraph_type = scenegraph_type + "-" + str(refid)
    if not os.path.exists(os.path.join(outdir, seq_name, "scene.glb")):
        pairs = make_pairs(imgs, scene_graph=scenegraph_type, prefilter=None, symmetrize=True)
        output = inference(pairs, model, device, batch_size=batch_size, verbose=not silent)
        gs_use_pts, gs_use_colors = intermediate_pcds(pairs, output, imgs,args)
        save_folder = f'{args.output_dir}/{seq_name}'  #default is 'demo_tmp/NULL'
        os.makedirs(save_folder, exist_ok=True)
        # with open(f'{outdir}/{seq_name}/gs_use_pts.pkl', 'wb') as f:
        #     # pickle.dump(gs_use_pts, f)
        #     save= {}
        #     save['gs_use_pts'] = gs_use_pts
        #     save['gs_use_colors'] = gs_use_colors
        #     pickle.dump(save, f)
            
        # if args.gs_refine and os.path.exists(os.path.join(args.output_dir, seq_name, "scene.glb")):
        #     pass
        # else:
        if len(imgs) > 2:
            mode = GlobalAlignerMode.PointCloudOptimizer  
            scene = global_aligner(output, device=device, mode=mode, verbose=not silent, shared_focal = shared_focal, temporal_smoothing_weight=temporal_smoothing_weight, translation_weight=translation_weight,
                                flow_loss_weight=flow_loss_weight, flow_loss_start_epoch=flow_loss_start_iter, flow_loss_thre=flow_loss_threshold, use_self_mask=not use_gt_mask,
                                num_total_iter=niter, empty_cache= len(filelist) > 72)
        else:
            mode = GlobalAlignerMode.PairViewer
            scene = global_aligner(output, device=device, mode=mode, verbose=not silent)

        lr = 0.01

        if mode == GlobalAlignerMode.PointCloudOptimizer:
            loss = scene.compute_global_alignment(init='mst', niter=niter, schedule=schedule, lr=lr) # estimate camera pose/intrinsic etc here
        
        save_folder = f'{args.output_dir}/{seq_name}'  #default is 'demo_tmp/NULL'
        os.makedirs(save_folder, exist_ok=True)

        outfile = get_3D_model_from_scene(save_folder, silent, scene, min_conf_thr, as_pointcloud, mask_sky,
                                clean_depth, transparent_cams, cam_size, show_cam)

        poses = scene.save_tum_poses(f'{save_folder}/pred_traj.txt')
        K = scene.save_intrinsics(f'{save_folder}/pred_intrinsics.txt')

        depth_path_save = f'{save_folder}/depths'
        sem_path_save = f'{save_folder}/semantics'
        if not os.path.exists(f'{save_folder}/semantics'):
            os.makedirs(depth_path_save, exist_ok=True)
            os.makedirs(sem_path_save, exist_ok=True)

        depth_maps = scene.save_depth_maps(depth_path_save)
        dynamic_masks = scene.save_dynamic_masks(sem_path_save)
        # conf = scene.save_conf_maps(save_folder)
        # init_conf = scene.save_init_conf_maps(save_folder)
        rgbs = scene.save_rgb_imgs(save_folder)
        # pts3d = scene.get_pts3d()

        enlarge_seg_masks(save_folder, kernel_size=5 if use_gt_mask else 3) 

        # also return rgb, depth and confidence imgs
        # depth is normalized with the max value for all images
        # we apply the jet colormap on the confidence maps
        rgbimg = scene.imgs
        depths = to_numpy(scene.get_depthmaps())
        confs = to_numpy([c for c in scene.im_conf])
        init_confs = to_numpy([c for c in scene.init_conf_maps])
        cmap = pl.get_cmap('jet')
        depths_max = max([d.max() for d in depths])
        depths = [cmap(d/depths_max) for d in depths]
        confs_max = max([d.max() for d in confs])
        confs = [cmap(d/confs_max) for d in confs]
        init_confs_max = max([d.max() for d in init_confs])
        init_confs = [cmap(d/init_confs_max) for d in init_confs]

        imgs = []
        for i in range(len(rgbimg)):
            imgs.append(rgbimg[i])
            imgs.append(rgb(depths[i]))
            imgs.append(rgb(confs[i]))
            imgs.append(rgb(init_confs[i]))

        # if two images, and the shape is same, we can compute the dynamic mask
        if len(rgbimg) == 2 and rgbimg[0].shape == rgbimg[1].shape:
            motion_mask_thre = 0.35
            error_map = get_dynamic_mask_from_pairviewer(scene, both_directions=True, output_dir=args.output_dir, motion_mask_thre=motion_mask_thre)
            # imgs.append(rgb(error_map))
            # apply threshold on the error map
            normalized_error_map = (error_map - error_map.min()) / (error_map.max() - error_map.min())
            error_map_max = normalized_error_map.max()
            error_map = cmap(normalized_error_map/error_map_max)
            imgs.append(rgb(error_map))
            binary_error_map = (normalized_error_map > motion_mask_thre).astype(np.uint8)
            imgs.append(rgb(binary_error_map*255))
        end_time = time.time()
        monst3r_time = end_time - start_time
        with Path(f"{outdir}/{seq_name}/time.txt").open("a") as f:
            f.write(f"MonST3R time: {monst3r_time:.2f}\n")
        # delete the results to clear the memory
        del scene, rgbimg, depths, confs, init_confs, depths_max, confs_max, init_confs_max, imgs
        gc.collect()

    if args.gs_pose:
        gs_pose_start_time = time.time()
        # del everything in monst3r
        if args.use_monst3r_intermediate:
            if os.path.exists(f'{outdir}/{seq_name}/gs_use_pts.pkl'):
                with open(f'{outdir}/{seq_name}/gs_use_pts.pkl', 'rb') as f:
                    gts_pkl= pickle.load(f)
                    gs_use_pts = gts_pkl['gs_use_pts']
                    gs_use_colors = gts_pkl['gs_use_colors']
            else:
                imgs = load_images(filelist, size=image_size, verbose=not silent, dynamic_mask_root=dynamic_mask_path, fps=fps, num_frames=num_frames)
                pairs = make_pairs(imgs, scene_graph=scenegraph_type, prefilter=None, symmetrize=True)
                output = inference(pairs, model, device, batch_size=batch_size, verbose=not silent)
                gs_use_pts, gs_use_colors = intermediate_pcds(pairs, output, imgs,args)
                # save
                save_gts = {}
                save_gts['gs_use_pts'] = gs_use_pts
                save_gts['gs_use_colors'] = gs_use_colors
                # with open(f'{outdir}/{seq_name}/gs_use_pts.pkl', 'wb') as f:
                #     pickle.dump(save_gts, f)

                del imgs, pairs, output
                gc.collect()
        
        print('using gaussian splatting to estimate camera poses, not refine!')
        args.workspace = f'{args.output_dir}/{seq_name}' 
        args.data_dir = args.input_dir
        args.monst3r_dir = f'{args.output_dir}/{seq_name}' 
        args.camera_smoothness_lambda = 1
        # args.datatype = 'custom'
        args.datatype = 'monst3r' if "own" not in args.output_dir else 'custom'
        if args.test_gs:
            args.datatype = 'custom'
        runner = Runner(local_rank=args.local_rank, world_rank=args.world_rank, world_size=args.world_size, opt=args)
        
        if args.use_monst3r_intermediate:
            print('using monst3r intermediate pointmaps')
            gs_pose = runner.train(monst3r_pointmap = gs_use_pts, monst3r_pointcolor = gs_use_colors)['global_cam_poses_est']
        else:
            # if args.test_gs:
            #     pose_ckpt = "/scratch/yy561/monst3r/paper_example/tum_rgb_refine_pose_copy/rgbd_dataset_freiburg3_walking_static_seq_390_419/traj/gs_camera_poses.pt"
            #     pose_ckpt = torch.load(pose_ckpt, map_location=args.device)
            #     global_cam_poses_est = pose_ckpt['global_cam_poses_est']
            #     runner.trainset.set_camera_to_world(global_cam_poses_est)
            #     runner.train_gs_doma_sem(args.sh_degree,initial_train_step = 12000) # inited splat, doma doma optimizer, doma scheduler
            #     # runner.train_gs_doma(args.sh_degree,global_cam_poses_est)
            #     return None, None, None
            if args.test_gs:
                pose_ckpt = "/scratch/yy561/monst3r/paper_example/tum_rgb_refine_pose_copy/rgbd_dataset_freiburg3_walking_static_seq_390_419/traj/gs_camera_poses.pt"
                pose_ckpt = torch.load(pose_ckpt, map_location=args.device)
                global_cam_poses_est = pose_ckpt['global_cam_poses_est']
                runner.trainset.set_camera_to_world(global_cam_poses_est)
                runner.train_gs_doma(args.sh_degree,initial_train_step = 1000) # inited splat, doma doma optimizer, doma scheduler
                return None, None, None
            elif args.draw_frame:
                runner.draw_frame(0,device=runner.device,opt=args)
                return None, None, None
            else:
                gs_pose = runner.train()['global_cam_poses_est']
        # transform back
        gs_pose = torch.cat([torch.Tensor(pose).unsqueeze(0) for pose in gs_pose], dim=0)
        gs_pose_R = gs_pose[:,:3, :3]
        gs_pose_R = gs_pose_R.cpu().numpy()
        quat = Rotation.from_matrix(gs_pose_R)
        quat = quat.as_quat()
        quat = torch.Tensor(quat)
        # translation
        t = gs_pose[:,:3, 3]
        # set the camera pose
        camera_pose = torch.cat([quat,t], dim=1).to(device)
        tostr = lambda a: " ".join(map(str, a))

        # save the refined camera poses
        with Path(f"{outdir}/{seq_name}/gs_pose.txt").open("w") as f:
            for i in range(len(camera_pose)):
                ith_pose = camera_pose[i]
                f.write(
                    f"{i} {tostr(ith_pose[-3::].detach().cpu().numpy())} {tostr(ith_pose[[0,1,2,3]].detach().cpu().numpy())}\n"
                )

        # gs_camera_end_time = time.time()
        # gs_camera_time = gs_camera_end_time - gs_camera_end_time
        gs_pose_end_time = time.time()
        gs_pose_time = gs_pose_end_time - gs_pose_start_time
        # write the times to a file
        with Path(f"{outdir}/{seq_name}/time.txt").open("a") as f:
            f.write(f"GS estimation time: {gs_pose_time:.2f}\n")
            for k,v in vars(args).items():
                f.write(f"{k}: {v}\n")

    if args.gs_refine:
        # monst3r_traj = scene.save_tum_poses(f'{outdir}/{seq_name}/pred_traj_before_refine.txt')
        # load the camera poses estimated by monst3r
        gs_refine_start_time = time.time()
        with Path(f"{outdir}/{seq_name}/pred_traj.txt").open("r") as f:
            monster_pose = f.readlines()
            monster_pose = np.array([list(map(float,pose.split())) for pose in monster_pose])
        monst3r_traj = torch.Tensor([pose[1:] for pose in monster_pose])

        print('using gaussian splatting to refine camera poses')
        args.workspace = f'{args.output_dir}/{seq_name}' 
        args.data_dir = args.input_dir
        args.monst3r_dir = f'{args.output_dir}/{seq_name}' 

        from gs.datasets.colmap import quaternion_to_matrix
        monst3r_traj = torch.Tensor(monst3r_traj)
        rot_mat = quaternion_to_matrix(monst3r_traj[:,3:7])
        t_mat = monst3r_traj[:,0:3]

        # compose to camera pose
        # turn to se3
        monst3r_camera_pose = torch.zeros((monst3r_traj.shape[0],4,4))
        monst3r_camera_pose[:,:3,:3] = rot_mat
        monst3r_camera_pose[:,:3,3] = t_mat
        monst3r_camera_pose[:,3,3] = 1
            
        # monst3r_camera_pose = runner.trainset.parser.camtoworlds 
        # transform the first frame camera to 0 rotation and translation 
        first_frame_pose = monst3r_camera_pose[0]
        first_frame_R = first_frame_pose[:3,:3]
        first_frame_t = first_frame_pose[:3,3]
        identity_pose = torch.eye(4,4)
        # identity_pose[3,3] = 1

        #relative transformaiton monst3r->indentity 
        R = torch.inverse(first_frame_R)
        t = -torch.matmul(R, first_frame_t)
        T = torch.cat([torch.cat([R, t.unsqueeze(1)], dim=1), torch.Tensor([[0,0,0,1]])], dim=0)
        # sanity check 
        assert torch.allclose(torch.matmul(T, first_frame_pose), identity_pose, atol=1e-6)

        #transform all the camera poses to the new coordinate system
        monst3r_camera_pose = monst3r_camera_pose @ T

        # turn to trajectory
        from evo.core.trajectory import PosePath3D
        traj_monst3r = PosePath3D(poses_se3=monst3r_camera_pose)
        tostr = lambda a: " ".join(map(str, a))

        # save the camera poses
        with Path(f"{outdir}/{seq_name}/pred_traj_monst3r_for_refine.txt").open("w") as f:
            for i in range(traj_monst3r.num_poses):
                f.write(
                    f"{i} {tostr(traj_monst3r.positions_xyz[i])} {tostr(traj_monst3r.orientations_quat_wxyz[i][[0,1,2,3]])}\n"
                )
    

        runner = Runner(local_rank=args.local_rank, world_rank=args.world_rank, world_size=args.world_size, opt=args)


        if args.use_monst3r_intermediate:
            print('use monst3r intermediate pointmaps')
            if "gs_use_pts" not in locals():
                if os.path.exists(f'{outdir}/{seq_name}/gs_use_pts.pkl'):
                    with open(f'{outdir}/{seq_name}/gs_use_pts.pkl', 'rb') as f:
                        gts_pkl= pickle.load(f)
                    gs_use_pts = gts_pkl['gs_use_pts'] 
                    gs_use_colors = gts_pkl['gs_use_colors']
                    
                else:
                    imgs = load_images(filelist, size=image_size, verbose=not silent, dynamic_mask_root=dynamic_mask_path, fps=fps, num_frames=num_frames)
                    pairs = make_pairs(imgs, scene_graph=scenegraph_type, prefilter=None, symmetrize=True)
                    output = inference(pairs, model, device, batch_size=batch_size, verbose=not silent)
                    gs_use_pts, gs_use_colors = intermediate_pcds(pairs, output, imgs,args)
                    # save
                    gts_save = {}
                    gts_save['gs_use_pts'] = gs_use_pts
                    gts_save['gs_use_colors'] = gs_use_colors
                    # with open(f'{outdir}/{seq_name}/gs_use_pts.pkl', 'wb') as f:
                    #     pickle.dump(gts_save, f)

                    del imgs, pairs, output
                    gc.collect()

            if args.test_gs:
                pose_ckpt = "/scratch/yy561/monst3r/paper_example/tum_rgb_refine_pose_copy/rgbd_dataset_freiburg3_walking_static_seq_390_419/traj/gs_camera_poses.pt"
                pose_ckpt = torch.load(pose_ckpt, map_location=args.device)
                global_cam_poses_est = pose_ckpt['global_cam_poses_est']
                runner.trainset.set_camera_to_world(global_cam_poses_est)
                runner.train_gs_doma(args.sh_degree,initial_train_step = 1000) # inited splat, doma doma optimizer, doma scheduler
                return None, None, None
            elif args.draw_frame:
                runner.draw_frame(0,device=runner.device)
                return None, None, None
            else:
                refined_pose = runner.train(monst3r_pointmap = gs_use_pts, monst3r_pointcolor = gs_use_colors)['global_cam_poses_est']
        else:
            if args.test_gs:
                pose_ckpt = "/scratch/yy561/monst3r/paper_example/tum_rgb_refine_pose_copy/rgbd_dataset_freiburg3_walking_static_seq_390_419/traj/gs_camera_poses.pt"
                pose_ckpt = torch.load(pose_ckpt, map_location=args.device)
                global_cam_poses_est = pose_ckpt['global_cam_poses_est']
                runner.trainset.set_camera_to_world(global_cam_poses_est)
                runner.train_gs_doma_sem(args.sh_degree,initial_train_step = 8000) # inited splat, doma doma optimizer, doma scheduler
                # runner.train_gs_doma(args.sh_degree,global_cam_poses_est)
                return None, None, None
            elif args.draw_frame:
                runner.draw_frame(0,device=runner.device,opt=args)
                return None, None, None
            else:
                refined_pose = runner.train()['global_cam_poses_est']
        # refined_pose = runner.train()['global_cam_poses_est']
        # transform back
        refined_pose = torch.cat([torch.Tensor(pose).unsqueeze(0) for pose in refined_pose], dim=0)
        refined_pose = refined_pose @ torch.inverse(T)

        # turn to quaternion
        gs_pose_R = refined_pose[:,:3, :3]
        gs_pose_R = gs_pose_R.cpu().numpy()
        quat = Rotation.from_matrix(gs_pose_R)
        quat = quat.as_quat()
        quat = torch.Tensor(quat)
        # translation
        t = refined_pose[:,:3, 3]
        # set the camera pose
        camera_pose = torch.cat([quat,t], dim=1).to(device)

        # scene.post_set_pose(camera_pose)
        # scene.im_poses = torch.nn.Parameter(camera_pose)
        # save the refined camera poses
        with Path(f"{outdir}/{seq_name}/refined_pose.txt").open("w") as f:
            for i in range(len(camera_pose)):
                ith_pose = camera_pose[i]
                f.write(
                    f"{i} {tostr(ith_pose[-3::].detach().cpu().numpy())} {tostr(ith_pose[[0,1,2,3]].detach().cpu().numpy())}\n"
                )

        # gs_end_time = time.time()
        # gs_refine_time = gs_end_time - gs_end_time
        # # print(f"GS refine completed in {gs_end_time-end_time:.2f} seconds. Output saved in {outdir}/{seq_name}")
        # total_time = gs_end_time - start_time
        # write the times to a file
        gs_refine_end_time = time.time()
        gs_refine_time = gs_refine_end_time - gs_refine_start_time
        with Path(f"{outdir}/{seq_name}/time.txt").open("a") as f:
            # f.write(f"MonST3R time: {monst3r_time:.2f}\n")
            f.write(f"GS refine time: {gs_refine_time:.2f}\n")
        # # run evaluation 
        # eval_monst3r_gs_poses(args)
        
    
    # return scene, outfile, imgs
    return None, None, None


def get_pose_from_gs(args):
    model = Runner(local_rank=args.local_rank, world_rank=args.world_rank, world_size=args.world_size, opt=args)
    args.workspace = f'{args.output_dir}/{args.seq_name}'
    args.data_dir = args.input_dir
    
    gs_pose = model.train()['global_cam_poses_est']
    breakpoint()
    

def set_scenegraph_options(inputfiles, winsize, refid, scenegraph_type):
    # if inputfiles[0] is a video, set the num_files to 200
    if inputfiles is not None and len(inputfiles) == 1 and inputfiles[0].name.endswith(('.mp4', '.avi', '.mov')):
        num_files = 200
    else:
        num_files = len(inputfiles) if inputfiles is not None else 1
    max_winsize = max(1, math.ceil((num_files-1)/2))
    if scenegraph_type == "swin" or scenegraph_type == "swin2stride" or scenegraph_type == "swinstride":
        winsize = gradio.Slider(label="Scene Graph: Window Size", value=min(max_winsize,5),
                                minimum=1, maximum=max_winsize, step=1, visible=True)
        refid = gradio.Slider(label="Scene Graph: Id", value=0, minimum=0,
                              maximum=num_files-1, step=1, visible=False)
    elif scenegraph_type == "oneref":
        winsize = gradio.Slider(label="Scene Graph: Window Size", value=max_winsize,
                                minimum=1, maximum=max_winsize, step=1, visible=False)
        refid = gradio.Slider(label="Scene Graph: Id", value=0, minimum=0,
                              maximum=num_files-1, step=1, visible=True)
    else:
        winsize = gradio.Slider(label="Scene Graph: Window Size", value=max_winsize,
                                minimum=1, maximum=max_winsize, step=1, visible=False)
        refid = gradio.Slider(label="Scene Graph: Id", value=0, minimum=0,
                              maximum=num_files-1, step=1, visible=False)
    return winsize, refid


def main_demo(tmpdirname, model, device, image_size, server_name, server_port, silent=False, args=None):
    recon_fun = functools.partial(get_reconstructed_scene, args, tmpdirname, model, device, silent, image_size)
    model_from_scene_fun = functools.partial(get_3D_model_from_scene, tmpdirname, silent)
    with gradio.Blocks(css=""".gradio-container {margin: 0 !important; min-width: 100%};""", title="DUSt3R Demo") as demo:
        # scene state is save so that you can change conf_thr, cam_size... without rerunning the inference
        scene = gradio.State(None)
        gradio.HTML(f'<h2 style="text-align: center;">DUSt3R Demo</h2>')
        with gradio.Column():
            inputfiles = gradio.File(file_count="multiple")
            with gradio.Row():
                schedule = gradio.Dropdown(["linear", "cosine"],
                                           value='linear', label="schedule", info="For global alignment!")
                niter = gradio.Number(value=300, precision=0, minimum=0, maximum=5000,
                                      label="num_iterations", info="For global alignment!")
                seq_name = gradio.Textbox(label="Sequence Name", placeholder="NULL", value=args.seq_name, info="For evaluation")
                scenegraph_type = gradio.Dropdown(["complete", "swin", "oneref", "swinstride", "swin2stride"],
                                                  value='swinstride', label="Scenegraph",
                                                  info="Define how to make pairs",
                                                  interactive=True)
                winsize = gradio.Slider(label="Scene Graph: Window Size", value=5,
                                        minimum=1, maximum=1, step=1, visible=False)
                refid = gradio.Slider(label="Scene Graph: Id", value=0, minimum=0, maximum=0, step=1, visible=False)

            run_btn = gradio.Button("Run")

            with gradio.Row():
                # adjust the confidence thresholdx
                min_conf_thr = gradio.Slider(label="min_conf_thr", value=1.1, minimum=0.0, maximum=20, step=0.01)
                # adjust the camera size in the output pointcloud
                cam_size = gradio.Slider(label="cam_size", value=0.05, minimum=0.001, maximum=0.1, step=0.001)
                # adjust the temporal smoothing weight
                temporal_smoothing_weight = gradio.Slider(label="temporal_smoothing_weight", value=0.01, minimum=0.0, maximum=0.1, step=0.001)
                # add translation weight
                translation_weight = gradio.Textbox(label="translation_weight", placeholder="1.0", value="1.0", info="For evaluation")
                # change to another model
                new_model_weights = gradio.Textbox(label="New Model", placeholder=args.weights, value=args.weights, info="Path to updated model weights")
            with gradio.Row():
                as_pointcloud = gradio.Checkbox(value=True, label="As pointcloud")
                # two post process implemented
                mask_sky = gradio.Checkbox(value=False, label="Mask sky")
                clean_depth = gradio.Checkbox(value=True, label="Clean-up depthmaps")
                transparent_cams = gradio.Checkbox(value=False, label="Transparent cameras")
                # not to show camera
                show_cam = gradio.Checkbox(value=True, label="Show Camera")
                shared_focal = gradio.Checkbox(value=True, label="Shared Focal Length")
                use_davis_gt_mask = gradio.Checkbox(value=False, label="Use GT Mask (DAVIS)")
            with gradio.Row():
                flow_loss_weight = gradio.Slider(label="Flow Loss Weight", value=0.01, minimum=0.0, maximum=1.0, step=0.001)
                flow_loss_start_iter = gradio.Slider(label="Flow Loss Start Iter", value=0.1, minimum=0, maximum=1, step=0.01)
                flow_loss_threshold = gradio.Slider(label="Flow Loss Threshold", value=25, minimum=0, maximum=100, step=1)
                # for video processing
                fps = gradio.Slider(label="FPS", value=0, minimum=0, maximum=60, step=1)
                num_frames = gradio.Slider(label="Num Frames", value=100, minimum=0, maximum=200, step=1)

            outmodel = gradio.Model3D()
            outgallery = gradio.Gallery(label='rgb,depth,confidence, init_conf', columns=4, height="100%")

            # events
            scenegraph_type.change(set_scenegraph_options,
                                   inputs=[inputfiles, winsize, refid, scenegraph_type],
                                   outputs=[winsize, refid])
            inputfiles.change(set_scenegraph_options,
                              inputs=[inputfiles, winsize, refid, scenegraph_type],
                              outputs=[winsize, refid])
            run_btn.click(fn=recon_fun,
                          inputs=[inputfiles, schedule, niter, min_conf_thr, as_pointcloud,
                                  mask_sky, clean_depth, transparent_cams, cam_size, show_cam,
                                  scenegraph_type, winsize, refid, seq_name, new_model_weights, 
                                  temporal_smoothing_weight, translation_weight, shared_focal, 
                                  flow_loss_weight, flow_loss_start_iter, flow_loss_threshold, use_davis_gt_mask,
                                  fps, num_frames],
                          outputs=[scene, outmodel, outgallery])
            min_conf_thr.release(fn=model_from_scene_fun,
                                 inputs=[scene, min_conf_thr, as_pointcloud, mask_sky,
                                         clean_depth, transparent_cams, cam_size, show_cam],
                                 outputs=outmodel)
            cam_size.change(fn=model_from_scene_fun,
                            inputs=[scene, min_conf_thr, as_pointcloud, mask_sky,
                                    clean_depth, transparent_cams, cam_size, show_cam],
                            outputs=outmodel)
            as_pointcloud.change(fn=model_from_scene_fun,
                                 inputs=[scene, min_conf_thr, as_pointcloud, mask_sky,
                                         clean_depth, transparent_cams, cam_size, show_cam],
                                 outputs=outmodel)
            mask_sky.change(fn=model_from_scene_fun,
                            inputs=[scene, min_conf_thr, as_pointcloud, mask_sky,
                                    clean_depth, transparent_cams, cam_size, show_cam],
                            outputs=outmodel)
            clean_depth.change(fn=model_from_scene_fun,
                               inputs=[scene, min_conf_thr, as_pointcloud, mask_sky,
                                       clean_depth, transparent_cams, cam_size, show_cam],
                               outputs=outmodel)
            transparent_cams.change(model_from_scene_fun,
                                    inputs=[scene, min_conf_thr, as_pointcloud, mask_sky,
                                            clean_depth, transparent_cams, cam_size, show_cam],
                                    outputs=outmodel)
    demo.launch(share=args.share, server_name=server_name, server_port=server_port)

if __name__ == '__main__':
    # set random seed
    torch.manual_seed(42)
    np.random.seed(42)

    parser = get_args_parser()
    args = parser.parse_args()

    if args.output_dir is not None:
        tmp_path = args.output_dir
        os.makedirs(tmp_path, exist_ok=True)
        tempfile.tempdir = tmp_path

    if args.server_name is not None:
        server_name = args.server_name
    else:
        server_name = '0.0.0.0' if args.local_network else '127.0.0.1'

    if args.weights is not None and os.path.exists(args.weights):
        weights_path = args.weights
    else:
        weights_path = args.model_name

    model = AsymmetricCroCo3DStereo.from_pretrained(weights_path).to(args.device)

    # Use the provided output_dir or create a temporary directory
    tmpdirname = args.output_dir if args.output_dir is not None else tempfile.mkdtemp(suffix='dust3r_gradio_demo')

    if not args.silent:
        print('Outputting stuff in', tmpdirname)

    if args.input_dir is not None:
        # if "pkl" in args.input_dir:   # input_dir is a metadata file
        # try:
        #     print('starting gradio')
        #     if args.doma_eval:
        #         #args.workspace = f'{args.output_dir}/{seq_name}' 
        #         args.data_dir = args.input_dir
        #         args.monst3r_dir = f'{args.output_dir}/{seq_name}' 
        #         args.camera_smoothness_lambda = 1
        #         # args.datatype = 'custom'
        #         args.datatype = 'monst3r' if "own" not in args.output_dir else 'custom'

        #         runner = Runner(local_rank=args.local_rank, world_rank=args.world_rank, world_size=args.world_size, opt=args)
        # except:
        #     pass
        
        if 'meta' in args.input_dir:
            # find parent dir 
            # parent_dir = os.path.dirname(args.input_dir)

            pkl_paths = [os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir) if f.endswith('.pkl')]
            for pkl_path in pkl_paths:
                # with open(args.input_dir, 'rb') as f:
                with open(pkl_path, 'rb') as f:
                    meta = pickle.load(f)
                # loop every index list of meta
                seq_ind = meta['selected_frames']
                cam_poses = meta['cam_poses']
                dataset_names = meta['dataset']
                deataset_paths = meta['dataset_path']
                image_paths = meta['image_paths']

                for i, ind in enumerate(seq_ind):
                    input_files = image_paths[i]
                    start_idx =ind[0]
                    end_idx = ind[-1]
                    args.seq_name = f"seq_{start_idx}_{end_idx}"
                    args.input_dir = input_files

                    recon_fun = functools.partial(get_reconstructed_scene, args, tmpdirname, model, args.device, args.silent, args.image_size)
                    scene, outfile, imgs = recon_fun(
                        filelist=input_files,
                        schedule='linear',
                        niter=300,
                        min_conf_thr=1.1,
                        as_pointcloud=True,
                        mask_sky=False,
                        clean_depth=True,
                        transparent_cams=False,
                        cam_size=0.05,
                        show_cam=True,
                        scenegraph_type='swinstride',
                        winsize=5,
                        refid=0,
                        seq_name=args.seq_name,
                        new_model_weights=args.weights,
                        temporal_smoothing_weight=0.01,
                        translation_weight='1.0',
                        shared_focal=True,
                        flow_loss_weight=0.01,
                        flow_loss_start_iter=0.1,
                        flow_loss_threshold=25,
                        use_gt_mask=args.use_gt_davis_masks,
                        fps=args.fps,
                        num_frames=args.num_frames
                    )
                # 
        elif 'pkl' in args.input_dir:
            with open(args.input_dir, 'rb') as f:
            # with open("/scratch/yy561/monst3r/data/tum/tum_cam_poses_all_theta0.07.pkl", 'rb') as f:    
                pkl = pickle.load(f)
            # loop every index list of meta
            seq_ind = pkl['selected_frames']
            cam_poses = pkl['cam_poses']
            dataset_names = pkl['dataset'] 
            deataset_paths = pkl['dataset_path']
            image_paths = pkl['image_paths']
            print("len(pkl)", len(seq_ind))
            import imageio
            for i, ind in enumerate(seq_ind):
                input_files = image_paths[i]
                start_idx =ind[0]
                end_idx = ind[-1]
                dataset_name = dataset_names[i]
                args.seq_name = f"{dataset_name}_seq_{start_idx}_{end_idx}"
                args.input_dir = input_files
                image = imageio.imread(input_files[0])
                image_size = image.size
                # if args.data_factor > 1:
                #     image_size = [image.shape[0]//args.data_factor, image.shape[1]//args.data_factor]
                # args.image_size = image_size

                recon_fun = functools.partial(get_reconstructed_scene, args, tmpdirname, model, args.device, args.silent, args.image_size)
                scene, outfile, imgs = recon_fun(
                    filelist=input_files,
                    schedule='linear',
                    niter=300,
                    min_conf_thr=1.1,
                    as_pointcloud=True,
                    mask_sky=False,
                    clean_depth=True,
                    transparent_cams=False,
                    cam_size=0.05,
                    show_cam=True,
                    scenegraph_type='swinstride',
                    winsize=5,
                    refid=0,
                    seq_name=args.seq_name,
                    new_model_weights=args.weights,
                    temporal_smoothing_weight=0.01,
                    translation_weight='1.0',
                    shared_focal=True,
                    flow_loss_weight=0.01,
                    flow_loss_start_iter=0.1,
                    flow_loss_threshold=25,
                    use_gt_mask=args.use_gt_davis_masks,
                    fps=args.fps,
                    num_frames=args.num_frames,
                )


        # Process images in the input directory with default parameters
        elif os.path.isdir(args.input_dir):    # input_dir is a directory of images
            input_files = [os.path.join(args.input_dir, fname) for fname in sorted(os.listdir(args.input_dir))]
            recon_fun = functools.partial(get_reconstructed_scene, args, tmpdirname, model, args.device, args.silent, args.image_size)
             # Call the function with default parameters
            scene, outfile, imgs = recon_fun(
                filelist=input_files,
                schedule='linear',
                niter=300,
                min_conf_thr=1.1,
                as_pointcloud=True,
                mask_sky=False,
                clean_depth=True,
                transparent_cams=False,
                cam_size=0.05,
                show_cam=True,
                scenegraph_type='swinstride',
                winsize=5,
                refid=0,
                seq_name=args.seq_name,
                new_model_weights=args.weights,
                temporal_smoothing_weight=0.01,
                translation_weight='1.0',
                shared_focal=True,
                flow_loss_weight=0.01,
                flow_loss_start_iter=0.1,
                flow_loss_threshold=25,
                use_gt_mask=args.use_gt_davis_masks,
                fps=args.fps,
                num_frames=args.num_frames,
            )
            
        else:   # input_dir is a video
            input_files = [args.input_dir]
            recon_fun = functools.partial(get_reconstructed_scene, args, tmpdirname, model, args.device, args.silent, args.image_size)
            # Call the function with default parameters
            scene, outfile, imgs = recon_fun(
                filelist=input_files,
                schedule='linear',
                niter=300,
                min_conf_thr=1.1,
                as_pointcloud=True,
                mask_sky=False,
                clean_depth=True,
                transparent_cams=False,
                cam_size=0.05,
                show_cam=True,
                scenegraph_type='swinstride',
                winsize=5,
                refid=0,
                seq_name=args.seq_name,
                new_model_weights=args.weights,
                temporal_smoothing_weight=0.01,
                translation_weight='1.0',
                shared_focal=True,
                flow_loss_weight=0.01,
                flow_loss_start_iter=0.1,
                flow_loss_threshold=25,
                use_gt_mask=args.use_gt_davis_masks,
                fps=args.fps,
                num_frames=args.num_frames,
            )
            print(f"Processing completed. Output saved in {tmpdirname}/{args.seq_name}")
    else:
        # Launch Gradio demo
        main_demo(tmpdirname, model, args.device, args.image_size, server_name, args.server_port, silent=args.silent, args=args)
