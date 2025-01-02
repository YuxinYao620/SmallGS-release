import torch
import argparse
import pandas as pd
import sys
from gs.gs_diff import Runner
from time import time
import os


if __name__ == '__main__':
    # See https://stackoverflow.com/questions/27433316/how-to-get-argparse-to-read-arguments-from-a-file-with-an-option-rather-than-pre
    class LoadFromFile (argparse.Action):
        def __call__ (self, parser, namespace, values, option_string = None):
            with values as f:
                # parse arguments in the file and store them in the target namespace
                parser.parse_args(f.read().split(), namespace)

    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=str, default='workspace')
    parser.add_argument('--seed', default=42)

    # Gaussian splatting args
    parser.add_argument('--disable_viewer', action='store_true', help="Disable viewer")
    parser.add_argument('--disable_plot', action='store_true', help="Disable plot")
    parser.add_argument('--ckpt', type=str, default=None, help="Path to the .pt file. If provided, it will skip training and render a video")
    parser.add_argument('--pose_ckpt', type=str, default=None, help="Path to the pose checkpoint file")

    parser.add_argument('--data_dir', type=str, default="data/360_v2/garden", help="Path to the Mip-NeRF 360 dataset")
    parser.add_argument('--data_factor', type=int, default=4, help="Downsample factor for the dataset")
    parser.add_argument('--result_dir', type=str, default="results/garden", help="Directory to save results")
    
    parser.add_argument('--global_scale', type=float, default=1.0, help="A global scaler that applies to the scene size related parameters")
    parser.add_argument('--batch_size', type=int, default=1, help="Batch size for training. Learning rates are scaled automatically")
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
    parser.add_argument('--reset_every', type=int, default=30000, help="Reset opacities every this steps")
    parser.add_argument('--refine_every', type=int, default=100, help="Refine GSs every this steps")

    parser.add_argument('--init_opa', type=float, default=0.1, help="Initial opacity of GS")
    parser.add_argument('--init_scale', type=float, default=1.0, help="Initial scale of GS")
    parser.add_argument('--ssim_lambda', type=float, default=0.2, help="Weight for SSIM loss")
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
    parser.add_argument('--depth_loss', action='store_true', help="Enable depth loss. (experimental)")
    parser.add_argument('--depth_lambda', type=float, default=1e-2, help="Weight for depth loss")

    parser.add_argument('--rotation_lr', type=float, default=1e-3, help="Learning rate for rotation")
    parser.add_argument('--datatype', type=str, default="custom", help="Datatype")
    parser.add_argument('--doma', action='store_true', help="Enable doma")
    parser.add_argument('--motion_model_name', type=str, default="transfield4d", help="Motion model name")
    parser.add_argument('--motion_color_name', type=str, default="transfield4d", help="Motion color name")
    parser.add_argument('--elastic_loss_weight', type=float, default=0.0, help="Elastic loss weight")
    parser.add_argument('--homo_loss_weight', type=float, default=0.1, help="Homo loss weight")
    parser.add_argument('--semantics_opt', action='store_true', help="Enable semantics optimization")
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
    parser.add_argument('--camera_smoothness_loss', action='store_true', help="Enable camera smoothness loss")

    parser.add_argument('--local_rank',type=int, default=0)
    parser.add_argument('--world_rank',type=int, default=0)
    parser.add_argument('--world_size',type=int, default=1)

    parser.add_argument('--test_gs', action='store_true', help="Test GS")
    parser.add_argument('--fix_strategy', action='store_true', help="Fix gaussian number ")
    parser.add_argument('--use_spec_points', action = 'store_true', help="Add SMPL vertices to the intialization for gaussian splatting")
    parser.add_argument('--background_color', type=float, nargs='*', default=None, help="Background color for the renderer")

    def adjust_steps(opt, factor: float):
        opt.eval_steps = [int(i * factor) for i in opt.eval_steps]
        opt.save_steps = [int(i * factor) for i in opt.save_steps]
        opt.max_steps = int(opt.max_steps * factor)
        opt.sh_degree_interval = int(opt.sh_degree_interval * factor)
        opt.refine_start_iter = int(opt.refine_start_iter * factor)
        opt.refine_stop_iter = int(opt.refine_stop_iter * factor)
        opt.reset_every = int(opt.reset_every * factor)
        opt.refine_every = int(opt.refine_every * factor)

    opt = parser.parse_args()



    if opt.seed is not None:
        seed_everything(int(opt.seed))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = Runner(local_rank=opt.local_rank, world_rank=opt.world_rank, world_size=opt.world_size, opt=opt)
    
    # time 
    start_time = time.time()

    model.device = device
    train_set = model.trainset
    # get camera poses
    if len(model.trainset) < 2 :
        global_cam_poses_est = torch.eye(4).unsqueeze(0).numpy()
    else:
        if opt.pose_ckpt is not None and os.path.exists(opt.pose_ckpt):
            pose_ckpt = torch.load(opt.pose_ckpt, map_location=device)
            global_cam_poses_est = pose_ckpt['global_cam_poses_est']
        else:
            print("est camera poses")
            adjust_steps(opt, opt.steps_scaler)
            camera_poses = model.train() # get estimated camera poses
            global_cam_poses_est = camera_poses['global_cam_poses_est']
                

    from gs.gs_diff import create_splats_with_optimizers,create_optimizers, create_splats_with_optimizers_sem


    if opt.doma:
        if opt.test_gs:

            frame_0_depth = train_set[0]['depth'].to(device)
            frame_0_rgb = train_set[0]['image'].reshape(-1, 3).to(device) / 255.0
            frame_0_semantics = train_set[0]['mask'].flatten().to(device)
            model.splats, gs_optimizer = create_splats_with_optimizers_sem(
                    model.parser,
                    init_type=opt.init_type,
                    init_num_pts=opt.init_num_pts,
                    init_extent=opt.init_extent,
                    init_opacity=opt.init_opa,
                    init_scale=opt.init_scale,
                    scene_scale=model.scene_scale,
                    sh_degree=opt.sh_degree,
                    sparse_grad=opt.sparse_grad,
                    batch_size=opt.batch_size,
                    feature_dim=32 if opt.app_opt else None,
                    device=device,
                    world_rank=model.world_rank,
                    world_size=model.world_size,
                    predict_depth = frame_0_depth,
                    predict_rgb = frame_0_rgb,
                    semantic_label = frame_0_semantics
                )
            
        # # try load doma and gsplat
        elif os.path.exists(f"{model.ckpt_dir}/gs_doma.pt") and os.path.exists(f"{model.ckpt_dir}/doma.pt"):
            print("Loading pre-trained gs_doma.pt")
            gs_pre = torch.load(f"{model.ckpt_dir}/gs_doma.pt")

            frame_0_depth = train_set[0]['depth'].to(device)
            frame_0_rgb = train_set[0]['image'].reshape(-1, 3).to(device) / 255.0
            if 'labels' in gs_pre['splats']:
            # if True:
                frame_0_semantics = train_set[0]['mask'].flatten().to(device)
                model.splats, gs_optimizer = create_splats_with_optimizers_sem(
                    model.parser,
                    init_type=opt.init_type,
                    init_num_pts=opt.init_num_pts,
                    init_extent=opt.init_extent,
                    init_opacity=opt.init_opa,
                    init_scale=opt.init_scale,
                    scene_scale=model.scene_scale,
                    sh_degree=opt.sh_degree,
                    sparse_grad=opt.sparse_grad,
                    batch_size=opt.batch_size,
                    feature_dim=32 if opt.app_opt else None,
                    device=device,
                    world_rank=model.world_rank,
                    world_size=model.world_size,
                    predict_depth = frame_0_depth,
                    predict_rgb = frame_0_rgb,
                    semantic_label = gs_pre['splats']['labels']
                )
                for k in model.splats.keys():
                    model.splats[k].data = gs_pre['splats'][k]
                # model.splats.load_state_dict(gs_pre['splats'])
                gs_optimizer_states = torch.load(f"{model.ckpt_dir}/gs_optimizer.pt")
                for name,optimizer in gs_optimizer.items():
                    if name =='labels':
                        continue
                    if len(gs_optimizer_states[name]) > 0:
                        optimizer.load_state_dict(gs_optimizer_states[name])
                        #print(lr)
                        print("learning rate for ", name, optimizer.param_groups[0]['lr'])
                
            
            else:
                model.splats, gs_optimizer =  create_splats_with_optimizers(
                    model.parser,
                    init_type=opt.init_type,
                    init_num_pts=opt.init_num_pts,
                    init_extent=opt.init_extent,
                    init_opacity=opt.init_opa,
                    init_scale=opt.init_scale,
                    scene_scale=model.scene_scale,
                    sh_degree=opt.sh_degree,
                    sparse_grad=opt.sparse_grad,
                    batch_size=opt.batch_size,
                    feature_dim=32 if opt.app_opt else None,
                    device=device,
                    world_rank=model.world_rank,
                    world_size=model.world_size,
                    predict_depth = frame_0_depth,
                    predict_rgb = frame_0_rgb
                )
                # for name,value in gs_pre['splats'].items():
                #     model.splats[name] = value
                # breakpoint()
                for k in model.splats.keys():
                    model.splats[k].data = gs_pre['splats'][k]
                # model.splats.load_state_dict(gs_pre['splats'])

                gs_optimizer = create_optimizers(model.splats,opt.batch_size,model.world_size,model.scene_scale, 
                                                opt.sparse_grad,feature_dim = 32 if opt.app_opt else None, sh_degree = opt.sh_degree)
                gs_optimizer_states = torch.load(f"{model.ckpt_dir}/gs_optimizer.pt")

                for name,optimizer in gs_optimizer.items():
                    if name =='labels':
                        continue
                    optimizer.load_state_dict(gs_optimizer_states[name])
        else:
            print("No pre-trained gs_doma.pt found, training from scratch")
            opt.deform_lr_max_steps = opt.initial_train_step

            if opt.semantics_opt:
                model.train_gs_doma_sem(opt.sh_degree,initial_train_step = opt.initial_train_step) # inited splat, doma doma optimizer, doma scheduler
            else:
                model.train_gs_doma(opt.sh_degree,initial_train_step = opt.initial_train_step)
            gs_optimizer = model.optimizers

            gs_optimizer = create_optimizers(model.splats,opt.batch_size,model.world_size,model.scene_scale, opt.sparse_grad,feature_dim = 32 if opt.app_opt else None, sh_degree = opt.sh_degree)
            # load optimizer state
            gs_optimizer_states = torch.load(f"{model.ckpt_dir}/gs_optimizer.pt")
            for name,optimizer in gs_optimizer.items():
                if name =='labels':
                    continue
                optimizer.load_state_dict(gs_optimizer_states[name])
            
            #intialize gs_app_opt
            if opt.app_opt:
                # load gs_app_opt
                gs_app_opt = model.app_optimizers
                app_opt_states = torch.load(f"{model.ckpt_dir}/gs_app_opt.pt")
                for name, app_opt in gs_app_opt.items():
                    app_opt.load_state_dict(app_opt_states[name])
            else:
                gs_app_opt = None

        # after doma and gs training
        doma_time = time.time()

        # stablize the learning rate
        opt.stable_steps = 0
        gs_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                gs_optimizer["means"], gamma=0.01 ** (1.0 / 1000)
            )
        
        if not opt.test_gs:
            gs_scheduler_states = torch.load(f"{model.ckpt_dir}/gs_scheduler.pt")

            # change lr of scheduler
            gs_scheduler.load_state_dict(gs_scheduler_states)

        if opt.doma:
            doma, doma_optimizer, doma_scheduler = model.init_doma(seq_len=len(train_set), 
                                                                device=device,
                                                                max_steps=5000,
            )
            model.train_gs_doma_sem(opt.sh_degree,initial_train_step = opt.initial_train_step)


    # model.create_strategy()

    model.eval_all(0, model.trainset, torch.stack([data['camtoworld'].squeeze(0) for data in model.trainset]).to(model.device),doma_eval=opt.doma,doma=model.doma)
