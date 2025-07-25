# SmallGS: Gaussian Splatting-based Camera Pose Estimation for Small-Baseline Videos

**SmallGS**  processes a dynamic video with small baseline to obtain the camera poses. It leverages Gaussian splatting to optimize camera poses while mitigating dynamic object interference via predicted semantic masks. It **does not rely on 3D alignments or triangulation**, alleviating the instability in camera pose estimation caused by limited parallax and weak geometric
constraints.


This repository is the official implementation of the paper:

[**SmallGS: Gaussian Splatting-based Camera Pose Estimation for Small-Baseline Videos**](https://arxiv.org/pdf/2504.17810)
[*Yuxin Yao*](https://yuxinyao620.github.io/)
[*Yan Zhang*](https://yz-cnsdqz.github.io/)
[*Zhening Huang*](https://zheninghuang.github.io/)
[*Joan Lasenby*](https://www.eng.cam.ac.uk/profiles/jl221)
Arxiv, 2025. [**[Project Page]**](https://yuxinyao620.github.io/SmallGS/) [**[Paper]**](https://arxiv.org/pdf/2504.17810)
![Teaser](assets/teaser.jpg)
 
### Installation
1. Clone SmallGS
```
git clone --recursive https://github.com/YuxinYao620/SmallGS-release.git
```

2. Download Checkpoints following  [MonST3R](https://github.com/Junyi42/monst3r.git)

3. Create the environment, here we use anaconda here:
```
conda env create -f environment.yml
```
3. If you didn't recursively clone this repository, set up the croco and viser according to MonST3R.
```
git clone https://github.com/junyi42/viser viser
git clone https://github.com/junyi42/croco croco
```
4. Download the TUM-Dynamics dataset following MonST3R. Then proprocess: 

```
cd data
bash download_tum_dynamics.sh
cd ..
python datasets_preprocess/camera_preprocess_tum.py 
```
5. Predict the camera pose. 
- "--gs_pose" triggers the SmallGS. 
- "--dino" triggers the SmallGS with DINOv2 feature maps. 
- "--dino_dim" triggers the number of channels in the DINOv2 feature map used for SmallGS. 
```
python main.py --input_dir data/tum/tum_meta.pkl  --output_dir tum/result_dir/ --gs_pose --dino --dino_dim
```
6. Evaluate the predicted camera trajectories for ATE, RPE and $\delta v$
```
python batch_eval_camera.py --meta_dir data/tum/tum_meta.pkl --result_dir tum/result_dir/
python batch_eval_camera_velocity.py --meta_dir data/tum/tum_meta.pkl --result_dir tum/result_dir/
```
### Citation
If you find our work useful, please cite:
```
@article{yao2025smallgs,
  title={SmallGS: Gaussian Splatting-based Camera Pose Estimation for Small-Baseline Videos},
  author={Yao, Yuxin and Zhang, Yan and Huang, Zhening and Lasenby, Joan},
  journal={arXiv preprint arXiv:2504.17810},
  year={2025}
}
```
### Acknowledgement
Our code is based on MonST3R and Gsplat, and our visualization code is based on Viser. We thank the authors for their excellent work!

<!-- ## Getting Started
# git clone https://github.com/junyi42/croco croco




1. Clone MonST3R.
```bash
git clone --recursive https://github.com/junyi42/monst3r
cd monst3r
## if you have already cloned monst3r:
# git clone https://github.com/junyi42/viser viser
# git clone https://github.com/junyi42/croco croco
```

2. Create the environment, here we show an example using conda.
```bash
conda create -n monst3r python=3.11 cmake=3.14.0
conda activate monst3r 
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia  # use the correct version of cuda for your system
pip install -r requirements.txt
# Optional: you can also install additional packages to:
# - training
# - evaluation on camera pose
# - dataset preparation
pip install -r requirements_optional.txt
```

3. Optional, install 4d visualization tool, `viser`.
```bash
pip install -e viser
```

4. Optional, compile the cuda kernels for RoPE (as in CroCo v2).
```bash
# DUST3R relies on RoPE positional embeddings for which you can compile some cuda kernels for faster runtime.
cd croco/models/curope/
python setup.py build_ext --inplace
cd ../../../
```

### Download Checkpoints

We currently provide fine-tuned model weights for MonST3R, which can be downloaded on [Google Drive](https://drive.google.com/file/d/1Z1jO_JmfZj0z3bgMvCwqfUhyZ1bIbc9E/view?usp=sharing) or via [Hugging Face](https://huggingface.co/Junyi42/MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt).


To download the weights of MonST3R and optical flow models, run the following commands:
```bash
# download the weights
cd data
bash download_ckpt.sh
cd ..
```

### Inference

To run the inference code, you can use the following command:
```bash
python demo.py # launch GUI, input can be a folder or a video
```

The results will be saved in the `demo_tmp/{Sequence Name}` (by default is `demo_tmp/NULL`) folder for future visualization.

You can also run the inference code in a non-interactive mode:
```bash
python demo.py --input demo_data/lady-running --output_dir demo_tmp --seq_name lady-running
# use video as input: --input demo_data/lady-running.mp4 --num_frames 65
```

> Currently, it takes about 33G VRAM to run the inference code on a 16:9 video of 65 frames. Use less frames or disable the `flow_loss` could reduce the memory usage. We are **welcome to any PRs** to improve the memory efficiency (one reasonable way is to implement window-wise optimzation in `optimizer.py`).

### Visualization

To visualize the interactive 4D results, you can use the following command:
```bash
python viser/visualizer_monst3r.py --data demo_tmp/lady-running
# to remove the floaters of foreground: --init_conf --fg_conf_thre 1.0 (thre can be adjusted)
```

## Evaluation

We provide here an example of joint dense reconstruction and camera pose estimation on the **DAVIS** dataset. 

First, download the dataset:
```bash
cd data; python download_davis.py; cd ..
```

Then, run the evaluation script:
```bash
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=29604 launch.py --mode=eval_pose  \
    --pretrained="checkpoints/MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt.pth"   \
    --eval_dataset=davis --output_dir="results/davis_joint" 
    # To use the ground truth dynamic mask for davis, add: --use_gt_mask
```

You could then use the `viser` to visualize the results:
```bash
python viser/visualizer_monst3r.py --data results/davis_joint/bear
# if the dynamic mask is noisy, one could visualize per-frame pointcloud by adding: --no_mask
```

#### For the complete scripts to evaluate the camera pose / video depth / single-frame depth estimation on the **Sintel**, **Bonn**, **KITTI**, **NYU-v2**, **TUM-dynamics**, **ScanNet**, and **DAVIS** datasets. Please refer to the [evaluation_script.md](data/evaluation_script.md) for more details.


## Training

Please refer to the [prepare_training.md](data/prepare_training.md) for preparing the pretrained models and training/testing datasets.

Then, you can train the model using the following command:
```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=29604 launch.py  --mode=train \
    --train_dataset="10_000 @ PointOdysseyDUSt3R(dset='train', z_far=80, dataset_location='data/point_odyssey', S=2, aug_crop=16, resolution=[(512, 288), (512, 384), (512, 336)], transform=ColorJitter, strides=[1,2,3,4,5,6,7,8,9], dist_type='linear_1_2', aug_focal=0.9)+ 5_000 @ TarTanAirDUSt3R(dset='Hard', z_far=80, dataset_location='data/tartanair', S=2, aug_crop=16, resolution=[(512, 288), (512, 384), (512, 336)], transform=ColorJitter, strides=[1,2,3,4,5,6,7,8,9], dist_type='linear_1_2', aug_focal=0.9)+ 1_000 @ SpringDUSt3R(dset='train', z_far=80, dataset_location='data/spring', S=2, aug_crop=16, resolution=[(512, 288), (512, 384), (512, 336)], transform=ColorJitter, strides=[1,2,3,4,5,6,7,8,9], dist_type='linear_1_2', aug_focal=0.9)+ 4_000 @ Waymo(ROOT='data/waymo_processed', pairs_npz_name='waymo_pairs_video.npz', aug_crop=16, resolution=[(512, 288), (512, 384), (512, 336)], transform=ColorJitter, aug_focal=0.9)"   \
    --test_dataset="1000 @ PointOdysseyDUSt3R(dset='test', z_far=80, dataset_location='data/point_odyssey', S=2, strides=[1,2,3,4,5,6,7,8,9], resolution=[(512, 288)], seed=777)+ 1000 @ SintelDUSt3R(dset='final', z_far=80, S=2, strides=[1,2,3,4,5,6,7,8,9], resolution=[(512, 224)], seed=777)"   \
    --train_criterion="ConfLoss(Regr3D(L21, norm_mode='avg_dis'), alpha=0.2)"  \
    --test_criterion="Regr3D_ScaleShiftInv(L21, gt_scale=True)"   \
    --pretrained="checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"   \
    --lr=0.00005 --min_lr=1e-06 --warmup_epochs=3 --epochs=50 --batch_size=4 --accum_iter=4  \
    --save_freq=3 --keep_freq=5 --eval_freq=1  \
    --output_dir="results/MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt"
```

## Citation

If you find our work useful, please cite:

```bibtex
@article{zhang2024monst3r,
  author    = {Zhang, Junyi and Herrmann, Charles and Hur, Junhwa and Jampani, Varun and Darrell, Trevor and Cole, Forrester and Sun, Deqing and Yang, Ming-Hsuan},
  title     = {MonST3R: A Simple Approach for Estimating Geometry in the Presence of Motion},
  journal   = {arXiv preprint arxiv:2410.03825},
  year      = {2024}
}
```

## Acknowledgements
Our code is based on [DUSt3R](https://github.com/naver/dust3r) and [CasualSAM](https://github.com/ztzhang/casualSAM), our camera pose estimation evaluation script is based on [LEAP-VO](https://github.com/chiaki530/leapvo), and our visualization code is based on [Viser](https://github.com/nerfstudio-project/viser). We thank the authors for their excellent work! -->
