import torch
import torch.backends.cudnn as cudnn
import numpy as np
import os
# from dust3r.training import get_args_parser
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import argparse
import glob
import imageio

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_dir", type=str, required=True, default="batch_test_results/scene_recording_20210910_S05_S06_0_ego1/seq_323_336_gs/")
    parser.add_argument("--gt_dir", type=str, required=True, default="batch_test/scene_recording_20210910_S05_S06_0_ego1/images/seq_323_336/")
    args = parser.parse_args() 
    pred_dir = args.pred_dir
    gt_dir = args.gt_dir

    print(f"pred_dir: {pred_dir}, gt_dir: {gt_dir}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
    lpips = LearnedPerceptualImagePatchSimilarity(normalize=True).to(
        device
    )
    metrics = {"psnr": [], "ssim": [], "lpips": []}
    # load pred_images 
    # breakpoint()
    pred_images_ls = glob.glob(os.path.join(pred_dir, "frame_*.png"))
    # load gt images 
    gt_images_ls = glob.glob(os.path.join(gt_dir, "rgb_*.jpg"))

    for pred_img_path, gt_img_path in zip(pred_images_ls, gt_images_ls):
        pred_img = torch.from_numpy(imageio.imread(pred_img_path)).to(device) / 255
        gt_img = torch.from_numpy(imageio.imread(gt_img_path)).to(device) / 255
        if pred_img.shape != gt_img.shape:
            # resize gt_img to pred_img shape
            # gt_img = torch.nn.functional.interpolate(gt_img.unsqueeze(0), size=pred_img.shape[-2:], mode='bilinear', align_corners=False).squeeze(0)
            gt_img = torch.nn.functional.interpolate(gt_img.unsqueeze(0).unsqueeze(0).float(), size=(pred_img.shape[0], pred_img.shape[1], pred_img.shape[2]), mode='nearest').squeeze()
            # save gt_img 
            imageio.imwrite("temp_gt_img.png", gt_img.cpu().numpy().astype(np.uint8))
        
        if len(pred_img.shape) == 3:
            pred_img = pred_img.unsqueeze(0)
            gt_img = gt_img.unsqueeze(0)
        canvas = torch.cat([pred_img, gt_img], dim=2)
        output_path =os.path.join(os.path.dirname(pred_img_path), "monst3r_render")
        if not os.path.exists(output_path):
            os.makedirs(output_path, exist_ok=True)
        imageio.imwrite(os.path.join(output_path, f"render_{pred_img_path.split('/')[-1].split('.')[0]}.png"), (canvas.squeeze().cpu().numpy()*255).astype(np.uint8))
        pred_img = pred_img.permute(0, 3, 1, 2)  # [1, 3, H, W]
        gt_img = gt_img.permute(0, 3, 1, 2)  # [1, 3, H, W]
        psnr_val = psnr(pred_img, gt_img)
        ssim_val = ssim(pred_img, gt_img)
        lpips_val = lpips(pred_img, gt_img)
        metrics["psnr"].append(psnr_val)
        metrics["ssim"].append(ssim_val)
        metrics["lpips"].append(lpips_val)
        # print(torch.sum(pred_img - gt_img))


    psnr = torch.stack(metrics["psnr"]).mean()
    ssim = torch.stack(metrics["ssim"]).mean()
    lpips = torch.stack(metrics["lpips"]).mean()
    print(f"PSNR: {psnr}, SSIM: {ssim}, LPIPS: {lpips}")
        



    
    
    
