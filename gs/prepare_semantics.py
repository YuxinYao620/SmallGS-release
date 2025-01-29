from lang_sam import LangSAM
from PIL import Image
from glob import glob
import numpy as np
import torch
import os

semantic_model = LangSAM()
# breakpoint()

data_root_root = "/scratch/yy561/monst3r/batch_test_0.07/scene_d78_ego1/images/"
# result_root = "/scratch/yy561/monst3r/batch_test_results_0.07/scene_d78_ego1/seq_195_224_gs"
# result_root = data_root.replace("batch_test_0.07", "batch_test_results_0.07")
# result_root = result_root.replace("/images", "")
import pickle
dataset_name = "scene_d78_ego1"
# with open('/scratch/yy561/pointodyssey/point_odyssey/cam_poses_{}.pkl'.format(dataset_name), 'rb') as f:
#     save = pickle.load(f)
os.makedirs("/scratch/yy561/monst3r/batch_test_results_0.07_puregs/scene_d78_ego1/", exist_ok=True)
for result_root in os.listdir("/scratch/yy561/monst3r/batch_test_results_0.07_gs/scene_d78_ego1/"):
    seq_name = result_root.split("/")[-1].split("_gs")[0]
    data_root = os.path.join(data_root_root, seq_name)
    print('data_root', data_root)
    
    # data_dir = data_root+"images"
    data_dir = os.path.join(data_root)
    # output_dir = data_root+"semantics"

        # os.mkdir(os.path.join("/scratch/yy561/monst3r/batch_test_results_0.07_newgs/scene_d78_ego1/", result_root), exist_ok=True)

    output_dir = os.path.join("/scratch/yy561/monst3r/batch_test_results_0.07_puregs/scene_d78_ego1/", result_root, "semantics_langsam")
    os.makedirs(output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    for image_path in sorted(glob(f"{data_dir}/*.jpg")):
        # print("image_path",image_path)
        image_pil = Image.open(image_path)
        # masks, boxes, phrases, logits  = semantic_model.predict([image_pil], ["dynamic"])[0]
        # scores, labels, boxes, masks, mask_scores  = semantic_model.predict([image_pil], ["dynamic"])[0]
        results = semantic_model.predict([image_pil], ["dynamic"])[0]
        masks = torch.Tensor(results["masks"])
        image_name = image_path.split("/")[-1].split(".")[0]
        
        if masks.shape[1::] != image_pil.size[::-1]:
            masks = torch.nn.functional.interpolate(masks, size=image_pil.size[::-1], mode="nearest")
        masks_path = f"{output_dir}/{image_name}.npy"
        if masks.shape[0]> 1:
            masks = masks[0].unsqueeze(0)
        # try:
        mask_return = (~masks.bool()).long().to(device)

        np.save(masks_path, mask_return.cpu().detach().numpy())

def get_semantic_mask(image_path, output_dir, device):
    semantic_model = LangSAM()
    image_pil = Image.open(image_path)
    masks, boxes, phrases, logits = semantic_model.predict(image_pil, "dynamic")
    image_name = image_path.split("/")[-1].split(".")[0]
    
    if masks.shape[1::] != image_pil.size[::-1]:
        masks = torch.nn.functional.interpolate(masks, size=image_pil.size[::-1], mode="nearest")
    masks_path = f"{output_dir}/{image_name}.npy"

    if masks.shape[0]> 1:
        masks = masks[0].unsqueeze(0)
    # try:
    mask_return = (~masks).long().to(device)

    np.save(masks_path, mask_return.cpu().detach().numpy())
    return mask_return.cpu().detach().numpy()
