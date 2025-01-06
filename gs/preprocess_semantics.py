from lang_sam import LangSAM
from PIL import Image
from glob import glob
import numpy as np
import torch
import os

def get_semantics(data_root, output_dir):

    semantic_model = LangSAM()


    # data_dir = data_root+"images"
    data_dir = os.path.join(data_root)
    # output_dir = data_root+"semantics"
    output_dir = os.path.join(output_dir, "semantics")
    os.makedirs(output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    for image_path in sorted(glob(f"{data_dir}/*.jpg")):
        # print("image_path",image_path)
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