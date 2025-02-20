from lang_sam import LangSAM
from PIL import Image
from glob import glob
import numpy as np
import torch
import os
import argparse
import pickle

def arg_parse():
    parser = argparse.ArgumentParser(description='Semantic segmentation')
    parser.add_argument('--input_dir', type=str, default='/scratch/yy561/monst3r/batch_test_0.07/scene_d78_ego1/images/', help='Input directory')
    parser.add_argument('--output_dir', type=str, default='/scratch/yy561/monst3r/batch_test_results_0.07_puregs/scene_d78_ego1/', help='Output directory')

    return parser.parse_args()

if __name__ == "__main__":
    args = arg_parse()
    semantic_model = LangSAM()
    if "meta" in args.input_dir:   # input_dir is a metadata file
        # find parent dir 
        parent_dir = os.path.dirname(args.input_dir)
        pkl_paths = [os.path.join(parent_dir, f) for f in os.listdir(parent_dir) if f.endswith('.pkl')]
    else:
        pkl_paths = [args.input_dir]
    for pkl_path in pkl_paths:
        print("pkl_path", pkl_path)
        # with open(args.input_dir, 'rb') as f:
        with open(pkl_path, 'rb') as f:
            meta = pickle.load(f)
        # loop every index list of meta
        seq_ind = meta['selected_frames']
        cam_poses = meta['cam_poses']
        dataset_names = meta['dataset']
        deataset_paths = meta['dataset_path']
        image_paths = meta['image_paths']
        dataset_names = meta['dataset'] 
        for i, ind in enumerate(seq_ind):
            input_files = image_paths[i]
            start_idx =ind[0]
            end_idx = ind[-1]
            # args.seq_name = f"seq_{start_idx}_{end_idx}"
            args.input_dir = input_files
            dataset_name = dataset_names[i]
            args.seq_name = f"{dataset_name}_seq_{start_idx}_{end_idx}"
            output_dir = os.path.join(args.output_dir, args.seq_name, "semantics_langsam")
            os.makedirs(output_dir, exist_ok=True)
            output_dir_overall = os.path.join(args.output_dir, args.seq_name, "semantics_langsam_overall")
            os.makedirs(output_dir_overall, exist_ok=True)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            for image_path in input_files:
                # print("image_path",image_path)
                image_pil = Image.open(image_path)
                # masks, boxes, phrases, logits  = semantic_model.predict([image_pil], ["dynamic"])[0]
                # scores, labels, boxes, masks, mask_scores  = semantic_model.predict([image_pil], ["dynamic"])[0]
                results = semantic_model.predict([image_pil], ["dynamic"])

                masks = torch.Tensor(results[0]["masks"])
                if "tum" in image_path:
                    image_name = image_path.split("/")[-1].split(".png")[0]
                else:
                    image_name = image_path.split("/")[-1].split(".")[0]
                
                if masks.shape[1::] != image_pil.size[::-1]:
                    masks = torch.nn.functional.interpolate(masks, size=image_pil.size[::-1], mode="nearest")
                masks_path = f"{output_dir}/{image_name}.npy"
                if masks.shape[0]> 1:
                    masks = masks[0].unsqueeze(0)
                # try:
                mask_return = (~masks.bool()).long().to(device)

                np.save(masks_path, mask_return.cpu().detach().numpy())

                overall_mask = torch.zeros(image_pil.size[::-1]).bool()
                for result in results:
                    masks = torch.Tensor(result["masks"])
                    if masks.shape[1::] != image_pil.size[::-1]:
                        masks = torch.nn.functional.interpolate(masks, size=image_pil.size[::-1], mode="nearest")
                    if masks.shape[0]> 1:
                        masks = masks[0].unsqueeze(0)
                    mask_return = (~masks.bool()).long()
                    overall_mask = overall_mask | mask_return.squeeze(0)

                overall_mask = overall_mask.unsqueeze(0)
                # image_name = image_path.split("/")[-1].split(".")[0]
                masks_path_overall = f"{output_dir_overall}/{image_name}.npy"
                np.save(masks_path_overall, overall_mask.cpu().detach().numpy())
                    

                    # masks = torch.Tensor(results["masks"])
                    # image_name = image_path.split("/")[-1].split(".")[0]
                    
                    # if masks.shape[1::] != image_pil.size[::-1]:
                    #     masks = torch.nn.functional.interpolate(masks, size=image_pil.size[::-1], mode="nearest")
                    # masks_path = f"{output_dir}/{image_name}.npy"
                    # if masks.shape[0]> 1:
                    #     masks = masks[0].unsqueeze(0)
                    # # try:
                    # mask_return = (~masks.bool()).long().to(device)

                    # np.save(masks_path, mask_return.cpu().detach().numpy())