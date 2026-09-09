import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import hydra
from hydra import compose
import gc
from pathlib import Path
from inference_utils.distributed import inference_worker
from utils import process_input, process_output, slice_nms

def load_case(file_path):
    data = np.load(file_path, allow_pickle=True)
    image = data["imgs"]
    text_prompts = data["text_prompts"].item()
    gt = data["gts"] if "gts" in data else None
    return image, text_prompts, gt


def merge_multiclass_masks(masks, ids):
    bg_mask = 0.5 * torch.ones_like(masks[0:1])
    keep_masks = torch.cat([bg_mask, masks], dim=0)
    class_mask = keep_masks.argmax(dim=0)

    id_map = {j + 1: int(ids[j]) for j in range(len(ids)) if j + 1 != int(ids[j])}
    if len(id_map) > 0:
        orig_mask = class_mask.clone()
        for j in id_map:
            class_mask[orig_mask == j] = id_map[j]

    return class_mask


def postprocess(model_outputs, object_existence, threshold=0.5, do_nms=True):
    if do_nms and model_outputs.shape[0] > 1:
        # do non-max suppression for each slice
        return slice_nms(model_outputs.sigmoid(), object_existence.sigmoid(), 
                                        iou_threshold=0.5, score_threshold=threshold)
    mask = (model_outputs.sigmoid()) * (
        object_existence.sigmoid() > threshold
    ).int().unsqueeze(-1).unsqueeze(-1)
    return mask


def compute_dice_coefficient(mask_gt, mask_pred):
    volume_sum = mask_gt.sum() + mask_pred.sum()
    if volume_sum == 0:
        return np.NaN
    volume_intersect = (mask_gt & mask_pred).sum()
    return 2 * volume_intersect / volume_sum


def print_memory_info(stage=""):
    print(
        f"[{stage}] GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB"
    )
    print(
        f"[{stage}] GPU memory reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB"
    )


def main(args):
    device, files = inference_worker(Path(args.input_dir).glob("*.npz"))
    if not files:
        return
    print("Using device:", device)

    if args.slice_batch_size < 1:
        raise ValueError("slice_batch_size must be positive.")
    config_dir = Path(__file__).resolve().parent / "configs/model"
    with hydra.initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name="biomedparse_3D")
    model = hydra.utils.instantiate(cfg, _convert_="object")
    checkpoint = args.checkpoint
    if checkpoint is None:
        from huggingface_hub import hf_hub_download
        checkpoint = hf_hub_download("microsoft/BiomedParse", "biomedparse_v2.ckpt")
    model.load_pretrained(checkpoint)
    model.to(device)
    model.eval()

    os.makedirs(args.output_dir, exist_ok=True)

    for file_path in files:
        print(f"\nProcessing: {file_path}")

        npz_data = np.load(file_path, allow_pickle=True)
        imgs = npz_data["imgs"]
        text_prompts = npz_data["text_prompts"].item()

        ids = [int(_) for _ in text_prompts.keys() if _ != "instance_label"]
        ids.sort()
        text = "[SEP]".join([text_prompts[str(i)] for i in ids])
        
        imgs, pad_width, padded_size, valid_axis = process_input(imgs, 512)

        imgs = imgs.to(device).int()

        # print_memory_info("Before model")

        input_tensor = {
            "image": imgs.unsqueeze(0),  # Add batch dimension
            "text": [text],
        }

        with torch.no_grad():
            output = model(input_tensor, mode="eval", slice_batch_size=args.slice_batch_size)

        mask_preds = output["predictions"]["pred_gmasks"]
        mask_preds = F.interpolate(
            mask_preds,
            size=(512, 512),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )

        mask_preds = postprocess(mask_preds, output["predictions"]["object_existence"])
        mask_preds = merge_multiclass_masks(mask_preds, ids)
        mask_preds = process_output(mask_preds, pad_width, padded_size, valid_axis)

        save_path = os.path.join(args.output_dir, file_path.name)
        np.savez_compressed(save_path, segs=mask_preds)

        # Cleanup
        del imgs, input_tensor, output, mask_preds
        gc.collect()
        torch.cuda.empty_cache()

        # print_memory_info("After cleanup")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--checkpoint", help="Local weights; otherwise download biomedparse_v2.ckpt from Hugging Face.")
    parser.add_argument("--slice_batch_size", type=int, default=4)
    args = parser.parse_args()
    main(args)
