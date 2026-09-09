"""Segment a preprocessed 8-bit 2D image with the v2 existence classifier."""

import argparse
from pathlib import Path

import hydra
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from inference_utils.distributed import inference_worker
from utils import get_padding


@torch.inference_mode()
def predict_image(model, image, prompts, threshold=0.5):
    if not prompts or any(not isinstance(prompt, str) or not prompt.strip() or "[SEP]" in prompt for prompt in prompts):
        raise ValueError("Provide a nonempty list of individual text prompts.")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1.")
    image = np.asarray(image)
    if image.dtype != np.uint8 or image.ndim not in (2, 3):
        raise ValueError("image must be an 8-bit grayscale or RGB array.")
    if image.ndim == 2:
        image = image[None]
    elif image.shape[-1] == 3:
        image = image.transpose(2, 0, 1)
    else:
        raise ValueError("RGB images must have shape [H, W, 3].")
    height, width = image.shape[-2:]
    if min(height, width) < 1:
        raise ValueError("Image dimensions must be positive.")
    padding, padded_size = get_padding(image)
    image = np.pad(image, padding)
    tensor = torch.from_numpy(image).unsqueeze(0).to(model.pixel_mean.device).float()
    tensor = F.interpolate(tensor, size=(512, 512), mode="bicubic", align_corners=False)
    predictions = model({"image": tensor, "text": ["[SEP]".join(prompts)]})["predictions"]
    logits = F.interpolate(
        predictions["pred_gmasks"], size=(padded_size, padded_size),
        mode="bicubic", align_corners=False, antialias=True,
    )[:, 0]
    logits = logits[:, padding[1][0]:padding[1][0] + height, padding[2][0]:padding[2][0] + width]
    existence = predictions["object_existence"].reshape(-1).sigmoid()
    probabilities = logits.sigmoid() * (existence > threshold)[:, None, None]
    return {
        "masks": (probabilities > 0.5).cpu().numpy(),
        "probabilities": probabilities.cpu().numpy(),
        "existence_probabilities": existence.cpu().numpy(),
        "prompts": np.asarray(prompts),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--image", help="Single input image")
    inputs.add_argument("--input_dir", type=Path, help="Folder of PNG, JPEG, BMP, or TIFF images")
    parser.add_argument("--prompt", action="append", required=True)
    parser.add_argument("--output", required=True, type=Path, help="Output .npz file, or output directory with --input_dir")
    parser.add_argument("--checkpoint", required=True, help="Your 2D fine-tuned checkpoint; dedicated pretrained 2D weights will be released later")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    if args.input_dir is not None:
        extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        files = [path for path in args.input_dir.iterdir() if path.is_file() and path.suffix.lower() in extensions]
    else:
        files = [Path(args.image)]
    device, files = inference_worker(files)
    if not files:
        return
    root = Path(__file__).resolve().parent
    with hydra.initialize_config_dir(version_base=None, config_dir=str(root / "configs/model")):
        config = hydra.compose(config_name="biomedparse_2D")
    model = hydra.utils.instantiate(config, _convert_="object")
    model.load_pretrained(args.checkpoint)
    model.to(device).eval()
    for file_path in files:
        with Image.open(file_path) as image:
            if image.mode not in ("L", "RGB", "RGBA", "P"):
                raise ValueError("Preprocess medical intensities into an 8-bit L or RGB image first.")
            image = np.asarray(image.convert("L" if image.mode == "L" else "RGB"))
        results = predict_image(model, image, args.prompt, args.threshold)
        output = args.output / f"{file_path.name}.npz" if args.input_dir is not None else args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("wb") as stream:
            np.savez_compressed(stream, **results)
        for prompt, score in zip(args.prompt, results["existence_probabilities"]):
            print(f"{file_path.name}: {prompt}: existence probability={score:.4f}")
        print(f"Saved {output}")


if __name__ == "__main__":
    main()