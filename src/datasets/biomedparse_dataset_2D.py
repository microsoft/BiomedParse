"""Binary-mask annotations with candidate prompts and automatic negative sampling."""

import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset


def candidate_prompts(annotation):
    sentences = annotation.get("sentences", [])
    prompts = [sentence.get("raw", sentence.get("sent")) for sentence in sentences]
    if not prompts or any(
        not isinstance(prompt, str) or not prompt.strip() or "[SEP]" in prompt
        for prompt in prompts
    ):
        raise ValueError("Each binary-mask annotation needs nonempty sentences with raw or sent text.")
    return prompts


def normalize_name(text):
    return " ".join(text.split()).casefold()


def class_info(mask_file):
    parts = Path(mask_file).stem.rsplit("_", 3)
    if len(parts) != 4 or any(not part.strip() for part in parts):
        return None
    return tuple(normalize_name(part.replace("+", " ")) for part in parts[-3:])


def read_mask(path, shape):
    with Image.open(path) as source:
        labels = np.asarray(source.convert("RGB") if source.mode in ("RGB", "RGBA", "P") else source)
        labels = np.any(labels > 0, axis=-1) if labels.ndim == 3 else labels > 0
    if labels.ndim != 2 or labels.shape != shape:
        raise ValueError("Binary masks must have the same size as the image.")
    return labels


def sample_crop(mask, min_crop_ratio, generator, foreground=False):
    """Sample an empty or foreground-containing square in the original image."""
    height, width = mask.shape
    minimum = max(1, int(min(height, width) * min_crop_ratio))
    maximum = min(height, width)
    integral = np.pad(mask.astype(np.int64), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    first_size = int(torch.randint(minimum, maximum + 1, (), generator=generator))
    for size in dict.fromkeys((first_size, minimum)):
        sums = integral[size:, size:] - integral[:-size, size:] - integral[size:, :-size] + integral[:-size, :-size]
        positions = np.argwhere(sums > 0 if foreground else sums == 0)
        if len(positions):
            top, left = positions[int(torch.randint(len(positions), (), generator=generator))]
            return slice(top, top + size), slice(left, left + size)
    return None


class BiomedParseDataset2D(Dataset):
    def __init__(
        self, root_dir, split="train", num_prompts=4, image_size=512,
        negative_ratio=0.15, min_crop_ratio=0.5,
        exhaustive_annotations=True, sampling_seed=None,
    ):
        if num_prompts < 1 or image_size < 1:
            raise ValueError("num_prompts and image_size must be positive.")
        if not 0 <= negative_ratio <= 1 or not 0 < min_crop_ratio <= 1:
            raise ValueError("negative_ratio must be in [0, 1] and min_crop_ratio in (0, 1].")
        self.root = Path(root_dir)
        self.split = split
        self.num_prompts = num_prompts
        self.negative_ratio = negative_ratio if split.startswith("train") else 0
        self.min_crop_ratio = min_crop_ratio
        self.exhaustive_annotations = exhaustive_annotations
        self.image_size = image_size
        self.sampling_seed = sampling_seed
        with (self.root / f"{split}.json").open() as stream:
            data = json.load(stream)
        self.annotations = data["annotations"]
        images = {entry["id"]: entry["file_name"] for entry in data.get("images", [])}
        self.filenames = []
        self.prompts = []
        self.image_classes = {}
        self.image_annotations = {}
        self.class_prompts = {}
        for index, annotation in enumerate(self.annotations):
            filename = annotation.get("file_name", images.get(annotation.get("image_id")))
            if not filename:
                raise ValueError("Each annotation needs file_name or an image_id in images.")
            if annotation.get("image_id") in images and images[annotation["image_id"]] != filename:
                raise ValueError("Annotation file_name conflicts with its images entry.")
            prompts = candidate_prompts(annotation)
            self.filenames.append(filename)
            self.prompts.append(prompts)
            self.image_annotations.setdefault(filename, []).append(index)
            info = class_info(annotation["mask_file"])
            self.image_classes.setdefault(filename, set()).add(info)
            if info is not None:
                self.class_prompts.setdefault(info, []).extend(prompts)
        self.class_aliases = {
            info: {info[-1], *(normalize_name(prompt) for prompt in prompts)}
            for info, prompts in self.class_prompts.items()
        }
        self.alternative_classes = []
        alternatives = self.root / "alt_classes.json"
        if alternatives.exists():
            with alternatives.open() as stream:
                self.alternative_classes = json.load(stream).get("negative", [])
            if not isinstance(self.alternative_classes, list) or any(
                not isinstance(label, str) or not label.strip() or "[SEP]" in label for label in self.alternative_classes
            ):
                raise ValueError("alt_classes.json negative must be a list of verified-absent class names.")
            known_labels = {normalize_name(prompt) for prompts in self.prompts for prompt in prompts}
            known_labels.update(info[-1] for info in self.class_prompts)
            if any(normalize_name(label) in known_labels for label in self.alternative_classes):
                raise ValueError("Alternative negative classes conflict with annotated classes.")

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        annotation = self.annotations[index]
        filename = self.filenames[index]
        with Image.open(self.root / self.split / filename) as source:
            if source.mode not in ("L", "RGB", "RGBA", "P"):
                raise ValueError("Preprocess images to 8-bit grayscale or RGB before training.")
            image = np.asarray(source.convert("RGB")).copy()
        mask_path = self.root / f"{self.split}_mask" / annotation["mask_file"]
        labels = read_mask(mask_path, image.shape[:2])
        generator = None if self.sampling_seed is None else torch.Generator().manual_seed(self.sampling_seed + index)
        choices = self.prompts[index]
        if torch.rand((), generator=generator).item() < self.negative_ratio:
            negative_choices = []
            present_aliases = {
                normalize_name(prompt)
                for other_index in self.image_annotations[filename]
                for prompt in self.prompts[other_index]
            }
            for info in self.image_classes[filename]:
                present_aliases.update(self.class_aliases.get(info, ()))
            info = class_info(annotation["mask_file"])
            if self.exhaustive_annotations and info is not None and None not in self.image_classes[filename]:
                for other, prompts in self.class_prompts.items():
                    if (
                        other[:2] == info[:2]
                        and other not in self.image_classes[filename]
                        and self.class_aliases[other].isdisjoint(present_aliases)
                    ):
                        negative_choices.extend(prompts)
            negative_choices.extend(self.alternative_classes)
            negative_choices = list(dict.fromkeys(
                prompt for prompt in negative_choices if normalize_name(prompt) not in present_aliases
            ))
            if negative_choices:
                choices = [negative_choices[int(torch.randint(len(negative_choices), (), generator=generator))]]
                labels = np.zeros_like(labels)
            else:
                foreground = labels.copy()
                for other_index in self.image_annotations[filename]:
                    if other_index != index:
                        other_path = self.root / f"{self.split}_mask" / self.annotations[other_index]["mask_file"]
                        foreground |= read_mask(other_path, image.shape[:2])
                region = sample_crop(foreground, self.min_crop_ratio, generator)
                if region is None:
                    region = sample_crop(labels, self.min_crop_ratio, generator, foreground=True)
                if region is not None:
                    image, labels = image[region], labels[region]
        if len(choices) == 1:
            prompts = choices * self.num_prompts
        else:
            prompts = [
                choices[int(torch.randint(len(choices), (), generator=generator))]
                for _ in range(self.num_prompts)
            ]

        image = torch.from_numpy(image.copy()).float().permute(2, 0, 1)
        masks = torch.from_numpy(labels.copy()).float().unsqueeze(0)
        height, width = labels.shape
        side = max(height, width)
        top, left = (side - height) // 2, (side - width) // 2
        padding = (left, side - width - left, top, side - height - top)
        image = F.interpolate(F.pad(image, padding).unsqueeze(0), size=(self.image_size, self.image_size), mode="bicubic", align_corners=False)[0]
        padded_masks = F.pad(masks, padding).unsqueeze(0)
        masks = F.interpolate(padded_masks, size=(self.image_size, self.image_size), mode="nearest")[0]
        if labels.any() and not masks.any():
            masks = F.adaptive_max_pool2d(padded_masks, (self.image_size, self.image_size))[0]
        return {
            "image": image,
            "labels": masks.expand(self.num_prompts, -1, -1).long(),
            "text": "[SEP]".join(prompts),
            "class_ids": "&".join(["0"] * self.num_prompts),
            "mask_file": str(mask_path),
            "instance_label": False,
            "multiclass_label": False,
        }