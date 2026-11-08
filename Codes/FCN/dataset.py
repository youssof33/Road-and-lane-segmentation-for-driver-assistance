"""KITTI Road dataset for binary road segmentation.

KITTI ground-truth PNGs are stored as three independent bit-planes
(see devkit_road/readme.txt):

    R channel > 0  -> valid evaluation area
    G channel      -> unused
    B channel > 0  -> road area

Pixels outside the valid area are ignored during training (target = 255).

Expected layout:

    <root>/
        training/
            image_2/       um_000000.png, umm_000000.png, uu_000000.png, ...
            gt_image_2/    um_road_000000.png, umm_road_000000.png, ...
        testing/
            image_2/       ...
"""

import random
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset

IGNORE_INDEX = 255
NUM_CLASSES = 2  # 0 = background, 1 = road

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def gt_path_for(image_path: Path) -> Path:
    """Map an image name to its road ground truth: um_000000 -> um_road_000000."""
    category, index = image_path.stem.split("_")
    return image_path.parent.parent / "gt_image_2" / f"{category}_road_{index}.png"


class KITTIRoadDataset(Dataset):
    def __init__(self, root, split="training", img_size=(256, 768), augment=False):
        self.root = Path(root)
        self.image_dir = self.root / split / "image_2"
        self.img_size = tuple(img_size)  # (height, width)
        self.augment = augment

        self.images = sorted(p.name for p in self.image_dir.glob("*.png"))
        if not self.images:
            raise FileNotFoundError(
                f"No images found in {self.image_dir}. "
                "Check --data-root (it must point to the KITTI data_road folder)."
            )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = self.image_dir / self.images[index]
        gt_file = gt_path_for(image_path)
        if not gt_file.exists():
            raise FileNotFoundError(f"Missing ground truth for {image_path}: {gt_file}")

        image = Image.open(image_path).convert("RGB")
        gt = np.asarray(Image.open(gt_file).convert("RGB"))

        road = (gt[:, :, 2] > 0).astype(np.uint8)   # B plane = road
        valid = gt[:, :, 0] > 0                      # R plane = valid area
        mask = np.where(valid, road, IGNORE_INDEX).astype(np.uint8)

        # Resize image (bilinear) and mask (nearest so labels stay intact).
        image = TF.resize(
            image, self.img_size, interpolation=TF.InterpolationMode.BILINEAR
        )
        mask = np.asarray(
            TF.resize(
                Image.fromarray(mask),
                self.img_size,
                interpolation=TF.InterpolationMode.NEAREST,
            )
        )

        if self.augment:
            if random.random() < 0.5:
                image = TF.hflip(image)
                mask = np.fliplr(mask).copy()
            if random.random() < 0.5:
                image = TF.adjust_brightness(image, random.uniform(0.8, 1.2))
                image = TF.adjust_contrast(image, random.uniform(0.8, 1.2))

        image = TF.normalize(TF.to_tensor(image), IMAGENET_MEAN, IMAGENET_STD)
        return image, torch.from_numpy(mask.copy()).long()
