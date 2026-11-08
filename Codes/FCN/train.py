"""Train FCN-8s on the KITTI Road dataset.

Example (run from this folder):
    python train.py --epochs 20 --batch-size 4 --lr 1e-3
"""

import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from dataset import IGNORE_INDEX, NUM_CLASSES, KITTIRoadDataset
from model import FCN8s

DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[2] / "datasets" / "KITTI" / "data_road"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=str,
        default=str(DEFAULT_DATA_ROOT),
        help="Path to the KITTI data_road folder (contains training/ and testing/).",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--img-size",
        type=int,
        nargs=2,
        default=[256, 768],
        metavar=("H", "W"),
        help="Training resolution (must be divisible by 32 for FCN-8s).",
    )
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "runs" / "fcn8s"),
    )
    parser.add_argument(
        "--pretrained-backbone",
        action="store_true",
        help="Start from ImageNet VGG16 weights (downloads ~500 MB).",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
    return running_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device, num_classes=NUM_CLASSES):
    model.eval()
    running_loss = 0.0
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.int64)

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        logits = model(images)
        running_loss += criterion(logits, masks).item() * images.size(0)

        pred = logits.argmax(dim=1)
        valid = masks != IGNORE_INDEX
        indices = masks[valid] * num_classes + pred[valid]
        confusion += torch.bincount(
            indices.cpu(), minlength=num_classes * num_classes
        ).reshape(num_classes, num_classes)

    tp = confusion.diag()
    fp = confusion.sum(dim=0) - tp
    fn = confusion.sum(dim=1) - tp
    iou = tp / (tp + fp + fn).clamp(min=1)
    present = (tp + fn) > 0  # only score classes that appear in the targets
    miou = iou[present].mean().item() if present.any() else 0.0
    accuracy = tp.sum().item() / max(confusion.sum().item(), 1)
    return running_loss / len(loader.dataset), accuracy, miou


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    img_size = tuple(args.img_size)

    # Two dataset instances so augmentation is applied only to the training split.
    base_set = KITTIRoadDataset(args.data_root, "training", img_size, augment=False)
    train_set = KITTIRoadDataset(args.data_root, "training", img_size, augment=True)

    generator = torch.Generator().manual_seed(args.seed)
    indices = torch.randperm(len(base_set), generator=generator).tolist()
    val_len = max(1, int(len(base_set) * args.val_fraction))
    val_indices = indices[:val_len]
    train_indices = indices[val_len:]

    train_subset = Subset(train_set, train_indices)
    val_subset = Subset(base_set, val_indices)

    train_loader = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = FCN8s(
        num_classes=NUM_CLASSES,
        pretrained_backbone=args.pretrained_backbone,
    ).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    print(f"Device: {device}")
    print(f"Train images: {len(train_subset)} | Val images: {len(val_subset)}")
    print(f"Image size: {img_size[0]}x{img_size[1]}")

    best_miou = -1.0
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_miou = evaluate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train loss {train_loss:.4f} | "
            f"val loss {val_loss:.4f} | "
            f"val acc {val_acc * 100:.2f}% | "
            f"val mIoU {val_miou * 100:.2f}% | "
            f"{time.time() - start:.1f}s"
        )

        checkpoint = {
            "model": model.state_dict(),
            "num_classes": NUM_CLASSES,
            "img_size": img_size,
            "args": vars(args),
        }
        torch.save(checkpoint, out_dir / "last_model.pth")
        if val_miou > best_miou:
            best_miou = val_miou
            torch.save(checkpoint, out_dir / "best_model.pth")

    print(f"Best val mIoU: {best_miou * 100:.2f}%")
    print(f"Checkpoint: {out_dir / 'best_model.pth'}")


if __name__ == "__main__":
    main()
