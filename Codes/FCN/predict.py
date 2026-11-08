"""Run a trained FCN-8s checkpoint on an image or a folder of images.

Example:
    python predict.py --checkpoint runs/fcn8s/best_model.pth --input some_image.png
    python predict.py --checkpoint runs/fcn8s/best_model.pth --input ../../datasets/KITTI/data_road/testing/image_2
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image

from dataset import IMAGENET_MEAN, IMAGENET_STD
from model import FCN8s

# 0 = background (black), 1 = road (green), 2 = optional extra class (red)
PALETTE = np.array([[0, 0, 0], [0, 255, 0], [255, 0, 0]], dtype=np.uint8)
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--input", type=str, required=True, help="Image file or folder.")
    parser.add_argument(
        "--output",
        type=str,
        default=str(Path(__file__).resolve().parent / "predictions"),
    )
    parser.add_argument(
        "--img-size",
        type=int,
        nargs=2,
        default=None,
        metavar=("H", "W"),
        help="Override the resolution stored in the checkpoint.",
    )
    parser.add_argument("--alpha", type=float, default=0.5, help="Overlay opacity.")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    num_classes = checkpoint.get("num_classes", 2)
    model = FCN8s(num_classes=num_classes).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint.get("img_size", (256, 768))


@torch.no_grad()
def segment(model, image, img_size, device):
    """Return an (H, W) uint8 label map at the original image resolution."""
    original_size = (image.height, image.width)
    resized = TF.resize(
        image, tuple(img_size), interpolation=TF.InterpolationMode.BILINEAR
    )
    tensor = TF.normalize(TF.to_tensor(resized), IMAGENET_MEAN, IMAGENET_STD)
    tensor = tensor.unsqueeze(0).to(device)

    logits = model(tensor)
    logits = F.interpolate(
        logits, size=original_size, mode="bilinear", align_corners=False
    )
    return logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)


def main():
    args = parse_args()
    device = torch.device(args.device)
    model, checkpoint_img_size = load_model(args.checkpoint, device)
    img_size = tuple(args.img_size) if args.img_size else tuple(checkpoint_img_size)

    input_path = Path(args.input)
    if input_path.is_dir():
        files = sorted(
            p for p in input_path.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
        )
    else:
        files = [input_path]

    if not files:
        raise SystemExit(f"No images found at {input_path}")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    for path in files:
        image = Image.open(path).convert("RGB")
        mask = segment(model, image, img_size, device)

        color_mask = PALETTE[mask]
        original = np.asarray(image).astype(np.float32)
        overlay = original * (1.0 - args.alpha) + color_mask.astype(np.float32) * args.alpha
        overlay = overlay.clip(0, 255).astype(np.uint8)

        Image.fromarray(color_mask).save(output_dir / f"{path.stem}_mask.png")
        Image.fromarray(overlay).save(output_dir / f"{path.stem}_overlay.png")
        print(f"{path.name} -> {output_dir / (path.stem + '_overlay.png')}")

    print(f"Saved {len(files)} prediction(s) to {output_dir}")


if __name__ == "__main__":
    main()
