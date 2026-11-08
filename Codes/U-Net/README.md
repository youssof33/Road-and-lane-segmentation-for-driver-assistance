# U-Net — KITTI Road Segmentation

Classic U-Net for binary road segmentation on the KITTI Road dataset.

## Files

| File | Purpose |
| --- | --- |
| `model.py` | U-Net architecture |
| `dataset.py` | KITTI Road loader (image + road mask) |
| `train.py` | Training / validation loop |
| `predict.py` | Inference and mask overlay |
| `requirements.txt` | Python dependencies |

## Dataset

The loader expects the standard KITTI `data_road` layout:

```
data_road/
  training/
    image_2/       # RGB images
    gt_image_2/    # <category>_road_<index>.png ground truth
  testing/
    image_2/
```

KITTI ground truth is stored as bit-planes: the **R** channel marks the valid
evaluation area, the **B** channel marks the road. Pixels outside the valid
area are ignored by the loss.

By default `--data-root` points to `datasets/KITTI/data_road` in this
repository, so no argument is needed when the dataset is in place.

## Install

```bash
pip install -r requirements.txt
```

## Train

Run from this folder:

```bash
python train.py --epochs 20 --batch-size 4 --lr 1e-3
```

Useful options:

- `--img-size 256 768` — training resolution (height width). Must be divisible by 16.
- `--val-fraction 0.15` — fraction of the training set used for validation.
- `--out-dir runs/unet` — where `best_model.pth` and `last_model.pth` are saved.

## Predict

```bash
python predict.py --checkpoint runs/unet/best_model.pth --input path/to/image.png
```

`--input` also accepts a folder. For every image the script writes:

- `<name>_mask.png` — colour mask (green = road)
- `<name>_overlay.png` — mask blended over the original image

## Notes

- The model outputs two classes: `0 = background`, `1 = road`. Change
  `NUM_CLASSES` in `dataset.py` and the model constructor to add more classes.
- KITTI road ground truth does not contain separate lane-marking labels. For
  lane markings, use BDD100K lane annotations and extend `dataset.py`.
