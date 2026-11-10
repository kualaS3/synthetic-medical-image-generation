"""Dataset loading and preprocessing.

- ``standardise_brightness`` and ``get_transforms`` are verbatim from the
  notebook (cells 7 and 8). The notebook defined ``standardise_brightness``
  twice (cells 7 and 12, identical); it lives here exactly once.
- ``KaggleDRDataset`` was *used* in the notebook (cells 8 and 31) but its
  class definition was missing from the notebook — the notebook only ran
  because of interactive state. It is reconstructed here from how it is
  used, and from the image-index code in notebook cells 10 and 26:
  read ``trainLabels.csv``, index ``train_*/*.jpeg`` by image name, expose
  ``.labels``, and apply CLAHE brightness standardisation at load time so
  the raw files on disk are never modified.
"""
import os

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

CLAHE_CLIP_LIMIT = 2.0  # Clip limit: lower = subtler, higher = more aggressive
CLAHE_TILE_SIZE = 8  # Local tile size (8x8 is standard for fundus images)


def standardise_brightness(img: Image.Image) -> Image.Image:
    """
    Standardise image brightness using CLAHE on the V channel (HSV).

    CLAHE (Contrast Limited Adaptive Histogram Equalisation) equalises
    brightness locally in small tiles, so the large black border around
    the fundus does not skew the result the way global equalisation would.
    The clip limit prevents noise amplification in flat regions.
    Hue and saturation are unchanged.
    """
    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=(CLAHE_TILE_SIZE, CLAHE_TILE_SIZE)
    )
    img_hsv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2HSV)
    img_hsv[:, :, 2] = clahe.apply(img_hsv[:, :, 2])
    return Image.fromarray(cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB))


def build_image_index(data_dir: str) -> dict:
    """Map image name (without extension) -> path, from ``train_*`` folders."""
    image_paths = {}
    for d in os.listdir(data_dir):
        folder = os.path.join(data_dir, d)
        if d.startswith("train") and os.path.isdir(folder):
            for fname in os.listdir(folder):
                if fname.endswith(".jpeg"):
                    image_paths[fname.replace(".jpeg", "")] = os.path.join(folder, fname)
    return image_paths


class KaggleDRDataset(Dataset):
    """Kaggle Diabetic Retinopathy Detection dataset (EyePacs).

    Expected layout under ``data_dir`` (see README for download link)::

        kaggle_dr/
            train_1/ ... train_5/   # .jpeg fundus images
            trainLabels.csv         # columns: image, level

    ``labels`` is a list of ints (DR grade 0-4), in the same order as the
    indexed images, so callers can do ``np.array(dataset.labels)`` for
    balanced subsampling, exactly as the notebook did.
    """

    def __init__(self, data_dir: str, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = build_image_index(data_dir)
        df = pd.read_csv(os.path.join(data_dir, "trainLabels.csv"))
        # Keep only rows whose image file was actually found.
        df = df[df["image"].isin(self.image_paths)].reset_index(drop=True)
        self.df = df
        self.labels = df["level"].astype(int).tolist()
        self.image_names = df["image"].tolist()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        img = Image.open(self.image_paths[self.image_names[idx]]).convert("RGB")
        img = standardise_brightness(img)
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label


def get_transforms(img_size):
    train_tf = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    eval_tf = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    return train_tf, eval_tf


class SyntheticAugmentedDataset(Dataset):
    """
    Wraps a real dataset and appends synthetic images from disk.
    The synthetic images are saved as PNG files from the generation step.
    """

    def __init__(self, real_dataset, synthetic_dirs, transform=None):
        self.real_dataset = real_dataset
        self.synthetic = []
        for grade, syn_dir in synthetic_dirs.items():
            for fname in os.listdir(syn_dir):
                self.synthetic.append((os.path.join(syn_dir, fname), int(grade)))
        self.transform = transform
        print(f"Dataset: {len(real_dataset)} real + {len(self.synthetic)} synthetic images")

    def __len__(self):
        return len(self.real_dataset) + len(self.synthetic)

    def __getitem__(self, idx):
        if idx < len(self.real_dataset):
            return self.real_dataset[idx]
        img_path, label = self.synthetic[idx - len(self.real_dataset)]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label
