"""Smoke tests. No Kaggle data needed, no training run.

Builds a tiny fake dataset (a few JPEGs + trainLabels.csv) in a temp
directory and checks that the split code actually works: the dataset
loads, the Generator/Discriminator run a forward pass, and one GAN
training step runs on CPU.
"""
import os
import sys

import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import CONFIG
from src.dataset import KaggleDRDataset, get_transforms
from src.models import Discriminator, Generator, weights_init


def _make_fake_data(tmp_path):
    data_dir = tmp_path / "kaggle_dr"
    (data_dir / "train_1").mkdir(parents=True)
    rows = []
    for i, grade in enumerate([0, 1, 2, 3, 4, 3]):
        name = f"img{i}"
        img = Image.fromarray(
            np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        )
        img.save(data_dir / "train_1" / f"{name}.jpeg")
        rows.append({"image": name, "level": grade})
    pd.DataFrame(rows).to_csv(data_dir / "trainLabels.csv", index=False)
    return str(data_dir)


def test_dataset_loads(tmp_path):
    data_dir = _make_fake_data(tmp_path)
    train_tf, _ = get_transforms(128)
    ds = KaggleDRDataset(data_dir, transform=train_tf)
    assert len(ds) == 6
    assert ds.labels == [0, 1, 2, 3, 4, 3]
    img, label = ds[0]
    assert img.shape == (3, 128, 128)
    assert label == 0


def test_generator_discriminator_forward():
    device = torch.device("cpu")
    G = Generator(64, 5, 128).to(device)
    D = Discriminator(5, 128).to(device)
    G.apply(weights_init)
    D.apply(weights_init)
    z = torch.randn(2, 64)
    labels = torch.tensor([3, 4])
    fake = G(z, labels)
    assert fake.shape == (2, 3, 128, 128)
    assert fake.min() >= -1.0 and fake.max() <= 1.0
    scores = D(fake, labels)
    assert scores.shape == (2,)
    assert torch.all((scores >= 0) & (scores <= 1))


def test_one_gan_step(tmp_path):
    from torch.utils.data import DataLoader, Subset

    from src.train import train_gan

    data_dir = _make_fake_data(tmp_path)
    train_tf, _ = get_transforms(128)
    ds = KaggleDRDataset(data_dir, transform=train_tf)
    loader = DataLoader(Subset(ds, list(range(6))), batch_size=2, shuffle=True)
    config = dict(CONFIG)
    config["epochs"] = 1
    config["output_dir"] = str(tmp_path / "outputs")
    os.makedirs(os.path.join(config["output_dir"], "checkpoints"), exist_ok=True)
    device = torch.device("cpu")
    G = Generator(64, 5, 128).to(device)
    D = Discriminator(5, 128).to(device)
    G.apply(weights_init)
    D.apply(weights_init)
    opt_G = torch.optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))
    g_losses, d_losses = train_gan(
        config, device, loader, G, D, opt_G, opt_D, torch.nn.BCELoss()
    )
    assert len(g_losses) == 1 and len(d_losses) == 1
