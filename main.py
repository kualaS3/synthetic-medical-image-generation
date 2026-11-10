"""End-to-end pipeline, mirroring the original notebook section by section.

Usage (from the repo root, after downloading the Kaggle data — see README)::

    pip install -r requirements.txt
    python main.py --data-dir ./kaggle_dr
    python main.py --data-dir ./kaggle_dr --epochs 5 --skip-fid --skip-classifier

This is a CPU-friendly run by default only in the sense that it will run on
CPU; the full 25-epoch GAN + classifier run is slow without a GPU, exactly
as it was in the notebook (which ran on CPU, torch 2.11.0+cpu).
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from src.config import CONFIG, SEED, device, ensure_output_dirs, set_seed
from src.dataset import KaggleDRDataset, SyntheticAugmentedDataset, get_transforms
from src.evaluate import compute_fid, print_fid_summary
from src.generate import generate_synthetic_images
from src.models import Discriminator, Generator, weights_init
from src.train import train_classifier, train_gan


def build_gan_dataloader(config):
    """Balanced subsample (300/class where available), as in the notebook."""
    train_tf, _eval_tf = get_transforms(config["img_size"])
    dataset = KaggleDRDataset(config["data_dir"], transform=train_tf)

    np.random.seed(SEED)
    labels = np.array(dataset.labels)
    indices = []
    samples_per_class = 300
    for grade in range(config["n_classes"]):
        grade_indices = np.where(labels == grade)[0]
        n = min(samples_per_class, len(grade_indices))
        selected = np.random.choice(grade_indices, n, replace=False)
        indices.extend(selected)
    indices = np.array(indices)
    np.random.shuffle(indices)
    dataset = Subset(dataset, indices)
    print(f"Using {len(dataset)} images ({samples_per_class} per class where available)")

    # FIX vs notebook: the notebook hard-coded batch_size=4 here while
    # CONFIG said 10. The .py version uses CONFIG["batch_size"].
    dataloader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=True,
    )
    print(f"Total images: {len(dataset)}")
    return dataloader


def plot_losses(g_losses, d_losses, config):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
    ax1.plot(g_losses, color="#E63946", linewidth=1.5, label="Generator Loss")
    ax1.plot(d_losses, color="#457B9D", linewidth=1.5, label="Discriminator Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("GAN Training Losses")
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax1.spines[["top", "right"]].set_visible(False)
    window = min(10, len(g_losses))
    g_smooth = np.convolve(g_losses, np.ones(window) / window, mode="valid")
    d_smooth = np.convolve(d_losses, np.ones(window) / window, mode="valid")
    ax2.plot(g_smooth, color="#E63946", linewidth=2, label="G (smoothed)")
    ax2.plot(d_smooth, color="#457B9D", linewidth=2, label="D (smoothed)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.set_title(f"Smoothed Losses (window={window})")
    ax2.legend()
    ax2.grid(alpha=0.3)
    ax2.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(config["output_dir"], "training_losses.png"), dpi=150)
    plt.close(fig)


def run_classifier(config):
    """Downstream classifier comparison (notebook section 8, cells 31-32)."""
    from src.dataset import get_transforms as _get_transforms

    _train_tf, eval_tf = _get_transforms(config["img_size"])
    full_ds = KaggleDRDataset(config["data_dir"], transform=eval_tf)
    all_labels = np.array(full_ds.labels)

    np.random.seed(SEED)
    selected_indices = []
    for grade in range(config["n_classes"]):
        grade_idx = np.where(all_labels == grade)[0]
        n = min(200, len(grade_idx))
        selected_indices.extend(np.random.choice(grade_idx, n, replace=False))
    selected_indices = np.array(selected_indices)
    np.random.shuffle(selected_indices)
    balanced_ds = Subset(full_ds, selected_indices)
    total = len(balanced_ds)
    n_val = int(0.2 * total)
    n_train = total - n_val
    print(f"Balanced subset: {total} images (200/class), train={n_train}, val={n_val}")
    train_ds, val_ds = torch.utils.data.random_split(
        balanced_ds, [n_train, n_val], generator=torch.Generator().manual_seed(SEED)
    )
    val_loader = DataLoader(
        val_ds, batch_size=32, shuffle=False, num_workers=config["num_workers"]
    )
    train_loader_real = DataLoader(
        train_ds, batch_size=32, shuffle=True, num_workers=config["num_workers"]
    )
    acc_real, hist_real = train_classifier(
        "Real only", train_loader_real, val_loader, device, epochs=15
    )
    synthetic_dirs = {
        g: os.path.join(config["output_dir"], "synthetic", f"grade_{g}")
        for g in config["target_grades"]
        if os.path.exists(os.path.join(config["output_dir"], "synthetic", f"grade_{g}"))
    }
    if not synthetic_dirs:
        print("No synthetic images found — generate them first (Section 6).")
        return acc_real, None
    aug_ds = SyntheticAugmentedDataset(train_ds, synthetic_dirs, transform=eval_tf)
    train_loader_aug = DataLoader(
        aug_ds, batch_size=32, shuffle=True, num_workers=config["num_workers"]
    )
    acc_aug, hist_aug = train_classifier(
        "Real+Synthetic", train_loader_aug, val_loader, device, epochs=15
    )
    delta = acc_aug - acc_real
    direction = "improvement" if delta > 0 else "degradation"
    print(f"\n{'=' * 45}")
    print(f"  Real only          best val acc: {acc_real:.4f} ({acc_real * 100:.1f}%)")
    print(f"  Real + Synthetic   best val acc: {acc_aug:.4f}  ({acc_aug * 100:.1f}%)")
    print(f"  Delta:             {delta:+.4f} ({direction})")
    print(f"{'=' * 45}")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(hist_real, label="Real only", marker="o", color="#E63946", linewidth=1.5)
    ax.plot(hist_aug, label="Real+Synthetic", marker="s", color="#2A9D8F", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Accuracy")
    ax.set_title("Classifier: Real Only vs GAN-Augmented Training")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(config["output_dir"], "classifier_comparison.png"), dpi=150)
    plt.close(fig)
    with open(os.path.join(config["output_dir"], "classifier_results.txt"), "w") as f:
        f.write(f"Real only best acc:      {acc_real:.4f}\n")
        f.write(f"Real+Synthetic best acc: {acc_aug:.4f}\n")
        f.write(f"Delta:                   {delta:+.4f}\n")
    return acc_real, acc_aug


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=CONFIG["data_dir"])
    parser.add_argument("--output-dir", default=CONFIG["output_dir"])
    parser.add_argument("--epochs", type=int, default=CONFIG["epochs"])
    parser.add_argument("--skip-fid", action="store_true")
    parser.add_argument("--skip-classifier", action="store_true")
    args = parser.parse_args()

    config = dict(CONFIG)
    config["data_dir"] = args.data_dir
    config["output_dir"] = args.output_dir
    config["epochs"] = args.epochs

    set_seed()
    ensure_output_dirs(config)
    print(f"Using device: {device}")

    # Fail early with a useful message instead of crashing inside pandas.
    from src.dataset import build_image_index  # noqa: E402

    if not os.path.isfile(os.path.join(config["data_dir"], "trainLabels.csv")):
        print(f"Data not found in {config['data_dir']}.")
        print("If you already have the data, point at it with --data-dir.")
        print("Otherwise download it once with:")
        print("  python scripts/download_data.py --data-dir "
              f"{config['data_dir']}")
        print("That script skips the download if the data is already there.")
        raise SystemExit(1)

    dataloader = build_gan_dataloader(config)

    G = Generator(config["latent_dim"], config["n_classes"], config["img_size"]).to(device)
    D = Discriminator(config["n_classes"], config["img_size"]).to(device)
    G.apply(weights_init)
    D.apply(weights_init)
    n_params_G = sum(p.numel() for p in G.parameters())
    n_params_D = sum(p.numel() for p in D.parameters())
    print(f"Generator     — {n_params_G:,} parameters")
    print(f"Discriminator — {n_params_D:,} parameters")
    print(f"Total         — {n_params_G + n_params_D:,} parameters")

    opt_G = optim.Adam(G.parameters(), lr=config["lr"], betas=(config["beta1"], 0.999))
    opt_D = optim.Adam(D.parameters(), lr=config["lr"], betas=(config["beta1"], 0.999))
    criterion = nn.BCELoss()
    print("Starting training...")
    g_losses, d_losses = train_gan(config, device, dataloader, G, D, opt_G, opt_D, criterion)
    plot_losses(g_losses, d_losses, config)

    generate_synthetic_images(G, config, device)

    if not args.skip_fid:
        fid_results = compute_fid(config, device)
        print_fid_summary(fid_results)

    if not args.skip_classifier:
        run_classifier(config)


if __name__ == "__main__":
    main()
