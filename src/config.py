"""Central configuration for the conditional DCGAN project.

Taken verbatim from the original notebook (cell 5), with one fix noted
below: the notebook's DataLoader hard-coded ``batch_size=4`` while
``CONFIG["batch_size"]`` said 10. The .py version uses CONFIG everywhere,
so this file is the single source of truth.
"""
import os
import random

import numpy as np
import torch

SEED = 42


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


# Centralise all hyperparameters here for easy tuning and reference.
CONFIG = {
    "data_dir": "./kaggle_dr",
    "output_dir": "./outputs_e20-s200",
    "epochs": 25,
    "batch_size": 10,  # Small batch size
    "lr": 0.0002,  # Standard DCGAN learning rate
    "beta1": 0.5,  # Adam beta1, paper recommends 0.5 for GANs
    "latent_dim": 64,  # Size of noise vector z
    "img_size": 128,  # Resize images
    "n_classes": 5,  # DR grades 0, 1, 2, 3, 4
    "target_grades": [3, 4],  # Grades we want to over-sample with synthetic data
    "n_generate": 100,  # Synthetic images per target grade
    "fid_batch": 32,
    "save_every": 10,
    "sample_every": 5,
    "num_workers": 0,
}

# Auto-select device, i dont have a compatible gpu :(
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


def ensure_output_dirs(config: dict = CONFIG) -> None:
    os.makedirs(config["output_dir"], exist_ok=True)
    os.makedirs(os.path.join(config["output_dir"], "samples"), exist_ok=True)
    os.makedirs(os.path.join(config["output_dir"], "checkpoints"), exist_ok=True)
