"""Evaluation: Fréchet Inception Distance (FID).

``compute_fid`` is verbatim from the notebook (cell 26), plus the
FID summary printing from cell 27 as ``print_fid_summary``.
"""
import os

import pandas as pd
import torchvision.utils as vutils
from PIL import Image
from torchvision import transforms
from tqdm import tqdm


def compute_fid(CONFIG, device):
    """
    Compute FID between real grade images and synthetic images.
    """
    from pytorch_fid import fid_score

    print("Computing FID scores...")
    fid_results = {}

    # Build image lookup once (train_1 ... train_5)
    image_lookup = {}
    for d in os.listdir(CONFIG["data_dir"]):
        folder = os.path.join(CONFIG["data_dir"], d)
        if d.startswith("train") and os.path.isdir(folder):
            for fname in os.listdir(folder):
                if fname.endswith(".jpeg"):
                    image_lookup[fname.replace(".jpeg", "")] = os.path.join(folder, fname)

    tf = transforms.Compose(
        [
            transforms.Resize((CONFIG["img_size"], CONFIG["img_size"])),
            transforms.ToTensor(),
        ]
    )

    for grade in CONFIG["target_grades"]:
        real_dir = os.path.join(CONFIG["data_dir"], "fid_real", f"grade_{grade}")
        fake_dir = os.path.join(CONFIG["output_dir"], "synthetic", f"grade_{grade}")

        # Prepare real images for FID (only if not already done)
        if not os.path.exists(real_dir) or len(os.listdir(real_dir)) == 0:
            print(f"  Preparing real grade {grade} images...")
            os.makedirs(real_dir, exist_ok=True)
            df = pd.read_csv(os.path.join(CONFIG["data_dir"], "trainLabels.csv"))
            sub = df[df["level"] == grade]
            for _, row in tqdm(sub.iterrows(), total=len(sub)):
                src = image_lookup.get(row["image"])
                if src is None or not os.path.exists(src):
                    continue
                img = Image.open(src).convert("RGB")
                img = tf(img)
                vutils.save_image(img, os.path.join(real_dir, row["image"] + ".png"))

        n_real = len(os.listdir(real_dir))
        n_fake = len(os.listdir(fake_dir)) if os.path.exists(fake_dir) else 0
        print(f"  Real dir: {real_dir}, count: {n_real}")
        print(f"  Fake dir: {fake_dir}, count: {n_fake}")

        if n_real == 0 or n_fake == 0:
            print(f"  Skipping Grade {grade}: not enough images (real={n_real}, fake={n_fake})")
            continue

        # Cap batch_size to available images to avoid batch_size=0
        batch_size = min(CONFIG["fid_batch"], n_real, n_fake)

        print(f"  FID: Grade {grade} real vs synthetic...")
        fid_val = fid_score.calculate_fid_given_paths(
            [real_dir, fake_dir], batch_size=batch_size, device=device, dims=2048
        )
        fid_results[grade] = fid_val
        print(f"  Grade {grade} FID = {fid_val:.2f}")

    # Save
    with open(os.path.join(CONFIG["output_dir"], "fid_scores.txt"), "w") as f:
        f.write("FID Scores\n" + "=" * 30 + "\n")
        for g, v in fid_results.items():
            f.write(f"Grade {g}: {v:.4f}\n")

    return fid_results


def print_fid_summary(fid_results):
    if fid_results:
        print("\nFID Score Summary")
        print("=" * 40)
        for grade, score in fid_results.items():
            if score < 20:
                quality = "good"
            elif score < 50:
                quality = "decent"
            else:
                quality = "poor — consider more epochs or larger model"
            print(f"  Grade {grade}: FID = {score:.2f}  [{quality}]")
        print()
        print("FID < 10 is excellent, 10-20 is good, 20-50 is acceptable")
        print("These thresholds are rough guides — medical image FID is not")
        print("directly comparable to natural image FID (different feature spaces).")
