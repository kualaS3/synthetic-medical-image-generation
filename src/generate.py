"""Generate synthetic fundus images from the trained generator.

``generate_synthetic_images`` is verbatim from the notebook (cell 23).
"""
import os

import torch
import torchvision.utils as vutils
from tqdm import tqdm


def generate_synthetic_images(G, CONFIG, device):
    """
    Sample from the trained generator conditioned on target DR grades.
    Saves images to output_dir/synthetic/grade_X/
    """
    G.eval()
    print("Generating synthetic images...")

    for grade in CONFIG["target_grades"]:
        save_dir = os.path.join(CONFIG["output_dir"], "synthetic", f"grade_{grade}")
        os.makedirs(save_dir, exist_ok=True)

        n_batches = (CONFIG["n_generate"] + CONFIG["fid_batch"] - 1) // CONFIG["fid_batch"]
        generated = 0

        with torch.no_grad():
            for _ in tqdm(range(n_batches), desc=f"Grade {grade}"):
                bs = min(CONFIG["fid_batch"], CONFIG["n_generate"] - generated)
                z = torch.randn(bs, CONFIG["latent_dim"], device=device)
                labels = torch.full((bs,), grade, dtype=torch.long, device=device)
                imgs = G(z, labels).cpu()
                imgs = (imgs + 1) / 2  # → [0, 1]

                for i, img in enumerate(imgs):
                    vutils.save_image(
                        img, os.path.join(save_dir, f"syn_{grade}_{generated+i:05d}.png")
                    )
                generated += bs

        print(f"  Grade {grade}: {generated} images saved → {save_dir}")
