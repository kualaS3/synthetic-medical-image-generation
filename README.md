# Synthetic Medical Image Generation — Conditional GAN

BN3406 group project, NUS, 2025. A conditional DCGAN that generates
synthetic diabetic retinopathy fundus images for the under-represented
severe grades (3 and 4), and tests whether those synthetic images help
a downstream classifier.

This is a portfolio / learning piece from a course project, not a
clinical model — and the results show that honestly.

## Results, honestly

From the original notebook run:

- **FID: Grade 3 = 319.51, Grade 4 = 301.33.** Poor, by the notebook's own
  thresholds (FID < 10 excellent, 20–50 acceptable). The generated images
  are not close to the real distribution in Inception feature space.
  Only 100 synthetic images per grade were generated, and FID is
  high-variance at that sample size.
- **Downstream classifier: 45.0% → 49.5% (+4.5pp).** A ResNet-18 trained
  on real data only got 45.0% best validation accuracy; adding the
  synthetic Grade 3/4 images got 49.5%. A modest improvement, on a small
  balanced subset (1,000 images, 200/class, 15 epochs) — interesting,
  not conclusive.

So the takeaway is mixed, and worth stating plainly: the GAN did not
produce high-fidelity images, but the synthetic data still gave a small
lift to the classifier. That is the whole result. No clinical claims.

## What is in here

- `notebook.ipynb` — the original workbook, cleaned (outputs stripped,
  so no embedded real fundus images and no local file paths leak out).
  The narrative, figures and discussion are in there.
- `src/config.py` — hyperparameters, seeds, device selection.
- `src/dataset.py` — Kaggle dataset loading, CLAHE brightness
  standardisation, transforms, synthetic-augmented dataset.
- `src/models.py` — conditional Generator, Discriminator, ResNet-18
  classifier.
- `src/train.py` — GAN training loop, classifier training loop.
- `src/generate.py` — sample synthetic images from the trained Generator.
- `src/evaluate.py` — FID computation.
- `main.py` — the full pipeline end-to-end (train → generate → FID →
  classifier comparison), mirroring the notebook.
- `tests/test_smoke.py` — smoke tests with fake data, no Kaggle
  download needed.

## Data — not included, download it once

The dataset is large and not ours to redistribute. Download it from
Kaggle first:

https://www.kaggle.com/c/diabetic-retinopathy-detection/data

It is the Diabetic Retinopathy Detection (EyePacs) competition set —
about 88,000 fundus photographs in total; the notebook indexed 35,126
train images.

Setup is one script, and it is safe to re-run — it checks first and
skips the download entirely if the data is already there, so you never
re-download the massive set by accident:

```
python scripts/download_data.py
```

That needs a Kaggle API token once (`~/.kaggle/kaggle.json`, from
https://www.kaggle.com/settings, after accepting the competition
rules). If you already downloaded the zip by hand, use it instead —
no re-download:

```
python scripts/download_data.py --from-zip ~/Downloads/diabetic-retinopathy-detection.zip
```

If you already have the data on your machine, skip the script
entirely and just point the code at your existing folder with
`--data-dir`. The expected layout is:

```
kaggle_dr/
  train_1/ ... train_5/   # .jpeg fundus images
  test_1/ ... test_7/
  trainLabels.csv         # columns: image, level
```

## Run it

```
pip install -r requirements.txt
python scripts/download_data.py   # first time only, skips if data is there
python main.py --data-dir ./kaggle_dr
```

Useful flags: `--epochs 5 --skip-fid --skip-classifier` for a short
smoke run. The original run was on CPU (PyTorch 2.11.0+cpu, no GPU),
128px images, 25 epochs in the config, batch size 10, Adam lr 0.0002,
beta1 0.5, latent dim 64. Expect it to be slow on CPU.

Or run just the tests (no data needed):

```
python -m pytest -q
```

## Notes on the split from notebook to .py files

The code is taken from the notebook, with three fixes that were needed
to make it actually re-runnable:

1. **`KaggleDRDataset` was missing from the notebook.** It was used in
   two places but never defined — the notebook only ran because of
   interactive state. It is reconstructed here from how it is used and
   from the image-index code in the notebook.
2. **`standardise_brightness` was defined twice** (identical code).
   It now lives once, in `src/dataset.py`.
3. **Config vs code disagreements.** The notebook's DataLoader
   hard-coded `batch_size=4` while `CONFIG` said 10 — the .py version
   uses CONFIG. The notebook text also said 500 synthetic images per
   grade; the code generates 100 per grade, and that is what the
   classifier actually used. This README and the code use 100.
   Re-running with the default config may give slightly different
   numbers from the snapshot above — trust your own run.

## Limitations

- 128px images, small training set (300/class for the GAN), limited
  epochs, CPU-only training.
- FID needs ~10k samples to be reliable; 100 generated images per
  grade is far below that.
- FID uses Inception-v3 trained on ImageNet, not fundus images, so it
  cannot judge clinical validity (lesion patterns, haemorrhages,
  neovascularisation).
- One GAN, one classifier, one seed. No clinician assessed the
  synthetic images.
