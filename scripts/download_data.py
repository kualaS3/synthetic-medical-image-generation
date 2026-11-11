"""Download the Kaggle Diabetic Retinopathy data, once, then never again.

Usage::

    python scripts/download_data.py
    python scripts/download_data.py --data-dir ./kaggle_dr
    python scripts/download_data.py --from-zip ~/Downloads/diabetic-retinopathy-detection.zip
    python scripts/download_data.py --force        # re-download even if data is present

The important bit: this script is idempotent. If ``kaggle_dr/`` already
contains ``trainLabels.csv`` and at least one ``train_*`` folder with
``.jpeg`` files, it prints that and exits immediately. It does NOT
re-download the massive dataset.

If you already have the data on your machine, just point ``main.py`` at
it instead (``--data-dir``). You do not need this script at all.

If the data is NOT present, the script downloads it with the Kaggle API,
which needs two things you have to do once by hand:
  1. Accept the competition rules at
     https://www.kaggle.com/c/diabetic-retinopathy-detection/data
  2. Create an API token at https://www.kaggle.com/settings (Account →
     API → "Create New Token") and save the downloaded ``kaggle.json``
     to ``~/.kaggle/kaggle.json`` (or set KAGGLE_USERNAME /
     KAGGLE_KEY in the environment).

The competition set is large (~88,000 images in total. The notebook
indexed 35,126 train images), so the first download takes a while and
needs several GB free. The downloaded zip is deleted after unzipping,
unless you passed ``--keep-zip``.
"""
import argparse
import os
import shutil
import subprocess
import sys
import zipfile

COMPETITION = "diabetic-retinopathy-detection"
KAGGLE_URL = "https://www.kaggle.com/c/diabetic-retinopathy-detection/data"


def count_images(data_dir: str) -> int:
    total = 0
    if not os.path.isdir(data_dir):
        return 0
    for name in os.listdir(data_dir):
        folder = os.path.join(data_dir, name)
        if name.startswith("train") and os.path.isdir(folder):
            total += sum(1 for f in os.listdir(folder) if f.endswith(".jpeg"))
    return total


def data_present(data_dir: str) -> bool:
    """True if the folder already looks like a usable dataset."""
    labels = os.path.join(data_dir, "trainLabels.csv")
    return os.path.isfile(labels) and count_images(data_dir) > 0


def unzip_all(zip_path: str, dest_dir: str) -> None:
    """Unzip a zip, then unzip any nested train/test zips inside it."""
    print(f"Unzipping {zip_path} ...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest_dir)
    # Kaggle's competition download is a zip of zips (train.zip, test.zip).
    for name in list(os.listdir(dest_dir)):
        path = os.path.join(dest_dir, name)
        if name.endswith(".zip") and zipfile.is_zipfile(path):
            print(f"Unzipping nested {name} ...")
            with zipfile.ZipFile(path) as zf:
                zf.extractall(dest_dir)
            os.remove(path)


def flatten_single_subdir(data_dir: str) -> None:
    """If everything landed one folder deep (e.g. kaggle_dr/kaggle_dr/),
    move its contents up one level."""
    entries = os.listdir(data_dir)
    if len(entries) == 1:
        only = os.path.join(data_dir, entries[0])
        if os.path.isdir(only) and os.path.isfile(os.path.join(only, "trainLabels.csv")):
            for name in os.listdir(only):
                shutil.move(os.path.join(only, name), os.path.join(data_dir, name))
            os.rmdir(only)


def ensure_kaggle_cli() -> bool:
    if shutil.which("kaggle"):
        return True
    print("Kaggle CLI not found — installing it (pip install kaggle) ...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "kaggle"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        return False
    return shutil.which("kaggle") is not None


def credentials_present() -> bool:
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    return os.path.isfile(os.path.expanduser("~/.kaggle/kaggle.json"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="./kaggle_dr")
    parser.add_argument("--from-zip", default=None,
                        help="Use a zip you already downloaded instead of "
                             "downloading from Kaggle.")
    parser.add_argument("--force", action="store_true",
                        help="Re-download even if data is already present.")
    parser.add_argument("--keep-zip", action="store_true")
    args = parser.parse_args()

    data_dir = args.data_dir
    os.makedirs(data_dir, exist_ok=True)

    if data_present(data_dir) and not args.force:
        print(f"Data already present in {data_dir} "
              f"({count_images(data_dir):,} train images, "
              f"trainLabels.csv found) — skipping download.")
        print("Nothing was re-downloaded. Run main.py with "
              f"--data-dir {data_dir} when you are ready.")
        return 0

    if args.from_zip:
        zip_path = args.from_zip
        if not os.path.isfile(zip_path):
            print(f"Zip not found: {zip_path}")
            return 1
        unzip_all(zip_path, data_dir)
    else:
        if not credentials_present():
            print("No Kaggle API credentials found.")
            print("  1. Accept the competition rules at:")
            print(f"     {KAGGLE_URL}")
            print("  2. At https://www.kaggle.com/settings, create an API token")
            print("     and save kaggle.json to ~/.kaggle/kaggle.json")
            print("     (or set KAGGLE_USERNAME / KAGGLE_KEY).")
            print("Then re-run this script. Nothing was downloaded.")
            return 1
        if not ensure_kaggle_cli():
            print("Could not install the Kaggle CLI. Install it by hand:")
            print("  pip install kaggle")
            return 1
        zip_path = os.path.join(data_dir, f"{COMPETITION}.zip")
        print(f"Downloading {COMPETITION} — this is large, it will take a while ...")
        result = subprocess.run(
            ["kaggle", "competitions", "download", "-c", COMPETITION, "-p", data_dir]
        )
        if result.returncode != 0 or not os.path.isfile(zip_path):
            print("Download failed. Check that you accepted the competition "
                  "rules and that your API token is valid, then re-run.")
            return 1
        unzip_all(zip_path, data_dir)
        if not args.keep_zip:
            os.remove(zip_path)

    flatten_single_subdir(data_dir)

    if data_present(data_dir):
        print(f"Done. {data_dir} now has trainLabels.csv and "
              f"{count_images(data_dir):,} train images.")
        print(f"Run: python main.py --data-dir {data_dir}")
        return 0
    print(f"Something is off: {data_dir} does not look complete after "
          f"unzip (trainLabels.csv or train_*.jpeg files missing).")
    print(f"Contents: {os.listdir(data_dir)}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
