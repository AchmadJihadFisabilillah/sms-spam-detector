"""Download SMS Spam Collection Dataset from Kaggle.

Prerequisites:
1. Install dependencies: pip install -r backend/requirements.txt
2. Create Kaggle API token from Kaggle account settings.
3. Put kaggle.json in ~/.kaggle/kaggle.json or set KAGGLE_USERNAME and KAGGLE_KEY.
4. Run from project root: python backend/scripts/download_kaggle_dataset.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "backend" / "data"
DATASET = "uciml/sms-spam-collection-dataset"


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "kaggle",
        "datasets",
        "download",
        "-d",
        DATASET,
        "-p",
        str(DATA_DIR),
        "--unzip",
    ]
    print("Downloading Kaggle dataset:", DATASET)
    subprocess.run(cmd, check=True)
    expected_file = DATA_DIR / "spam.csv"
    if expected_file.exists():
        print(f"Dataset downloaded successfully: {expected_file}")
    else:
        print("Download finished, but spam.csv was not found. Check Kaggle output in backend/data.")


if __name__ == "__main__":
    main()
