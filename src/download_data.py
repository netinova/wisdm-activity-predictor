import argparse
import os
import sys
import zipfile

import requests
from tqdm import tqdm

from config import DATA_DIR, DATASET_DOWNLOAD_LINK


def download_file(url, dest_path, force=False):

    if (
        os.path.exists(dest_path)
        or os.path.exists(os.path.join(os.path.dirname(dest_path), "wisdm-dataset"))
        and not force
    ):
        print(f"   File already exists: {dest_path}")
        print("   Use --force to re-download.")
        return False

    print(f"   Downloading from {url}")
    headers = {"User-Agent": "WISDM-downloader/1.0"}

    try:
        response = requests.get(url, stream=True, headers=headers, timeout=60)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"   [ERROR] Download failed: {e}", file=sys.stderr)
        sys.exit(1)

    total_size = int(response.headers.get("content-length", 295e6))
    with open(dest_path, "wb") as f:
        with tqdm(
            total=total_size,
            unit="B",
            unit_scale=True,
            desc="   Progress",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    print("   [OK] Download complete.")
    return True


def extract_zip(zip_path, extract_to, force=False):

    inner_zip = os.path.join(os.path.dirname(zip_path), "wisdm-dataset.zip")

    if not zipfile.is_zipfile(zip_path):
        print(f"   [ERROR] '{zip_path}' is not a valid ZIP file.", file=sys.stderr)
        sys.exit(1)

    os.makedirs(extract_to, exist_ok=True)

    print(f"   Extracting to {extract_to} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)
    with zipfile.ZipFile(inner_zip, "r") as zf:
        zf.extractall(extract_to)
    print("   [OK] Extraction finished.")

    os.remove(zip_path)
    os.remove(inner_zip)

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Download and extract the WISDM dataset."
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Re-download and overwrite existing files even if they already exist.",
    )
    args = parser.parse_args()

    print("WISDM Dataset Downloader")

    zip_filename = "wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset.zip"
    zip_path = os.path.join(DATA_DIR, zip_filename)

    print("\n[1/2] Downloading dataset ...")
    downloaded = download_file(DATASET_DOWNLOAD_LINK, zip_path, force=args.force)

    print("\n[2/2] Extracting dataset ...")
    extract_zip(zip_path, DATA_DIR, force=args.force)

    print("\n   Done. Data is ready in:", os.path.abspath(DATA_DIR))


if __name__ == "__main__":
    main()
