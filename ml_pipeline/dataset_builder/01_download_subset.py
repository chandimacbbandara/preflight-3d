import os
import json
import shutil
import zipfile
import requests
import yaml
import pandas as pd
from tqdm import tqdm

with open("ml_pipeline/dataset_builder/config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

RAW_DIR = os.path.abspath(cfg["uco3d_root"])
os.makedirs(RAW_DIR, exist_ok=True)

# Only extract lightweight files needed for our CV + 3D pipeline (saves ~75% disk space)
ALLOWED_EXTENSIONS = (".mp4", ".mkv", ".ply", ".sqlite", ".json")


def find_hf_urls(urls_json_path):
    with open(urls_json_path, "r") as f:
        data = json.load(f)

    targets = {}

    def walk(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, str) and "huggingface.co" in v:
                    if "examples.zip" in v or k == "examples":
                        targets["examples.zip"] = v
                    elif "metadata.zip" in v or k == "metadata":
                        targets["metadata.zip"] = v
                else:
                    walk(v)
        elif isinstance(obj, list):
            for item in obj:
                walk(item)

    walk(data)
    # Fallback direct Hugging Face URLs for facebook/uco3d if structure differs
    targets.setdefault(
        "examples.zip",
        "https://huggingface.co/datasets/facebook/uco3d/resolve/main/examples.zip",
    )
    return targets


def download_file(url, dest_path):
    print(f"Downloading {os.path.basename(dest_path)} from Hugging Face...")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total_size = int(r.headers.get("content-length", 0))
        with open(dest_path, "wb") as f, tqdm(
            total=total_size, unit="B", unit_scale=True, unit_divisor=1024
        ) as bar:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))


def selective_extract_and_cleanup(zip_path, extract_dir):
    print(f"Selectively extracting RGB videos, masks, point clouds & metadata from {os.path.basename(zip_path)}...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = [
            m for m in zf.namelist()
            if not m.endswith("/")
            and m.lower().endswith(ALLOWED_EXTENSIONS)
            and "gaussian_splats" not in m
        ]
        for member in tqdm(members, desc="Extracting files"):
            zf.extract(member, extract_dir)

    os.remove(zip_path)
    print(f"Deleted archive {os.path.basename(zip_path)} to free disk space.")


def download_safe_subset():
    # Clean up failed previous attempt folder if present
    in_progress = os.path.join(RAW_DIR, "_in_progress")
    if os.path.exists(in_progress):
        shutil.rmtree(in_progress, ignore_errors=True)

    urls_json = os.path.join(RAW_DIR, "uco3d_dataset_download_urls.json")
    if not os.path.exists(urls_json):
        r = requests.get(
            "https://raw.githubusercontent.com/uco3d/uco3d.github.io/refs/heads/main/links/uco3d_dataset_download_urls.json",
            timeout=30,
        )
        r.raise_for_status()
        with open(urls_json, "wb") as f:
            f.write(r.content)

    targets = find_hf_urls(urls_json)
    for filename, url in targets.items():
        zip_path = os.path.join(RAW_DIR, filename)
        download_file(url, zip_path)
        selective_extract_and_cleanup(zip_path, RAW_DIR)


def inspect_downloaded_categories():
    records = []
    for super_cat in sorted(os.listdir(RAW_DIR)):
        super_path = os.path.join(RAW_DIR, super_cat)
        if not os.path.isdir(super_path) or super_cat == "set_lists":
            continue
        for cat in sorted(os.listdir(super_path)):
            cat_path = os.path.join(super_path, cat)
            if not os.path.isdir(cat_path):
                continue
            seqs = [
                s for s in os.listdir(cat_path)
                if os.path.isdir(os.path.join(cat_path, s))
            ]
            if seqs:
                records.append({
                    "super_category": super_cat,
                    "category": cat,
                    "downloaded_sequences": len(seqs),
                })

    if not records:
        print("[WARNING] No sequence folders found. Check directory layout inside data/raw_uco3d.")
        return

    df_summary = pd.DataFrame(records).sort_values(
        by="downloaded_sequences", ascending=False
    )
    os.makedirs("data/processed/video_metadata", exist_ok=True)
    summary_csv = "data/processed/video_metadata/available_categories.csv"
    df_summary.to_csv(summary_csv, index=False)

    print(f"\nTotal categories downloaded: {len(df_summary)}")
    print(f"Total sequences downloaded:  {df_summary['downloaded_sequences'].sum()}")
    print("\nDownloaded Categories Breakdown:")
    print(df_summary.to_string(index=False))


if __name__ == "__main__":
    download_safe_subset()
    inspect_downloaded_categories()