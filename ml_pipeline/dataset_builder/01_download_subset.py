import os
import sqlite3
import subprocess
import yaml
import pandas as pd

with open("ml_pipeline/dataset_builder/config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

RAW_DIR = os.path.abspath(cfg["uco3d_root"])
os.makedirs(RAW_DIR, exist_ok=True)


def download_safe_subset():
    print(f"Downloading lightweight uCO3D subset (52 sequences, no depth/splats) into {RAW_DIR}...")
    cmd = [
        "python",
        "external/uco3d/dataset_download/download_dataset.py",
        "--download_small_subset",
        "--download_folder", RAW_DIR,
        "--download_modalities", "rgb_videos,mask_videos,point_clouds",
        "--n_download_workers", "2",
        "--n_extract_workers", "2",
        "--clear_archives_after_unpacking",
    ]
    subprocess.run(cmd, check=True)


def inspect_downloaded_categories():
    db_path = os.path.join(RAW_DIR, "metadata.sqlite")
    if not os.path.exists(db_path):
        print(f"[INFO] metadata.sqlite not at root, scanning subdirectories in {RAW_DIR}...")

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