import os
import glob
import shutil
import sqlite3
import cv2
import yaml
import pandas as pd
from tqdm import tqdm

with open("ml_pipeline/dataset_builder/config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

RAW_DIR = os.path.abspath(cfg["uco3d_root"])
PROC_DIR = os.path.abspath(cfg["processed_root"])
TARGET_CATS = cfg["target_categories"]
PER_CAT = cfg["sequences_per_category"]
SAMPLE_FPS = cfg["sample_fps"]


def collect_selected_sequences():
    available_by_cat = {}
    for super_cat in sorted(os.listdir(RAW_DIR)):
        super_path = os.path.join(RAW_DIR, super_cat)
        if not os.path.isdir(super_path) or super_cat == "set_lists":
            continue
        for cat in sorted(os.listdir(super_path)):
            cat_path = os.path.join(super_path, cat)
            if not os.path.isdir(cat_path):
                continue
            for seq_id in sorted(os.listdir(cat_path)):
                seq_dir = os.path.join(cat_path, seq_id)
                rgb_path = os.path.join(seq_dir, "rgb_video.mp4")
                mask_path = os.path.join(seq_dir, "mask_video.mkv")
                if os.path.exists(rgb_path) and os.path.exists(mask_path):
                    available_by_cat.setdefault(cat, []).append({
                        "sequence_id": seq_id,
                        "super_category": super_cat,
                        "category": cat,
                        "raw_seq_dir": seq_dir,
                    })

    selected = []
    for cat in TARGET_CATS:
        items = available_by_cat.get(cat, [])
        if not items:
            print(f"[WARNING] Category '{cat}' not found in {RAW_DIR}.")
            continue
        selected.extend(items[:PER_CAT])

    print(f"Selected {len(selected)} sequences across {len(set(x['category'] for x in selected))} categories.")
    return selected


def extract_frames_and_masks(seq_info):
    seq_id = seq_info["sequence_id"]
    raw_dir = seq_info["raw_seq_dir"]
    rgb_src = os.path.join(raw_dir, "rgb_video.mp4")
    mask_src = os.path.join(raw_dir, "mask_video.mkv")

    # 1. Symlink original video (keeps raw_uco3d untouched & uses 0 extra disk space)
    orig_dir = os.path.join(PROC_DIR, "original_videos")
    os.makedirs(orig_dir, exist_ok=True)
    orig_link = os.path.join(orig_dir, f"{seq_id}.mp4")
    if not os.path.exists(orig_link):
        os.symlink(rgb_src, orig_link)

    # 2. Copy reconstruction .ply point clouds
    recon_dir = os.path.join(PROC_DIR, "reconstruction_results", seq_id)
    os.makedirs(recon_dir, exist_ok=True)
    for ply_name in ["sparse_point_cloud.ply", "point_cloud.ply", "segmented_point_cloud.ply"]:
        src_ply = os.path.join(raw_dir, ply_name)
        if os.path.exists(src_ply):
            shutil.copy2(src_ply, os.path.join(recon_dir, ply_name))

    # 3. Stream & extract sampled RGB frames + binary masks (low RAM footprint)
    frame_out_dir = os.path.join(PROC_DIR, "extracted_frames", seq_id, "rgb")
    mask_out_dir = os.path.join(PROC_DIR, "extracted_frames", seq_id, "masks")
    os.makedirs(frame_out_dir, exist_ok=True)
    os.makedirs(mask_out_dir, exist_ok=True)

    cap_rgb = cv2.VideoCapture(rgb_src)
    cap_mask = cv2.VideoCapture(mask_src)

    fps = cap_rgb.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap_rgb.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap_rgb.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_rgb.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 else 0.0

    step = max(1, int(round(fps / SAMPLE_FPS)))
    frame_idx = 0
    saved_count = 0

    while True:
        ret_r, frame_r = cap_rgb.read()
        ret_m, frame_m = cap_mask.read()
        if not ret_r or not ret_m:
            break

        if frame_idx % step == 0:
            cv2.imwrite(os.path.join(frame_out_dir, f"frame_{frame_idx:05d}.jpg"), frame_r)
            gray_mask = cv2.cvtColor(frame_m, cv2.COLOR_BGR2GRAY)
            _, bin_mask = cv2.threshold(gray_mask, 127, 255, cv2.THRESH_BINARY)
            cv2.imwrite(os.path.join(mask_out_dir, f"mask_{frame_idx:05d}.png"), bin_mask)
            saved_count += 1

        frame_idx += 1

    cap_rgb.release()
    cap_mask.release()

    return {
        "sequence_id": seq_id,
        "super_category": seq_info["super_category"],
        "category": seq_info["category"],
        "width": width,
        "height": height,
        "fps": round(fps, 2),
        "total_frames": total_frames,
        "duration_sec": round(duration, 2),
        "extracted_frames": saved_count,
        "raw_path": raw_dir,
    }


def export_camera_annotations(sequence_ids):
    sqlite_files = glob.glob(os.path.join(RAW_DIR, "**", "*.sqlite"), recursive=True)
    cam_out_dir = os.path.join(PROC_DIR, "camera_annotations")
    os.makedirs(cam_out_dir, exist_ok=True)

    if not sqlite_files:
        print("[INFO] No .sqlite database found inside examples subset (camera poses will be estimated via CV/SfM).")
        return

    db_path = sqlite_files[0]
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [r[0] for r in cursor.fetchall()]
    print(f"Found SQLite database ({os.path.basename(db_path)}) with tables: {tables}")

    for table in tables:
        cols_df = pd.read_sql(f"PRAGMA table_info({table});", conn)
        cols = cols_df["name"].tolist()
        seq_col = next((c for c in cols if "sequence" in c.lower()), None)
        if seq_col:
            placeholders = ",".join(["?"] * len(sequence_ids))
            query = f"SELECT * FROM {table} WHERE {seq_col} IN ({placeholders})"
            df_sub = pd.read_sql(query, conn, params=sequence_ids)
            if not df_sub.empty:
                out_file = os.path.join(cam_out_dir, f"{table}_selected.csv")
                df_sub.to_csv(out_file, index=False)
                print(f"Exported {len(df_sub)} rows from '{table}' to {out_file}")

    conn.close()


if __name__ == "__main__":
    sequences = collect_selected_sequences()
    meta_records = []
    for seq in tqdm(sequences, desc="Organizing 40 sequences"):
        meta = extract_frames_and_masks(seq)
        meta_records.append(meta)

    os.makedirs(os.path.join(PROC_DIR, "video_metadata"), exist_ok=True)
    df_meta = pd.DataFrame(meta_records)
    meta_csv = os.path.join(PROC_DIR, "video_metadata", "selected_sequences_metadata.csv")
    df_meta.to_csv(meta_csv, index=False)
    print(f"\nSaved metadata for {len(df_meta)} sequences to {meta_csv}")

    export_camera_annotations([s["sequence_id"] for s in sequences])