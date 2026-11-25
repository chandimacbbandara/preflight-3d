import os
import cv2
import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm

with open("ml_pipeline/dataset_builder/config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

PROC_DIR = os.path.abspath(cfg["processed_root"])
DEGRADED_DIR = os.path.abspath("data/degraded_videos")
BASELINE_CSV = os.path.join(PROC_DIR, "quality_measurements", "baseline_quality.csv")

MAX_DIM = 640  # Keeps file sizes ~2-3 MB each to protect disk space


def resize_keep_aspect(frame, max_dim=MAX_DIM):
    h, w = frame.shape[:2]
    scale = max_dim / max(h, w)
    if scale >= 1.0:
        return frame
    new_w = int(round(w * scale)) // 2 * 2
    new_h = int(round(h * scale)) // 2 * 2
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def apply_motion_blur(frame, kernel_size=19):
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    kernel[kernel_size // 2, :] = np.ones(kernel_size, dtype=np.float32)
    kernel /= kernel_size
    return cv2.filter2D(frame, -1, kernel)


def apply_low_light(frame, gamma=2.5):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).astype("uint8")
    dark = cv2.LUT(frame, table)
    noise = np.random.normal(0, 8, dark.shape).astype(np.int16)
    return np.clip(dark.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def apply_harsh_glare(frame, mask):
    # Overexpose image + inject strong specular glare hotspot inside object region
    overexposed = cv2.convertScaleAbs(frame, alpha=1.55, beta=45)
    h, w = frame.shape[:2]
    ys, xs = np.where(mask > 127)
    if len(xs) > 0:
        cx, cy = int(np.mean(xs)), int(np.mean(ys))
    else:
        cx, cy = w // 2, h // 2

    Y, X = np.ogrid[:h, :w]
    radius = max(min(h, w) // 5, 25)
    dist_sq = (X - cx) ** 2 + (Y - cy) ** 2
    hotspot = np.exp(-dist_sq / (2.0 * (radius ** 2))) * 220.0
    glare = np.clip(overexposed.astype(np.float32) + hotspot[..., None], 0, 255).astype(np.uint8)
    return glare


def apply_exposure_flicker(frame, frame_idx):
    # Abrupt auto-exposure oscillation every ~12 frames
    factor = 1.45 if (frame_idx // 12) % 2 == 0 else 0.55
    return cv2.convertScaleAbs(frame, alpha=factor, beta=0)


def generate_variants_for_sequence(row):
    seq_id = row["sequence_id"]
    category = row["category"]
    orig_video_path = os.path.join(PROC_DIR, "original_videos", f"{seq_id}.mp4")

    # Find raw mask_video.mkv via raw_uco3d
    meta_df = pd.read_csv(os.path.join(PROC_DIR, "video_metadata", "selected_sequences_metadata.csv"))
    raw_dir = meta_df.loc[meta_df["sequence_id"] == seq_id, "raw_path"].values[0]
    orig_mask_path = os.path.join(raw_dir, "mask_video.mkv")

    cap_rgb = cv2.VideoCapture(orig_video_path)
    cap_mask = cv2.VideoCapture(orig_mask_path)

    fps = cap_rgb.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap_rgb.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    defect_types = [
        "motion_blur",
        "low_light",
        "harsh_glare",
        "exposure_flicker",
        "low_overlap_fast_move",
        "incomplete_orbit",
    ]

    writers = {}
    out_paths = {}
    for d_type in defect_types:
        d_folder = os.path.join(DEGRADED_DIR, d_type)
        os.makedirs(d_folder, exist_ok=True)
        out_paths[d_type] = os.path.join(d_folder, f"{seq_id}__{d_type}.mp4")

    frame_idx = 0
    orbit_cutoff = int(total_frames * 0.35)

    while True:
        ret_r, frame_r = cap_rgb.read()
        ret_m, frame_m = cap_mask.read()
        if not ret_r:
            break

        frame_small = resize_keep_aspect(frame_r, MAX_DIM)
        h, w = frame_small.shape[:2]

        if ret_m and frame_m is not None:
            mask_small = cv2.resize(
                cv2.cvtColor(frame_m, cv2.COLOR_BGR2GRAY), (w, h), interpolation=cv2.INTER_NEAREST
            )
        else:
            mask_small = np.ones((h, w), dtype=np.uint8) * 255

        # Initialize writers once frame dimensions are known
        if not writers:
            for d_type in defect_types:
                out_fps = fps / 3.0 if d_type == "low_overlap_fast_move" else fps
                writers[d_type] = cv2.VideoWriter(out_paths[d_type], fourcc, max(out_fps, 5.0), (w, h))

        # 1. Motion blur
        writers["motion_blur"].write(apply_motion_blur(frame_small))

        # 2. Low light
        writers["low_light"].write(apply_low_light(frame_small))

        # 3. Harsh glare
        writers["harsh_glare"].write(apply_harsh_glare(frame_small, mask_small))

        # 4. Exposure flicker
        writers["exposure_flicker"].write(apply_exposure_flicker(frame_small, frame_idx))

        # 5. Low overlap (keep only every 3rd frame -> 67% frame drop)
        if frame_idx % 3 == 0:
            writers["low_overlap_fast_move"].write(frame_small)

        # 6. Incomplete orbit (keep only first 35% of capture)
        if frame_idx <= orbit_cutoff:
            writers["incomplete_orbit"].write(frame_small)

        frame_idx += 1

    cap_rgb.release()
    cap_mask.release()
    for w_obj in writers.values():
        w_obj.release()

    # Build manifest rows (1 GOOD original + 6 POOR degraded variants)
    records = [{
        "sample_id": f"{seq_id}__clean",
        "sequence_id": seq_id,
        "category": category,
        "video_path": orig_video_path,
        "mask_video_path": orig_mask_path,
        "defect_type": "none",
        "readiness_label": "GOOD",
        "baseline_3d_points": row["segmented_3d_points"],
    }]

    for d_type in defect_types:
        records.append({
            "sample_id": f"{seq_id}__{d_type}",
            "sequence_id": seq_id,
            "category": category,
            "video_path": out_paths[d_type],
            "mask_video_path": orig_mask_path,
            "defect_type": d_type,
            "readiness_label": "POOR",
            "baseline_3d_points": row["segmented_3d_points"],
        })

    return records


if __name__ == "__main__":
    np.random.seed(42)
    df_base = pd.read_csv(BASELINE_CSV)
    all_records = []

    for _, row in tqdm(df_base.iterrows(), total=len(df_base), desc="Generating degraded dataset"):
        all_records.extend(generate_variants_for_sequence(row))

    df_manifest = pd.DataFrame(all_records)
    manifest_csv = os.path.join(PROC_DIR, "quality_measurements", "dataset_manifest.csv")
    df_manifest.to_csv(manifest_csv, index=False)

    print(f"\nGenerated {len(df_manifest)} total labeled samples ({sum(df_manifest['readiness_label']=='GOOD')} GOOD, {sum(df_manifest['readiness_label']=='POOR')} POOR).")
    print(f"Manifest saved to: {manifest_csv}")
    print("\nClass Distribution:")
    print(df_manifest["defect_type"].value_counts().to_string())