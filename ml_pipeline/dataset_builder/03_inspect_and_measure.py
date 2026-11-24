import os
import ast
import glob
import cv2
import yaml
import trimesh
import numpy as np
import pandas as pd
from tqdm import tqdm

with open("ml_pipeline/dataset_builder/config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

PROC_DIR = os.path.abspath(cfg["processed_root"])
META_CSV = os.path.join(PROC_DIR, "video_metadata", "selected_sequences_metadata.csv")
FRAME_ANNOTS_CSV = os.path.join(PROC_DIR, "camera_annotations", "frame_annots_selected.csv")
SEQ_ANNOTS_CSV = os.path.join(PROC_DIR, "camera_annotations", "sequence_annots_selected.csv")


def compute_camera_elevation_stats(df_cam_seq):
    """Extract camera height/elevation diversity from uCO3D viewpoint_T if available."""
    t_col = next((c for c in df_cam_seq.columns if c.endswith("_T") or c == "T"), None)
    if t_col is None or df_cam_seq.empty:
        return {"cam_elevation_std": np.nan, "cam_elevation_range": np.nan}

    translations = []
    for val in df_cam_seq[t_col].dropna():
        try:
            vec = ast.literal_eval(val) if isinstance(val, str) else val
            if isinstance(vec, (list, tuple)) and len(vec) == 3:
                translations.append([float(x) for x in vec])
        except (ValueError, SyntaxError):
            continue

    if len(translations) < 3:
        return {"cam_elevation_std": np.nan, "cam_elevation_range": np.nan}

    pts = np.array(translations)
    norms = np.linalg.norm(pts, axis=1, keepdims=True) + 1e-8
    unit_pts = pts / norms
    # Y-axis (index 1) typically corresponds to vertical elevation in PyTorch3D coordinates
    elev_angles_deg = np.degrees(np.arcsin(np.clip(unit_pts[:, 1], -1.0, 1.0)))
    return {
        "cam_elevation_std": round(float(np.std(elev_angles_deg)), 2),
        "cam_elevation_range": round(float(np.ptp(elev_angles_deg)), 2),
    }


def analyze_sequence(row, df_frame_annots):
    seq_id = row["sequence_id"]
    rgb_paths = sorted(glob.glob(os.path.join(PROC_DIR, "extracted_frames", seq_id, "rgb", "*.jpg")))
    mask_paths = sorted(glob.glob(os.path.join(PROC_DIR, "extracted_frames", seq_id, "masks", "*.png")))

    mask_coverages = []
    edge_clipping_frames = 0
    sharpness_scores = []
    brightness_means = []
    flow_magnitudes = []
    orb_densities = []

    orb = cv2.ORB_create(nfeatures=1000)
    prev_gray = None

    for r_path, m_path in zip(rgb_paths, mask_paths):
        img = cv2.imread(r_path)
        mask = cv2.imread(m_path, cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            continue

        h, w = mask.shape
        total_pixels = h * w
        obj_pixels = np.count_nonzero(mask > 127)
        coverage = obj_pixels / total_pixels
        mask_coverages.append(coverage)

        # Check if object is clipped at frame edges
        border_touch = (
            np.any(mask[0, :] > 127) or np.any(mask[-1, :] > 127) or
            np.any(mask[:, 0] > 127) or np.any(mask[:, -1] > 127)
        )
        if border_touch:
            edge_clipping_frames += 1

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Sharpness (Laplacian variance strictly inside foreground object mask)
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        obj_lap_var = lap[mask > 127].var() if obj_pixels > 100 else lap.var()
        sharpness_scores.append(obj_lap_var)

        # Mean brightness inside foreground object mask
        obj_brightness = gray[mask > 127].mean() if obj_pixels > 100 else gray.mean()
        brightness_means.append(obj_brightness)

        # Texture richness (ORB keypoints per 10k object pixels)
        kps = orb.detect(gray, mask)
        orb_densities.append((len(kps) / max(obj_pixels, 1)) * 10000.0)

        # Camera movement (Farneback Optical Flow on downscaled frames for low CPU/RAM cost)
        small_gray = cv2.resize(gray, (320, 180))
        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, small_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            flow_magnitudes.append(float(np.mean(mag)))
        prev_gray = small_gray

    # 3D Reconstruction ground-truth point counts
    seg_ply_path = os.path.join(PROC_DIR, "reconstruction_results", seq_id, "segmented_point_cloud.ply")
    sparse_ply_path = os.path.join(PROC_DIR, "reconstruction_results", seq_id, "sparse_point_cloud.ply")

    seg_points = 0
    sparse_points = 0
    if os.path.exists(seg_ply_path):
        pc_seg = trimesh.load(seg_ply_path, process=False)
        seg_points = len(pc_seg.vertices) if hasattr(pc_seg, "vertices") else 0
    if os.path.exists(sparse_ply_path):
        pc_sparse = trimesh.load(sparse_ply_path, process=False)
        sparse_points = len(pc_sparse.vertices) if hasattr(pc_sparse, "vertices") else 0

    # Camera pose elevation stats from exported frame annotations
    cam_stats = {"cam_elevation_std": np.nan, "cam_elevation_range": np.nan}
    if df_frame_annots is not None:
        seq_col = next((c for c in df_frame_annots.columns if "sequence" in c.lower()), None)
        if seq_col:
            cam_stats = compute_camera_elevation_stats(df_frame_annots[df_frame_annots[seq_col] == seq_id])

    n_frames = max(len(mask_coverages), 1)
    visible_ratio = sum(1 for c in mask_coverages if c > 0.01) / n_frames

    return {
        "sequence_id": seq_id,
        "category": row["category"],
        "resolution": f"{row['width']}x{row['height']}",
        "fps": row["fps"],
        "duration_sec": row["duration_sec"],
        "object_visible_frame_ratio": round(visible_ratio, 4),
        "mean_object_screen_coverage": round(float(np.mean(mask_coverages)), 4),
        "border_clipping_ratio": round(edge_clipping_frames / n_frames, 4),
        "mean_laplacian_sharpness": round(float(np.mean(sharpness_scores)), 2),
        "min_laplacian_sharpness": round(float(np.min(sharpness_scores)), 2),
        "mean_brightness": round(float(np.mean(brightness_means)), 2),
        "exposure_stability_std": round(float(np.std(brightness_means)), 2),
        "mean_orb_density_per_10k_px": round(float(np.mean(orb_densities)), 2),
        "mean_optical_flow_px": round(float(np.mean(flow_magnitudes)) if flow_magnitudes else 0.0, 3),
        "std_optical_flow_px": round(float(np.std(flow_magnitudes)) if flow_magnitudes else 0.0, 3),
        "cam_elevation_std": cam_stats["cam_elevation_std"],
        "cam_elevation_range": cam_stats["cam_elevation_range"],
        "segmented_3d_points": seg_points,
        "sparse_sfm_points": sparse_points,
        "defect_type": "none",
        "quality_label": "GOOD",
    }


if __name__ == "__main__":
    df_meta = pd.read_csv(META_CSV)
    df_frame_annots = pd.read_csv(FRAME_ANNOTS_CSV) if os.path.exists(FRAME_ANNOTS_CSV) else None

    results = []
    for _, row in tqdm(df_meta.iterrows(), total=len(df_meta), desc="Inspecting 40 sequences"):
        results.append(analyze_sequence(row, df_frame_annots))

    df_quality = pd.DataFrame(results)
    os.makedirs(os.path.join(PROC_DIR, "quality_measurements"), exist_ok=True)
    out_csv = os.path.join(PROC_DIR, "quality_measurements", "baseline_quality.csv")
    df_quality.to_csv(out_csv, index=False)

    print(f"\nSaved baseline quality report to: {out_csv}")
    print("\nSummary Overview:")
    cols = [
        "sequence_id", "category", "resolution",
        "object_visible_frame_ratio", "mean_laplacian_sharpness",
        "mean_orb_density_per_10k_px", "segmented_3d_points"
    ]
    print(df_quality[cols].head(15).to_string(index=False))
    print(f"\nMean Visible Frame Ratio: {df_quality['object_visible_frame_ratio'].mean():.3f}")
    print(f"Mean Segmented 3D Points: {df_quality['segmented_3d_points'].mean():.0f}")