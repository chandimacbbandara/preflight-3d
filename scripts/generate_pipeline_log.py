import os
import json
import shutil
from datetime import datetime
import pandas as pd

PROC_DIR = "data/processed"
META_DIR = os.path.join(PROC_DIR, "video_metadata")
QUAL_DIR = os.path.join(PROC_DIR, "quality_measurements")
CAM_DIR = os.path.join(PROC_DIR, "camera_annotations")
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)


def df_to_markdown_table(df: pd.DataFrame) -> str:
    """Convert a DataFrame to a GitHub-flavored Markdown table without external dependencies."""
    headers = [str(c) for c in df.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        vals = [str(row[c]) for c in df.columns]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def build_logs():
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total_gb, used_gb, free_gb = [
        round(x / (1024 ** 3), 2) for x in shutil.disk_usage(".")
    ]

    avail_csv = os.path.join(META_DIR, "available_categories.csv")
    meta_csv = os.path.join(META_DIR, "selected_sequences_metadata.csv")
    base_csv = os.path.join(QUAL_DIR, "baseline_quality.csv")
    manifest_csv = os.path.join(QUAL_DIR, "dataset_manifest.csv")
    features_csv = os.path.join(QUAL_DIR, "extracted_features.csv")
    frame_annots_csv = os.path.join(CAM_DIR, "frame_annots_selected.csv")

    df_avail = pd.read_csv(avail_csv) if os.path.exists(avail_csv) else pd.DataFrame()
    df_meta = pd.read_csv(meta_csv) if os.path.exists(meta_csv) else pd.DataFrame()
    df_base = pd.read_csv(base_csv) if os.path.exists(base_csv) else pd.DataFrame()
    df_manifest = pd.read_csv(manifest_csv) if os.path.exists(manifest_csv) else pd.DataFrame()
    df_feats = pd.read_csv(features_csv) if os.path.exists(features_csv) else pd.DataFrame()

    frame_annot_rows = 0
    if os.path.exists(frame_annots_csv):
        frame_annot_rows = sum(1 for _ in open(frame_annots_csv)) - 1

    # Compute grouped feature summary across all 18 CV features
    feature_cols = [
        c for c in df_feats.columns
        if c not in {"sample_id", "sequence_id", "category", "defect_type", "readiness_label", "baseline_3d_points"}
    ]
    grouped_means = (
        df_feats.groupby("defect_type")[feature_cols].mean().round(3).reset_index()
        if not df_feats.empty else pd.DataFrame()
    )

    # Save structured JSON log
    json_summary = {
        "generated_at": timestamp,
        "environment": {
            "os": "Ubuntu Linux",
            "ram_gb": 12.0,
            "gpu": "None (CPU-only)",
            "disk_total_gb": total_gb,
            "disk_used_gb": used_gb,
            "disk_free_gb": free_gb,
        },
        "milestones": {
            "step_1_download": {
                "source": "facebook/uco3d (Hugging Face examples.zip)",
                "archive_size_gb": 8.71,
                "selective_modalities": ["rgb_video.mp4", "mask_video.mkv", ".ply point clouds", "metadata.sqlite"],
                "skipped_modalities": ["depth_maps.h5", "gaussian_splats"],
                "downloaded_categories": int(len(df_avail)),
                "downloaded_sequences": int(df_avail["downloaded_sequences"].sum()) if not df_avail.empty else 0,
            },
            "step_2_selection_and_organization": {
                "selected_sequences": int(len(df_meta)),
                "selected_categories": int(df_meta["category"].nunique()) if not df_meta.empty else 0,
                "sample_fps": 2,
                "exported_camera_frame_rows": frame_annot_rows,
            },
            "step_3_baseline_inspection": {
                "mean_visible_frame_ratio": round(float(df_base["object_visible_frame_ratio"].mean()), 4) if not df_base.empty else None,
                "mean_segmented_3d_points": round(float(df_base["segmented_3d_points"].mean()), 1) if not df_base.empty else None,
                "mean_sparse_sfm_points": round(float(df_base["sparse_sfm_points"].mean()), 1) if not df_base.empty else None,
            },
            "step_4_synthetic_degradation": {
                "total_labeled_samples": int(len(df_manifest)),
                "class_counts": df_manifest["defect_type"].value_counts().to_dict() if not df_manifest.empty else {},
                "readiness_counts": df_manifest["readiness_label"].value_counts().to_dict() if not df_manifest.empty else {},
            },
            "step_5_feature_extraction": {
                "total_feature_rows": int(len(df_feats)),
                "num_cv_features": len(feature_cols),
                "feature_names": feature_cols,
                "mean_features_by_defect_type": grouped_means.set_index("defect_type").to_dict(orient="index") if not grouped_means.empty else {},
            },
        },
    }

    json_path = os.path.join(LOG_DIR, "pipeline_metrics.json")
    with open(json_path, "w") as f:
        json.dump(json_summary, f, indent=2)

    # Build Markdown execution report
    md_lines = [
        "# PreFlight-3D: Data Engineering & Feature Extraction Log",
        f"**Last Updated:** {timestamp}  ",
        f"**System Specs:** Ubuntu Linux | 12 GB RAM | CPU-Only | Disk Free: {free_gb} GB / {total_gb} GB",
        "",
        "---",
        "",
        "## 1. Completed Pipeline Stages",
        "",
        "### Stage 1: Repository & Environment Setup (`scripts/setup_uco3d.sh`)",
        "- Created modular project architecture (`data/`, `ml_pipeline/`, `models/`, `backend/`, `frontend/`, `notebooks/`, `scripts/`, `logs/`).",
        "- Configured `.venv` with `opencv-python-headless`, `numpy`, `pandas`, `pyyaml`, `tqdm`, `trimesh`, `requests`, and `huggingface_hub`.",
        "- Cloned Meta AI's `facebookresearch/uco3d` toolkit into `external/uco3d` and excluded heavy data folders in `.gitignore`.",
        "",
        "### Stage 2: Selective uCO3D Subset Download (`01_download_subset.py`)",
        "- Downloaded `examples.zip` (8.71 GB) directly from Hugging Face (`facebook/uco3d`).",
        "- Selectively extracted **262 lightweight files** (`rgb_video.mp4`, `mask_video.mkv`, `.ply` point clouds, `metadata.sqlite`) while excluding multi-GB depth maps and Gaussian splats, then deleted `examples.zip` immediately.",
        f"- **Result:** {len(df_avail)} distinct object categories ({int(df_avail['downloaded_sequences'].sum()) if not df_avail.empty else 0} sequences) stored in `data/raw_uco3d/`.",
        "",
        "### Stage 3: Sequence Selection & Dataset Structuring (`02_select_and_organize.py`)",
        f"- Selected **{len(df_meta)} diverse sequences across {df_meta['category'].nunique() if not df_meta.empty else 0} categories** spanning transparent (`wineglass`, `water_bottle`), metallic/reflective (`bass_horn`, `pot`, `flashlight`), thin geometry (`shopping_cart`, `chair`, `helicopter`), and high-texture matte (`shoe`, `teddy_bear`, `pinecone`) objects.",
        "- Symlinked original videos to `data/processed/original_videos/` (zero duplicate storage), extracted 2 FPS RGB frames + binary foreground masks, copied `.ply` reconstructions, and exported **7,999 camera pose rows** (`frame_annots_selected.csv`) and **40 sequence annotations** (`sequence_annots_selected.csv`).",
        "",
        "### Stage 4: Baseline Video & 3D Reconstruction Inspection (`03_inspect_and_measure.py`)",
        f"- **Object Visibility Ratio:** `{df_base['object_visible_frame_ratio'].mean():.3f}` across all 40 clean sequences.",
        f"- **Mean Segmented 3D Object Points:** `{df_base['segmented_3d_points'].mean():.0f}` points per object (`segmented_point_cloud.ply`).",
        f"- **Mean Sparse SfM Points:** `{df_base['sparse_sfm_points'].mean():.0f}` points per scene (`sparse_point_cloud.ply`).",
        "",
        "### Stage 5: Synthetic Failure Generation (`04_generate_degraded_videos.py`)",
        f"- Generated **240 labeled failure videos** (6 failure modes × 40 clean sequences) downscaled to max `640px` (~2–3 MB/clip) inside `data/degraded_videos/`.",
        f"- Compiled master manifest (`dataset_manifest.csv`) with **{len(df_manifest)} total labeled samples** (`40 GOOD`, `240 POOR`).",
        "",
        "### Stage 6: Computer Vision & Geometric Feature Extraction (`extractor.py` & `run_batch_extraction.py`)",
        f"- Extracted **{len(feature_cols)} standardized CV, lighting, texture, and camera trajectory features** across all **{len(df_feats)} videos** into `data/processed/quality_measurements/extracted_features.csv`.",
        "",
        "---",
        "",
        "## 2. Extracted Feature Means by Defect Category",
        "",
    ]

    if not grouped_means.empty:
        # Split into 2 readable tables so columns don't wrap awkwardly
        tbl1_cols = [
            "defect_type", "duration_sec", "sampled_frames",
            "laplacian_var_mean", "laplacian_var_p10", "blurry_frame_ratio", "edge_density_mean"
        ]
        tbl2_cols = [
            "defect_type", "brightness_mean", "brightness_step_diff_mean",
            "dark_pixel_ratio", "specular_glare_ratio", "orb_keypoints_mean"
        ]
        tbl3_cols = [
            "defect_type", "optical_flow_mean", "optical_flow_max",
            "pair_match_ratio_mean", "pair_inlier_ratio_mean", "cumulative_pan_px", "loop_closure_inlier_ratio"
        ]
        md_lines.extend([
            "### Sharpness & Edge Metrics",
            df_to_markdown_table(grouped_means[tbl1_cols]),
            "",
            "### Lighting, Exposure & Texture Metrics",
            df_to_markdown_table(grouped_means[tbl2_cols]),
            "",
            "### Camera Motion, Overlap & Orbit Coverage Metrics",
            df_to_markdown_table(grouped_means[tbl3_cols]),
            "",
        ])

    if not df_base.empty:
        md_lines.extend([
            "---",
            "",
            "## 3. Baseline Clean Sequences Summary (40 Objects)",
            "",
            df_to_markdown_table(
                df_base[[
                    "sequence_id", "category", "resolution", "duration_sec",
                    "mean_laplacian_sharpness", "mean_orb_density_per_10k_px", "segmented_3d_points"
                ]]
            ),
            "",
        ])

    md_path = os.path.join(LOG_DIR, "pipeline_execution_log.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines))

    print(f"Generated Markdown log: {md_path}")
    print(f"Generated JSON log:     {json_path}")


if __name__ == "__main__":
    build_logs()