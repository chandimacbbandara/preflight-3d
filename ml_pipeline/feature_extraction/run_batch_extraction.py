import os
import sys
import pandas as pd
from tqdm import tqdm

# Ensure project root is importable
sys.path.append(os.path.abspath("."))
from ml_pipeline.feature_extraction.extractor import extract_video_features

MANIFEST_CSV = "data/processed/quality_measurements/dataset_manifest.csv"
OUTPUT_CSV = "data/processed/quality_measurements/extracted_features.csv"


def main():
    df_manifest = pd.read_csv(MANIFEST_CSV)
    rows = []

    for _, item in tqdm(df_manifest.iterrows(), total=len(df_manifest), desc="Extracting CV features (280 videos)"):
        try:
            feats = extract_video_features(item["video_path"])
            record = {
                "sample_id": item["sample_id"],
                "sequence_id": item["sequence_id"],
                "category": item["category"],
                "defect_type": item["defect_type"],
                "readiness_label": item["readiness_label"],
                "baseline_3d_points": item["baseline_3d_points"],
                **feats,
            }
            rows.append(record)
        except Exception as e:
            print(f"[ERROR] Failed on {item['sample_id']}: {e}")

    df_features = pd.DataFrame(rows)
    df_features.to_csv(OUTPUT_CSV, index=False)

    print(f"\nSaved {len(df_features)} feature vectors to: {OUTPUT_CSV}")
    print("\nMean Feature Comparison by Defect Type:")
    summary_cols = [
        "laplacian_var_mean",
        "brightness_mean",
        "brightness_step_diff_mean",
        "specular_glare_ratio",
        "optical_flow_mean",
        "cumulative_pan_px",
    ]
    print(df_features.groupby("defect_type")[summary_cols].mean().round(3).to_string())


if __name__ == "__main__":
    main()