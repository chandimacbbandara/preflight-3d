# PreFlight-3D: Master Pipeline Execution & Experiment Log
**Last Updated:** 2026-11-25 10:23:02  
**System Specs:** Ubuntu Linux | 12 GB RAM | CPU-Only | Disk Free: 9.6 GB / 232.64 GB

---

## 1. Completed Pipeline Stages

### Stage 1–3: uCO3D Subset Acquisition & Dataset Organization
- Downloaded `examples.zip` (8.71 GB) from Hugging Face (`facebook/uco3d`) and selectively extracted 262 lightweight files (52 categories).
- Organized **40 diverse sequences** across 40 object categories with 2 FPS sampled frames, binary masks, `.ply` point clouds, and 7999 camera annotation rows.

### Stage 4–5: Baseline 3D Inspection & Synthetic Failure Generation
- Verified `1.000` object visibility ratio and `18361` mean 3D object points across clean captures.
- Generated **240 labeled failure videos** across 6 defect categories (`motion_blur`, `low_light`, `harsh_glare`, `exposure_flicker`, `low_overlap_fast_move`, `incomplete_orbit`) + 40 clean videos = **280 total samples**.

### Stage 6: Computer Vision & Geometric Feature Extraction
- Extracted **19 standardized CV & camera trajectory features** across all 280 videos (`extracted_features.csv`).

### Stage 7: Model Training, Hyperparameter Tuning & Soft-Voting Ensemble
- Evaluated **Random Forest**, **XGBoost**, and **LightGBM** (Baselines + `RandomizedSearchCV` Tuned + `SoftVoting_Ensemble`) using 5-fold `StratifiedGroupKFold` grouped by `sequence_id` to guarantee zero object leakage.
- **Winning 7-Class Defect Diagnosis Model:** `SoftVoting_Ensemble` (5-Fold CV Accuracy: `77.14%`, Macro F1: `0.7655`, Unseen Holdout Accuracy: `76.79%`).
- **Winning Binary Readiness Model:** `XGBoost_Tuned` (5-Fold CV Accuracy: `86.79%`, Macro F1: `0.7302`, Unseen Holdout Accuracy: `85.71%`).
- Saved serialized production bundle to `models/preflight_model_bundle.joblib`.

---

## 2. Model Benchmark Results (Baseline vs. Tuned vs. Ensemble)

| task | stage | model | cv_accuracy_mean | cv_accuracy_std | cv_macro_f1_mean | holdout_accuracy | holdout_macro_f1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Defect_Diagnosis_7Class | 1_Baseline | RandomForest | 0.7393 | 0.0571 | 0.7267 | 0.7321 | 0.743 |
| Defect_Diagnosis_7Class | 1_Baseline | XGBoost | 0.7607 | 0.072 | 0.7516 | 0.7679 | 0.764 |
| Defect_Diagnosis_7Class | 1_Baseline | LightGBM | 0.7536 | 0.0602 | 0.747 | 0.75 | 0.7577 |
| Defect_Diagnosis_7Class | 2_Tuned | RandomForest_Tuned | 0.7607 | 0.0702 | 0.7491 | 0.7679 | 0.7617 |
| Defect_Diagnosis_7Class | 2_Tuned | XGBoost_Tuned | 0.7643 | 0.0398 | 0.7607 | 0.7321 | 0.7287 |
| Defect_Diagnosis_7Class | 2_Tuned | LightGBM_Tuned | 0.7643 | 0.0633 | 0.7619 | 0.7857 | 0.799 |
| Defect_Diagnosis_7Class | 3_Ensemble | SoftVoting_Ensemble | 0.7714 | 0.0522 | 0.7655 | 0.7679 | 0.7721 |
| Readiness_Binary | 1_Baseline | RandomForest | 0.8321 | 0.0182 | 0.7021 | 0.8036 | 0.7168 |
| Readiness_Binary | 1_Baseline | XGBoost | 0.8607 | 0.0208 | 0.715 | 0.875 | 0.7573 |
| Readiness_Binary | 1_Baseline | LightGBM | 0.8393 | 0.0226 | 0.6809 | 0.8571 | 0.7352 |
| Readiness_Binary | 2_Tuned | RandomForest_Tuned | 0.8464 | 0.0182 | 0.7268 | 0.8393 | 0.7363 |
| Readiness_Binary | 2_Tuned | XGBoost_Tuned | 0.8679 | 0.0143 | 0.7302 | 0.8571 | 0.7352 |
| Readiness_Binary | 2_Tuned | LightGBM_Tuned | 0.8536 | 0.0262 | 0.6975 | 0.8571 | 0.7352 |
| Readiness_Binary | 3_Ensemble | SoftVoting_Ensemble | 0.85 | 0.0182 | 0.7092 | 0.8571 | 0.7352 |

---

## 3. Extracted Feature Means by Defect Category

### Sharpness & Edge Metrics
| defect_type | duration_sec | sampled_frames | laplacian_var_mean | laplacian_var_p10 | blurry_frame_ratio | edge_density_mean |
| --- | --- | --- | --- | --- | --- | --- |
| exposure_flicker | 31.73 | 79.6 | 431.258 | 150.073 | 0.079 | 0.062 |
| harsh_glare | 31.73 | 79.6 | 173.532 | 80.185 | 0.239 | 0.034 |
| incomplete_orbit | 11.121 | 28.45 | 395.629 | 241.857 | 0.02 | 0.069 |
| low_light | 31.73 | 79.6 | 449.125 | 280.007 | 0.0 | 0.061 |
| low_overlap_fast_move | 31.759 | 79.7 | 364.792 | 211.402 | 0.021 | 0.067 |
| motion_blur | 31.73 | 79.6 | 47.992 | 21.642 | 0.53 | 0.023 |
| none | 31.73 | 79.6 | 654.996 | 361.87 | 0.005 | 0.074 |

### Lighting, Exposure & Texture Metrics
| defect_type | brightness_mean | brightness_step_diff_mean | dark_pixel_ratio | specular_glare_ratio | orb_keypoints_mean |
| --- | --- | --- | --- | --- | --- |
| exposure_flicker | 123.494 | 76.578 | 0.126 | 0.11 | 395.427 |
| harsh_glare | 223.553 | 3.082 | 0.0 | 0.597 | 251.799 |
| incomplete_orbit | 117.113 | 3.709 | 0.102 | 0.007 | 444.034 |
| low_light | 57.705 | 3.424 | 0.452 | 0.004 | 385.478 |
| low_overlap_fast_move | 117.629 | 3.974 | 0.099 | 0.007 | 436.04 |
| motion_blur | 117.582 | 3.979 | 0.085 | 0.005 | 139.144 |
| none | 119.454 | 3.985 | 0.094 | 0.007 | 452.839 |

### Camera Motion, Overlap & Orbit Coverage Metrics
| defect_type | optical_flow_mean | optical_flow_max | pair_match_ratio_mean | pair_inlier_ratio_mean | cumulative_pan_px | loop_closure_inlier_ratio |
| --- | --- | --- | --- | --- | --- | --- |
| exposure_flicker | 9.596 | 19.985 | 0.274 | 0.769 | 35.198 | 0.532 |
| harsh_glare | 8.443 | 18.284 | 0.245 | 0.681 | 29.824 | 0.381 |
| incomplete_orbit | 9.113 | 15.324 | 0.301 | 0.851 | 17.752 | 0.095 |
| low_light | 8.416 | 16.615 | 0.244 | 0.823 | 31.286 | 0.512 |
| low_overlap_fast_move | 9.483 | 18.675 | 0.283 | 0.84 | 36.246 | 0.521 |
| motion_blur | 6.214 | 13.081 | 0.302 | 0.709 | 11.87 | 0.355 |
| none | 9.72 | 18.686 | 0.28 | 0.843 | 39.011 | 0.533 |
