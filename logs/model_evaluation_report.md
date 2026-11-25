# PreFlight-3D: Model Training & Evaluation Report
**Generated:** 2026-11-25 10:20:20  
**Validation Strategy:** 5-Fold `StratifiedGroupKFold` grouped by `sequence_id` (zero object leakage)

---

## 1. Model Benchmark Comparison
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

**Selected Defect Diagnosis Model:** `SoftVoting_Ensemble`  
**Selected Readiness Score Model:** `XGBoost_Tuned`

---

## 2. Unseen Object Holdout Classification Report (56 Test Videos)
```text
precision    recall  f1-score   support

     exposure_flicker      0.778     0.875     0.824         8
          harsh_glare      1.000     1.000     1.000         8
     incomplete_orbit      1.000     0.875     0.933         8
            low_light      1.000     0.875     0.933         8
low_overlap_fast_move      0.333     0.250     0.286         8
          motion_blur      1.000     0.750     0.857         8
                 none      0.462     0.750     0.571         8

             accuracy                          0.768        56
            macro avg      0.796     0.768     0.772        56
         weighted avg      0.796     0.768     0.772        56
```

## 3. Holdout Confusion Matrix
| Actual \ Predicted | exposure_flicker | harsh_glare | incomplete_orbit | low_light | low_overlap_fast_move | motion_blur | none |
| --- | --- | --- | --- | --- | --- | --- | --- |
| True: exposure_flicker | 7 | 0 | 0 | 0 | 0 | 0 | 1 |
| True: harsh_glare | 0 | 8 | 0 | 0 | 0 | 0 | 0 |
| True: incomplete_orbit | 0 | 0 | 7 | 0 | 1 | 0 | 0 |
| True: low_light | 0 | 0 | 0 | 7 | 0 | 0 | 1 |
| True: low_overlap_fast_move | 1 | 0 | 0 | 0 | 2 | 0 | 5 |
| True: motion_blur | 0 | 0 | 0 | 0 | 2 | 6 | 0 |
| True: none | 1 | 0 | 0 | 0 | 1 | 0 | 6 |

---

## 4. Feature Importance Ranking
| Feature | Importance |
| --- | --- |
| dark_pixel_ratio | 0.1574 |
| brightness_mean | 0.1503 |
| laplacian_var_mean | 0.0898 |
| duration_sec | 0.0842 |
| brightness_step_diff_mean | 0.0795 |
| specular_glare_ratio | 0.0794 |
| sampled_frames | 0.0629 |
| brightness_std | 0.0604 |
| laplacian_var_p10 | 0.0499 |
| blurry_frame_ratio | 0.0418 |
| loop_closure_inlier_ratio | 0.0313 |
| orb_keypoints_mean | 0.0204 |
| pair_inlier_ratio_mean | 0.0173 |
| optical_flow_max | 0.0144 |
| orb_keypoints_min | 0.0142 |
| edge_density_mean | 0.0128 |
| cumulative_pan_px | 0.012 |
| pair_match_ratio_mean | 0.0114 |
| optical_flow_mean | 0.0105 |
