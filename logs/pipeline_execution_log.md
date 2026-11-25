# PreFlight-3D: Data Engineering & Feature Extraction Log
**Last Updated:** 2026-11-25 09:31:00  
**System Specs:** Ubuntu Linux | 12 GB RAM | CPU-Only | Disk Free: 9.7 GB / 232.64 GB

---

## 1. Completed Pipeline Stages

### Stage 1: Repository & Environment Setup (`scripts/setup_uco3d.sh`)
- Created modular project architecture (`data/`, `ml_pipeline/`, `models/`, `backend/`, `frontend/`, `notebooks/`, `scripts/`, `logs/`).
- Configured `.venv` with `opencv-python-headless`, `numpy`, `pandas`, `pyyaml`, `tqdm`, `trimesh`, `requests`, and `huggingface_hub`.
- Cloned Meta AI's `facebookresearch/uco3d` toolkit into `external/uco3d` and excluded heavy data folders in `.gitignore`.

### Stage 2: Selective uCO3D Subset Download (`01_download_subset.py`)
- Downloaded `examples.zip` (8.71 GB) directly from Hugging Face (`facebook/uco3d`).
- Selectively extracted **262 lightweight files** (`rgb_video.mp4`, `mask_video.mkv`, `.ply` point clouds, `metadata.sqlite`) while excluding multi-GB depth maps and Gaussian splats, then deleted `examples.zip` immediately.
- **Result:** 52 distinct object categories (52 sequences) stored in `data/raw_uco3d/`.

### Stage 3: Sequence Selection & Dataset Structuring (`02_select_and_organize.py`)
- Selected **40 diverse sequences across 40 categories** spanning transparent (`wineglass`, `water_bottle`), metallic/reflective (`bass_horn`, `pot`, `flashlight`), thin geometry (`shopping_cart`, `chair`, `helicopter`), and high-texture matte (`shoe`, `teddy_bear`, `pinecone`) objects.
- Symlinked original videos to `data/processed/original_videos/` (zero duplicate storage), extracted 2 FPS RGB frames + binary foreground masks, copied `.ply` reconstructions, and exported **7,999 camera pose rows** (`frame_annots_selected.csv`) and **40 sequence annotations** (`sequence_annots_selected.csv`).

### Stage 4: Baseline Video & 3D Reconstruction Inspection (`03_inspect_and_measure.py`)
- **Object Visibility Ratio:** `1.000` across all 40 clean sequences.
- **Mean Segmented 3D Object Points:** `18361` points per object (`segmented_point_cloud.ply`).
- **Mean Sparse SfM Points:** `16047` points per scene (`sparse_point_cloud.ply`).

### Stage 5: Synthetic Failure Generation (`04_generate_degraded_videos.py`)
- Generated **240 labeled failure videos** (6 failure modes × 40 clean sequences) downscaled to max `640px` (~2–3 MB/clip) inside `data/degraded_videos/`.
- Compiled master manifest (`dataset_manifest.csv`) with **280 total labeled samples** (`40 GOOD`, `240 POOR`).

### Stage 6: Computer Vision & Geometric Feature Extraction (`extractor.py` & `run_batch_extraction.py`)
- Extracted **19 standardized CV, lighting, texture, and camera trajectory features** across all **280 videos** into `data/processed/quality_measurements/extracted_features.csv`.

---

## 2. Extracted Feature Means by Defect Category

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

---

## 3. Baseline Clean Sequences Summary (40 Objects)

| sequence_id | category | resolution | duration_sec | mean_laplacian_sharpness | mean_orb_density_per_10k_px | segmented_3d_points |
| --- | --- | --- | --- | --- | --- | --- |
| 27-79013-34290 | wineglass | 1080x1920 | 30.67 | 14.84 | 6.11 | 5414 |
| 70-1164-44120 | water_bottle | 1920x1080 | 44.7 | 28.24 | 16.53 | 11765 |
| 605-31775-92635 | perfume | 1920x1080 | 18.95 | 139.97 | 59.84 | 6869 |
| 47433-58842-22217 | vase | 1920x1088 | 21.13 | 56.95 | 43.54 | 15951 |
| 33-11430-17759 | ashtray | 1080x1920 | 30.33 | 191.76 | 41.52 | 19512 |
| 3-9266-65507 | bass_horn | 1080x1920 | 39.15 | 73.22 | 33.05 | 10289 |
| 16-11679-97735 | pot | 1920x1080 | 27.87 | 72.27 | 16.23 | 38142 |
| 494-29795-17364 | flashlight | 1920x1080 | 29.78 | 115.12 | 42.55 | 16187 |
| 153-67003-22766 | aerosol_can | 1080x1920 | 38.07 | 309.37 | 83.37 | 9771 |
| 8-20919-19870 | fire_extinguisher | 1920x1080 | 27.03 | 112.29 | 55.11 | 10327 |
| 20-38390-68010 | motor | 1920x1080 | 31.85 | 98.38 | 34.48 | 22914 |
| 27-34327-21011 | binoculars | 1080x1920 | 33.59 | 313.09 | 72.58 | 8629 |
| 533-16958-69462 | cooker | 3840x2160 | 16.01 | 12.54 | 3.96 | 40561 |
| 12-10308-28681 | shopping_cart | 1920x1080 | 30.81 | 63.55 | 31.45 | 24119 |
| 2017-26879-34905 | chair | 1920x1080 | 48.07 | 119.43 | 28.37 | 23037 |
| 90-44074-92954 | helicopter | 1920x1080 | 27.12 | 117.82 | 45.52 | 16449 |
| 1-89973-52840 | speaker_stero_equipment | 1080x1920 | 35.26 | 120.45 | 33.38 | 21923 |
| 96-692-17773 | telephone | 1920x1080 | 12.0 | 53.36 | 27.61 | 27216 |
| 156-6699-92799 | funnel | 1080x1920 | 37.27 | 50.66 | 18.04 | 7575 |
| 732-27981-57173 | shoe | 1080x1920 | 28.36 | 78.51 | 35.65 | 20533 |
| 47480-57073-46870 | teddy_bear | 1080x1920 | 57.47 | 36.67 | 21.09 | 30545 |
| 38-75852-49792 | pinecone | 1080x1920 | 26.19 | 189.66 | 123.32 | 6148 |
| 42033-32228-40100 | handbag | 1920x1080 | 25.17 | 224.78 | 33.05 | 23371 |
| 475-99485-99578 | pillow | 1080x1920 | 11.29 | 23.07 | 3.07 | 27490 |
| 117-3686-24815 | hand_towel | 1080x1920 | 41.23 | 198.13 | 42.46 | 17069 |
| 305-68822-66727 | shirt | 1920x1080 | 11.86 | 37.06 | 17.33 | 40474 |
| 1-90320-26441 | cap_headwear | 1080x1920 | 30.13 | 5.95 | 0.65 | 23728 |
| 7-88671-19551 | muffin | 1080x1920 | 47.1 | 250.13 | 192.0 | 3799 |
| 10-78890-3542 | bagel | 1080x1920 | 32.66 | 249.18 | 29.47 | 27620 |
| 767-14445-38072 | onion | 1080x1920 | 38.93 | 158.98 | 109.71 | 6527 |
| 54207-48862-40002 | orange_fruit | 1080x1920 | 73.95 | 12.19 | 7.55 | 4854 |
| 28006-14019-9638 | book | 1080x1920 | 14.17 | 221.12 | 38.6 | 20407 |
| 62387-12826-25112 | mouse_computer_equipment | 1080x1920 | 28.0 | 109.32 | 32.91 | 14777 |
| 401-36913-71936 | flowerpot | 1080x1920 | 40.37 | 126.44 | 17.94 | 38353 |
| 267-59948-42658 | sculpture | 1920x1080 | 26.91 | 54.65 | 62.87 | 10879 |
| 91-56583-481 | first-aid_kit | 1080x1920 | 30.7 | 20.01 | 14.78 | 35368 |
| 538-38847-906 | battery | 1920x1080 | 26.43 | 251.14 | 137.17 | 5499 |
| 57-41030-21122 | dog_collar | 1080x1920 | 37.23 | 1072.57 | 220.43 | 3899 |
| 26-14460-30571 | dustpan | 1080x1920 | 28.73 | 17.01 | 3.98 | 30151 |
| 4-91751-93241 | scraper | 1920x1080 | 32.67 | 72.38 | 21.16 | 6287 |
