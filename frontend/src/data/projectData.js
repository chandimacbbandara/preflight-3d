export const HERO_STATS = [
  { label: "uCO3D Objects", value: "40", sub: "40 diverse categories" },
  { label: "Labeled Clips", value: "280", sub: "40 clean + 240 degraded" },
  { label: "CV & Orbit Features", value: "19", sub: "~0.5s CPU extraction" },
  { label: "7-Class CV Acc", value: "77.14%", sub: "SoftVoting_Ensemble" },
  { label: "Binary Gate CV Acc", value: "86.79%", sub: "XGBoost_Tuned" },
];

export const PROBLEMS_LIST = [
  {
    title: "Silent Compute Waste in 3D Pipelines",
    detail:
      "COLMAP, NeRF, and 3D Gaussian Splatting often run for 20–90 minutes before failing or producing hollow meshes, floating floaters, and torn textures.",
  },
  {
    title: "Six Distinct Capture Failure Modes",
    detail:
      "Motion blur destroys ORB/SIFT keypoints; underexposure adds sensor noise; harsh specular glare shifts across viewpoints; auto-exposure flicker breaks color consistency; fast panning kills frame overlap; and incomplete orbits leave unmodeled backsides.",
  },
  {
    title: "Pure ML Alone Suffers Real-World Edge Gaps",
    detail:
      "At 2.5 FPS downsampling, fast-move frame drops mimic slower clean orbits, while normal directional room lighting can falsely trigger exposure-flicker predictions.",
  },
];

export const SOLUTIONS_LIST = [
  {
    title: "CPU-First 19-Feature Diagnostic Telemetry",
    detail:
      "Inspects sharpness (Laplacian variance & Canny edges), HSV glare & dark pixel ratio, ORB texture density, Farneback optical flow, and start-to-end RANSAC loop closure in ~0.3–0.5s on CPU.",
  },
  {
    title: "Zero-Leakage Group-Aware Ensemble",
    detail:
      "Trained with 5-Fold StratifiedGroupKFold grouped by sequence_id so all 7 variants of an object stay together. Combines tuned RF + XGBoost + LightGBM (SoftVoting_Ensemble) with XGBoost_Tuned binary gating.",
  },
  {
    title: "Deterministic Physics Gates & Guardrails",
    detail:
      "Verifies ML alerts against physical CV measurements, suppresses false positives on turntable/indoor gradients, catches extreme trajectory gaps deterministically, and outputs a calibrated 0–100 Readiness Score with step-by-step capture fixes.",
  },
];

export const PIPELINE_STAGES = [
  {
    id: "A",
    title: "Toolkit & Environment Setup",
    script: "scripts/setup_uco3d.sh",
    tag: "Setup",
    summary:
      "Installs OpenCV, NumPy, Pandas, PyYAML, tqdm, and trimesh, and clones Meta's official facebookresearch/uco3d toolkit into external/uco3d.",
    highlights: ["CPU-only Ubuntu environment", "12 GB RAM constraint", "Zero GPU required"],
  },
  {
    id: "B",
    title: "Selective uCO3D Subset Acquisition",
    script: "ml_pipeline/dataset_builder/01_download_subset.py",
    tag: "Ingestion",
    summary:
      "Downloads examples.zip (8.71 GB) from Hugging Face (facebook/uco3d), selectively extracts only .mp4, .mkv, .ply, .sqlite, and .json files (skipping heavy gaussian_splats), and deletes the archive.",
    highlights: ["52 categories / 52 sequences", "~75% disk space saved", "available_categories.csv"],
  },
  {
    id: "C",
    title: "40-Category Curation & Extraction",
    script: "ml_pipeline/dataset_builder/02_select_and_organize.py",
    tag: "Curation",
    summary:
      "Selects 40 diverse sequences across 5 material/geometry groups (Glossy, Metallic, Thin/Complex, Organic Matte, Rigid), symlinks RGB videos, extracts 2 FPS frames & binary masks, copies .ply point clouds, and exports camera annotations.",
    highlights: ["40 object categories", "7,999 camera annotation rows", "21.15 min total source footage"],
  },
  {
    id: "D",
    title: "Baseline 3D & Quality Audit",
    script: "ml_pipeline/dataset_builder/03_inspect_and_measure.py",
    tag: "3D Audit",
    summary:
      "Audits all 40 clean sequences against their foreground masks and .ply reconstructions to verify clean ground-truth baselines before synthetic degradation.",
    highlights: ["1.000 object visible ratio", "18,360.7 mean segmented 3D pts", "16,047.5 mean sparse SfM pts"],
  },
  {
    id: "E",
    title: "Synthetic Capture Failure Generation",
    script: "ml_pipeline/dataset_builder/04_generate_degraded_videos.py",
    tag: "Simulation",
    summary:
      "Generates 6 degraded variants per clean sequence at max_dim=640 (240 POOR clips + 40 clean GOOD clips = 280 labeled videos) covering blur, lighting, glare, flicker, overlap, and orbit completeness.",
    highlights: [
      "motion_blur (19x19 kernel)",
      "low_light (gamma 2.5 + noise)",
      "harsh_glare (alpha=1.55 + mask hotspot)",
      "exposure_flicker (1.45x/0.55x oscillation)",
      "low_overlap_fast_move (67% frame drop)",
      "incomplete_orbit (35% cutoff)",
    ],
  },
  {
    id: "F",
    title: "19-Feature CV & Trajectory Extraction",
    script: "ml_pipeline/feature_extraction/extractor.py",
    tag: "Features",
    summary:
      "Samples at 2.5 FPS (min 12 frames), resizes to 480px, and computes 19 sharpness, lighting, glare, ORB texture, Farneback optical flow, RANSAC match/inlier, and loop-closure features inside an elliptical subject ROI.",
    highlights: ["19 standardized features", "~0.3–0.5s per video on CPU", "extracted_features.csv"],
  },
  {
    id: "G",
    title: "Group-Aware Training, Tuning & Voting",
    script: "ml_pipeline/training/train_models.py",
    tag: "Training",
    summary:
      "Evaluates Random Forest, XGBoost, and LightGBM (Baselines, RandomizedSearchCV Tuned, and SoftVoting_Ensemble) under 5-Fold StratifiedGroupKFold grouped by sequence_id across 7-class defect diagnosis and binary readiness.",
    highlights: [
      "Zero object leakage (32 train / 8 holdout objects)",
      "7-Class Winner: SoftVoting_Ensemble (77.14% CV)",
      "Binary Winner: XGBoost_Tuned (86.79% CV)",
    ],
  },
  {
    id: "H",
    title: "Hybrid Physics Gate & Calibration",
    script: "backend/app/services/recommendation_engine.py",
    tag: "Inference",
    summary:
      "Gates ML predictions with deterministic physics verification (_passes_physics_verification), geometric & trajectory guardrails, a 4-pillar CV health score, and calibrated 0–100 Readiness Verdicts with actionable fixes.",
    highlights: [
      "Suppresses false turntable/indoor alerts",
      "Catches extreme low-overlap & short orbits",
      "READY / WARNINGS / RECAPTURE verdicts",
    ],
  },
  {
    id: "I",
    title: "FastAPI Service & React 19 Dashboard",
    script: "backend/app/main.py & frontend/src/App.jsx",
    tag: "Full-Stack",
    summary:
      "Exposes /api/health, /api/analyze, /api/samples, and /api/analyze-sample/{id} to power an interactive diagnostic studio with drag-and-drop video upload, benchmark sampling, probability bars, and live telemetry.",
    highlights: ["FastAPI + Uvicorn (:8000)", "React 19 + Vite (:5173)", "14 curated benchmark samples"],
  },
];

export const TAXONOMY_GROUPS = [
  {
    group: "Transparent & Glossy (5)",
    items: "wineglass, water_bottle, perfume, vase, ashtray",
    challenge: "Refraction & view-dependent specular highlights",
  },
  {
    group: "Metallic & Reflective (8)",
    items: "bass_horn, pot, flashlight, aerosol_can, fire_extinguisher, motor, binoculars, cooker",
    challenge: "Glare clipping & mirror-like low-feature regions",
  },
  {
    group: "Thin & Complex Geometry (6)",
    items: "shopping_cart, chair, helicopter, speaker_stero_equipment, telephone, funnel",
    challenge: "Thin wires/rotors causing SfM tracking dropouts",
  },
  {
    group: "Organic & High-Texture Matte (12)",
    items: "shoe, teddy_bear, pinecone, handbag, pillow, hand_towel, shirt, cap_headwear, muffin, bagel, onion, orange_fruit",
    challenge: "Strong textured reconstruction baselines",
  },
  {
    group: "Rigid Everyday Objects (9)",
    items: "book, mouse_computer_equipment, flowerpot, sculpture, first-aid_kit, battery, dog_collar, dustpan, scraper",
    challenge: "Varied scales, flat sides & everyday indoor lighting",
  },
];

export const BENCHMARK_RESULTS = {
  Defect_Diagnosis_7Class: [
    { stage: "1_Baseline", model: "RandomForest", cvAcc: "73.93%", cvStd: "0.0571", cvF1: "0.7267", holdAcc: "73.21%", holdF1: "0.7430" },
    { stage: "1_Baseline", model: "XGBoost", cvAcc: "76.07%", cvStd: "0.0720", cvF1: "0.7516", holdAcc: "76.79%", holdF1: "0.7640" },
    { stage: "1_Baseline", model: "LightGBM", cvAcc: "75.36%", cvStd: "0.0602", cvF1: "0.7470", holdAcc: "75.00%", holdF1: "0.7577" },
    { stage: "2_Tuned", model: "RandomForest_Tuned", cvAcc: "76.07%", cvStd: "0.0702", cvF1: "0.7491", holdAcc: "76.79%", holdF1: "0.7617" },
    { stage: "2_Tuned", model: "XGBoost_Tuned", cvAcc: "76.43%", cvStd: "0.0398", cvF1: "0.7607", holdAcc: "73.21%", holdF1: "0.7287" },
    { stage: "2_Tuned", model: "LightGBM_Tuned", cvAcc: "76.43%", cvStd: "0.0633", cvF1: "0.7619", holdAcc: "78.57%", holdF1: "0.7990" },
    { stage: "3_Ensemble", model: "SoftVoting_Ensemble", cvAcc: "77.14%", cvStd: "0.0522", cvF1: "0.7655", holdAcc: "76.79%", holdF1: "0.7721", winner: true },
  ],
  Readiness_Binary: [
    { stage: "1_Baseline", model: "RandomForest", cvAcc: "83.21%", cvStd: "0.0182", cvF1: "0.7021", holdAcc: "80.36%", holdF1: "0.7168" },
    { stage: "1_Baseline", model: "XGBoost", cvAcc: "86.07%", cvStd: "0.0208", cvF1: "0.7150", holdAcc: "87.50%", holdF1: "0.7573" },
    { stage: "1_Baseline", model: "LightGBM", cvAcc: "83.93%", cvStd: "0.0226", cvF1: "0.6809", holdAcc: "85.71%", holdF1: "0.7352" },
    { stage: "2_Tuned", model: "RandomForest_Tuned", cvAcc: "84.64%", cvStd: "0.0182", cvF1: "0.7268", holdAcc: "83.93%", holdF1: "0.7363" },
    { stage: "2_Tuned", model: "XGBoost_Tuned", cvAcc: "86.79%", cvStd: "0.0143", cvF1: "0.7302", holdAcc: "85.71%", holdF1: "0.7352", winner: true },
    { stage: "2_Tuned", model: "LightGBM_Tuned", cvAcc: "85.36%", cvStd: "0.0262", cvF1: "0.6975", holdAcc: "85.71%", holdF1: "0.7352" },
    { stage: "3_Ensemble", model: "SoftVoting_Ensemble", cvAcc: "85.00%", cvStd: "0.0182", cvF1: "0.7092", holdAcc: "85.71%", holdF1: "0.7352" },
  ],
};

export const HOLDOUT_CLASS_REPORT = [
  { cls: "harsh_glare", precision: "1.000", recall: "1.000", f1: 1.0, support: 8 },
  { cls: "incomplete_orbit", precision: "1.000", recall: "0.875", f1: 0.933, support: 8 },
  { cls: "low_light", precision: "1.000", recall: "0.875", f1: 0.933, support: 8 },
  { cls: "motion_blur", precision: "1.000", recall: "0.750", f1: 0.857, support: 8 },
  { cls: "exposure_flicker", precision: "0.778", recall: "0.875", f1: 0.824, support: 8 },
  { cls: "none (Clean)", precision: "0.462", recall: "0.750", f1: 0.571, support: 8 },
  { cls: "low_overlap_fast_move", precision: "0.333", recall: "0.250", f1: 0.286, support: 8 },
];

export const TOP_FEATURES = [
  { name: "dark_pixel_ratio", importance: 0.1574, role: "Underexposure & shadow clipping" },
  { name: "brightness_mean", importance: 0.1503, role: "Global exposure & highlight blowout" },
  { name: "laplacian_var_mean", importance: 0.0898, role: "Mean subject sharpness vs. motion blur" },
  { name: "duration_sec", importance: 0.0842, role: "Capture pacing & truncated orbits" },
  { name: "brightness_step_diff_mean", importance: 0.0795, role: "Auto-exposure flicker & pumping" },
  { name: "specular_glare_ratio", importance: 0.0794, role: "Glare hotspots (V>240 & S<45)" },
  { name: "sampled_frames", importance: 0.0629, role: "Temporal frame coverage" },
  { name: "brightness_std", importance: 0.0604, role: "Luminance variance across orbit" },
  { name: "laplacian_var_p10", importance: 0.0499, role: "Worst-decile frame blur" },
  { name: "blurry_frame_ratio", importance: 0.0418, role: "Share of frames below sharpness floor" },
];

export const FEATURE_CLASS_MEANS = [
  { defect: "none (Clean)", duration: "31.73", laplacian: "655.00", blurry: "0.005", brightness: "119.45", stepDiff: "3.99", dark: "0.094", glare: "0.007", orb: "452.84", loop: "0.533" },
  { defect: "motion_blur", duration: "31.73", laplacian: "47.99", blurry: "0.530", brightness: "117.58", stepDiff: "3.98", dark: "0.085", glare: "0.005", orb: "139.14", loop: "0.355" },
  { defect: "low_light", duration: "31.73", laplacian: "449.13", blurry: "0.000", brightness: "57.71", stepDiff: "3.42", dark: "0.452", glare: "0.004", orb: "385.48", loop: "0.512" },
  { defect: "harsh_glare", duration: "31.73", laplacian: "173.53", blurry: "0.239", brightness: "223.55", stepDiff: "3.08", dark: "0.000", glare: "0.597", orb: "251.80", loop: "0.381" },
  { defect: "exposure_flicker", duration: "31.73", laplacian: "431.26", blurry: "0.079", brightness: "123.49", stepDiff: "76.58", dark: "0.126", glare: "0.110", orb: "395.43", loop: "0.532" },
  { defect: "low_overlap_fast_move", duration: "31.76", laplacian: "364.79", blurry: "0.021", brightness: "117.63", stepDiff: "3.97", dark: "0.099", glare: "0.007", orb: "436.04", loop: "0.521" },
  { defect: "incomplete_orbit", duration: "11.12", laplacian: "395.63", blurry: "0.020", brightness: "117.11", stepDiff: "3.71", dark: "0.102", glare: "0.007", orb: "444.03", loop: "0.095" },
];

export const FINAL_OUTPUT_GALLERY = [
  {
    src: "/img/57939c0f-a5c6-4f4f-a697-8239f34f55d0.png",
    file: "1-34403-89911.mp4.down.mp4",
    score: 54,
    verdict: "USABLE WITH WARNINGS",
    tier: "warn",
    trigger: "Geometric Guardrail (72%)",
    summary:
      "Potted plant capture (28.7s). Even though the raw ML classifier leaned Clean (81.5%), Peak Optical Flow spiked to 25.49 px — the deterministic Geometric Guardrail caught insufficient frame overlap and flagged fast panning.",
  },
  {
    src: "/img/a3122948-cbd7-4800-af0d-147b602b41b8.png",
    file: "1-41035-27122.mp4.down.mp4",
    score: 68,
    verdict: "USABLE WITH WARNINGS",
    tier: "warn",
    trigger: "ML Model (26%)",
    summary:
      "Textured woven object capture (28.7s, 88.6% RANSAC inliers). The Soft-Voting Ensemble detected 26.1% Low Overlap Fast Move (Peak Optical Flow 18.63 px) and recommended maintaining >70% frame overlap.",
  },
  {
    src: "/img/ddd3a8f2-305d-417a-881e-74e3b66e599c.png",
    file: "WhatsApp Video 2026-11-19 at 23.12.35.mp4",
    score: 29,
    verdict: "RECAPTURE RECOMMENDED",
    tier: "bad",
    trigger: "ML Model (33%)",
    summary:
      "Real smartphone capture (2.83s, 0.32 MB). Dark shadows pushed Mean Brightness down to 75.2/255 with 22.3% Underexposed Pixels, triggering a High-Severity Low-Light warning and actionable lighting fixes.",
  },
];
