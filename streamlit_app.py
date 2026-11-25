import base64
import html
import os
import tempfile
import textwrap
import pandas as pd
import streamlit as st

from backend.app.services.recommendation_engine import PreFlightAnalyzer

st.set_page_config(
    page_title="PreFlight-3D | 3D Reconstruction Quality Gate",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Crisp Inline SVG Icons (Emoji-Free, matching Lucide icons in App.jsx)
# ---------------------------------------------------------------------------

ICONS = {
    "box": '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16Z"/><path d="m3.3 7 8.7 5 8.7-5"/><path d="M12 22V12"/></svg>',
    "film": '<svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="18" height="18" x="3" y="3" rx="2"/><path d="M7 3v18"/><path d="M3 7.5h4"/><path d="M3 12h18"/><path d="M3 16.5h4"/><path d="M17 3v18"/><path d="M17 7.5h4"/><path d="M17 16.5h4"/></svg>',
    "cpu": '<svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="16" height="16" x="4" y="4" rx="2"/><rect width="6" height="6" x="9" y="9" rx="1"/><path d="M15 2v2"/><path d="M15 20v2"/><path d="M2 15h2"/><path d="M2 9h2"/><path d="M20 15h2"/><path d="M20 9h2"/><path d="M9 2v2"/><path d="M9 20v2"/></svg>',
    "gauge": '<svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m12 14 4-4"/><path d="M3.34 19a10 10 0 1 1 17.32 0"/></svg>',
    "alert": '<svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><path d="M12 9v4"/><path d="M12 17h.01"/></svg>',
    "check": '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="m9 12 2 2 4-4"/></svg>',
    "xcircle": '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="m15 9-6 6"/><path d="m9 9 6 6"/></svg>',
    "activity": '<svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 12h-2.48a2 2 0 0 0-1.93 1.46l-2.35 8.36a.25.25 0 0 1-.48 0L9.24 2.18a.25.25 0 0 0-.48 0l-2.35 8.36A2 2 0 0 1 4.49 12H2"/></svg>',
    "sliders": '<svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="4" x2="4" y1="21" y2="14"/><line x1="4" x2="4" y1="10" y2="3"/><line x1="12" x2="12" y1="21" y2="12"/><line x1="12" x2="12" y1="8" y2="3"/><line x1="20" x2="20" y1="21" y2="16"/><line x1="20" x2="20" y1="12" y2="3"/><line x1="2" x2="6" y1="14" y2="14"/><line x1="10" x2="14" y1="8" y2="8"/><line x1="18" x2="22" y1="16" y2="16"/></svg>',
    "eye": '<svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M2.062 12.348a1 1 0 0 1 0-.696 10.75 10.75 0 0 1 19.876 0 1 1 0 0 1 0 .696 10.75 10.75 0 0 1-19.876 0"/><circle cx="12" cy="12" r="3"/></svg>',
    "branch": '<svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="6" x2="6" y1="3" y2="15"/><circle cx="18" cy="6" r="3"/><circle cx="6" cy="18" r="3"/><path d="M18 9a9 9 0 0 1-9 9"/></svg>',
    "db": '<svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5V19A9 3 0 0 0 21 19V5"/><path d="M3 12A9 3 0 0 0 21 12"/></svg>',
    "award": '<svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="8" r="6"/><path d="M15.477 12.89 17 22l-5-3-5 3 1.523-9.11"/></svg>',
    "chart": '<svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 3v18h18"/><path d="M18 17V9"/><path d="M13 17V5"/><path d="M8 17v-3"/></svg>',
    "layers": '<svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m12.83 2.18a2 2 0 0 0-1.66 0L2.6 6.08a1 1 0 0 0 0 1.83l8.58 3.91a2 2 0 0 0 1.66 0l8.58-3.9a1 1 0 0 0 0-1.83Z"/><path d="m22 17.65-9.17 4.16a2 2 0 0 1-1.66 0L2 17.65"/><path d="m22 12.65-9.17 4.16a2 2 0 0 1-1.66 0L2 12.65"/></svg>',
    "monitor": '<svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="20" height="14" x="2" y="3" rx="2"/><line x1="8" x2="16" y1="21" y2="21"/><line x1="12" x2="12" y1="17" y2="21"/></svg>',
    "sparkles": '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9.937 15.5A2 2 0 0 0 8.5 14.063l-6.135-1.582a.5.5 0 0 1 0-.962L8.5 9.936A2 2 0 0 0 9.937 8.5l1.582-6.135a.5.5 0 0 1 .963 0L14.063 8.5A2 2 0 0 0 15.5 9.937l6.135 1.581a.5.5 0 0 1 0 .964L15.5 14.063a2 2 0 0 0-1.437 1.437l-1.582 6.135a.5.5 0 0 1-.963 0z"/></svg>',
    "shield": '<svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z"/><path d="m9 12 2 2 4-4"/></svg>',
}

# ---------------------------------------------------------------------------
# Data & Metadata (Synced with frontend/src/App.jsx & projectData.js)
# ---------------------------------------------------------------------------

METRIC_LABELS = [
    ("duration_sec", "Duration (sec)", "s", False, "Orbit"),
    ("sampled_frames", "Sampled Frames", "", False, "Orbit"),
    ("laplacian_var_mean", "Sharpness (Laplacian)", "", False, "Focus"),
    ("blurry_frame_ratio", "Blurry Frame Ratio", "%", True, "Focus"),
    ("edge_density_mean", "Edge Density", "%", True, "Focus"),
    ("brightness_mean", "Mean Brightness", "/255", False, "Light"),
    ("brightness_step_diff_mean", "Exposure Jump", "", False, "Light"),
    ("dark_pixel_ratio", "Underexposed Pixels", "%", True, "Light"),
    ("specular_glare_ratio", "Specular Glare Ratio", "%", True, "Light"),
    ("orb_keypoints_mean", "Avg ORB Keypoints", "", False, "Track"),
    ("optical_flow_mean", "Mean Optical Flow", " px", False, "Track"),
    ("optical_flow_max", "Peak Optical Flow", " px", False, "Track"),
    ("pair_match_ratio_mean", "Feature Match Ratio", "%", True, "Track"),
    ("pair_inlier_ratio_mean", "RANSAC Inlier Ratio", "%", True, "Track"),
    ("cumulative_pan_px", "Cumulative Orbit Pan", " px", False, "Orbit"),
    ("loop_closure_inlier_ratio", "Loop Closure Inliers", "%", True, "Orbit"),
]

HERO_STATS = [
    {"label": "uCO3D Objects", "value": "40", "sub": "40 diverse categories"},
    {"label": "Labeled Clips", "value": "280", "sub": "40 clean + 240 degraded"},
    {"label": "CV & Orbit Features", "value": "19", "sub": "~0.5s CPU extraction"},
    {"label": "7-Class CV Acc", "value": "77.14%", "sub": "SoftVoting_Ensemble"},
    {"label": "Binary Gate CV Acc", "value": "86.79%", "sub": "XGBoost_Tuned"},
]

PROBLEMS_LIST = [
    {
        "title": "Silent Compute Waste in 3D Pipelines",
        "detail": "COLMAP, NeRF, and 3D Gaussian Splatting often run for 20-90 minutes before failing or producing hollow meshes, floating floaters, and torn textures.",
    },
    {
        "title": "Six Distinct Capture Failure Modes",
        "detail": "Motion blur destroys ORB/SIFT keypoints; underexposure adds sensor noise; harsh specular glare shifts across viewpoints; auto-exposure flicker breaks color consistency; fast panning kills frame overlap; and incomplete orbits leave unmodeled backsides.",
    },
    {
        "title": "Pure ML Alone Suffers Real-World Edge Gaps",
        "detail": "At 2.5 FPS downsampling, fast-move frame drops mimic slower clean orbits, while normal directional room lighting can falsely trigger exposure-flicker predictions.",
    },
]

SOLUTIONS_LIST = [
    {
        "title": "CPU-First 19-Feature Diagnostic Telemetry",
        "detail": "Inspects sharpness (Laplacian variance & Canny edges), HSV glare & dark pixel ratio, ORB texture density, Farneback optical flow, and start-to-end RANSAC loop closure in ~0.3-0.5s on CPU.",
    },
    {
        "title": "Zero-Leakage Group-Aware Ensemble",
        "detail": "Trained with 5-Fold StratifiedGroupKFold grouped by sequence_id so all 7 variants of an object stay together. Combines tuned RF + XGBoost + LightGBM (SoftVoting_Ensemble) with XGBoost_Tuned binary gating.",
    },
    {
        "title": "Deterministic Physics Gates & Guardrails",
        "detail": "Verifies ML alerts against physical CV measurements, suppresses false positives on turntable/indoor gradients, catches extreme trajectory gaps deterministically, and outputs a calibrated 0-100 Readiness Score with step-by-step capture fixes.",
    },
]

PIPELINE_STAGES = [
    {
        "id": "A",
        "title": "Toolkit & Environment Setup",
        "script": "scripts/setup_uco3d.sh",
        "tag": "Setup",
        "summary": "Installs OpenCV, NumPy, Pandas, PyYAML, tqdm, and trimesh, and clones Meta's official facebookresearch/uco3d toolkit into external/uco3d.",
        "highlights": ["CPU-only Ubuntu environment", "12 GB RAM constraint", "Zero GPU required"],
    },
    {
        "id": "B",
        "title": "Selective uCO3D Subset Acquisition",
        "script": "ml_pipeline/dataset_builder/01_download_subset.py",
        "tag": "Ingestion",
        "summary": "Downloads examples.zip (8.71 GB) from Hugging Face (facebook/uco3d), selectively extracts only .mp4, .mkv, .ply, .sqlite, and .json files (skipping heavy gaussian_splats), and deletes the archive.",
        "highlights": ["52 categories / 52 sequences", "~75% disk space saved", "available_categories.csv"],
    },
    {
        "id": "C",
        "title": "40-Category Curation & Extraction",
        "script": "ml_pipeline/dataset_builder/02_select_and_organize.py",
        "tag": "Curation",
        "summary": "Selects 40 diverse sequences across 5 material/geometry groups (Glossy, Metallic, Thin/Complex, Organic Matte, Rigid), symlinks RGB videos, extracts 2 FPS frames & binary masks, copies .ply point clouds, and exports camera annotations.",
        "highlights": ["40 object categories", "7,999 camera annotation rows", "21.15 min total source footage"],
    },
    {
        "id": "D",
        "title": "Baseline 3D & Quality Audit",
        "script": "ml_pipeline/dataset_builder/03_inspect_and_measure.py",
        "tag": "3D Audit",
        "summary": "Audits all 40 clean sequences against their foreground masks and .ply reconstructions to verify clean ground-truth baselines before synthetic degradation.",
        "highlights": ["1.000 object visible ratio", "18,360.7 mean segmented 3D pts", "16,047.5 mean sparse SfM pts"],
    },
    {
        "id": "E",
        "title": "Synthetic Capture Failure Generation",
        "script": "ml_pipeline/dataset_builder/04_generate_degraded_videos.py",
        "tag": "Simulation",
        "summary": "Generates 6 degraded variants per clean sequence at max_dim=640 (240 POOR clips + 40 clean GOOD clips = 280 labeled videos) covering blur, lighting, glare, flicker, overlap, and orbit completeness.",
        "highlights": [
            "motion_blur (19x19 kernel)",
            "low_light (gamma 2.5 + noise)",
            "harsh_glare (alpha=1.55 + mask hotspot)",
            "exposure_flicker (1.45x/0.55x oscillation)",
            "low_overlap_fast_move (67% frame drop)",
            "incomplete_orbit (35% cutoff)",
        ],
    },
    {
        "id": "F",
        "title": "19-Feature CV & Trajectory Extraction",
        "script": "ml_pipeline/feature_extraction/extractor.py",
        "tag": "Features",
        "summary": "Samples at 2.5 FPS (min 12 frames), resizes to 480px, and computes 19 sharpness, lighting, glare, ORB texture, Farneback optical flow, RANSAC match/inlier, and loop-closure features inside an elliptical subject ROI.",
        "highlights": ["19 standardized features", "~0.3-0.5s per video on CPU", "extracted_features.csv"],
    },
    {
        "id": "G",
        "title": "Group-Aware Training, Tuning & Voting",
        "script": "ml_pipeline/training/train_models.py",
        "tag": "Training",
        "summary": "Evaluates Random Forest, XGBoost, and LightGBM (Baselines, RandomizedSearchCV Tuned, and SoftVoting_Ensemble) under 5-Fold StratifiedGroupKFold grouped by sequence_id across 7-class defect diagnosis and binary readiness.",
        "highlights": [
            "Zero object leakage (32 train / 8 holdout objects)",
            "7-Class Winner: SoftVoting_Ensemble (77.14% CV)",
            "Binary Winner: XGBoost_Tuned (86.79% CV)",
        ],
    },
    {
        "id": "H",
        "title": "Hybrid Physics Gate & Calibration",
        "script": "backend/app/services/recommendation_engine.py",
        "tag": "Inference",
        "summary": "Gates ML predictions with deterministic physics verification (_passes_physics_verification), geometric & trajectory guardrails, a 4-pillar CV health score, and calibrated 0-100 Readiness Verdicts with actionable fixes.",
        "highlights": [
            "Suppresses false turntable/indoor alerts",
            "Catches extreme low-overlap & short orbits",
            "READY / WARNINGS / RECAPTURE verdicts",
        ],
    },
    {
        "id": "I",
        "title": "Interactive Diagnostic Dashboard",
        "script": "streamlit_app.py & backend/app/main.py",
        "tag": "Full-Stack",
        "summary": "Powers an interactive diagnostic studio with drag-and-drop video upload, benchmark sampling, 4-pillar capture health breakdown, probability bars, and live telemetry.",
        "highlights": ["In-Process Python Engine", "Streamlit + FastAPI/React", "Curated benchmark samples"],
    },
]

TAXONOMY_GROUPS = [
    {
        "group": "Transparent & Glossy (5)",
        "items": "wineglass, water_bottle, perfume, vase, ashtray",
        "challenge": "Refraction & view-dependent specular highlights",
    },
    {
        "group": "Metallic & Reflective (8)",
        "items": "bass_horn, pot, flashlight, aerosol_can, fire_extinguisher, motor, binoculars, cooker",
        "challenge": "Glare clipping & mirror-like low-feature regions",
    },
    {
        "group": "Thin & Complex Geometry (6)",
        "items": "shopping_cart, chair, helicopter, speaker_stero_equipment, telephone, funnel",
        "challenge": "Thin wires/rotors causing SfM tracking dropouts",
    },
    {
        "group": "Organic & High-Texture Matte (12)",
        "items": "shoe, teddy_bear, pinecone, handbag, pillow, hand_towel, shirt, cap_headwear, muffin, bagel, onion, orange_fruit",
        "challenge": "Strong textured reconstruction baselines",
    },
    {
        "group": "Rigid Everyday Objects (9)",
        "items": "book, mouse_computer_equipment, flowerpot, sculpture, first-aid_kit, battery, dog_collar, dustpan, scraper",
        "challenge": "Varied scales, flat sides & everyday indoor lighting",
    },
]

BENCHMARK_RESULTS = {
    "Defect_Diagnosis_7Class": [
        {"stage": "1_Baseline", "model": "RandomForest", "cvAcc": "73.93%", "cvStd": "0.0571", "cvF1": "0.7267", "holdAcc": "73.21%", "holdF1": "0.7430"},
        {"stage": "1_Baseline", "model": "XGBoost", "cvAcc": "76.07%", "cvStd": "0.0720", "cvF1": "0.7516", "holdAcc": "76.79%", "holdF1": "0.7640"},
        {"stage": "1_Baseline", "model": "LightGBM", "cvAcc": "75.36%", "cvStd": "0.0602", "cvF1": "0.7470", "holdAcc": "75.00%", "holdF1": "0.7577"},
        {"stage": "2_Tuned", "model": "RandomForest_Tuned", "cvAcc": "76.07%", "cvStd": "0.0702", "cvF1": "0.7491", "holdAcc": "76.79%", "holdF1": "0.7617"},
        {"stage": "2_Tuned", "model": "XGBoost_Tuned", "cvAcc": "76.43%", "cvStd": "0.0398", "cvF1": "0.7607", "holdAcc": "73.21%", "holdF1": "0.7287"},
        {"stage": "2_Tuned", "model": "LightGBM_Tuned", "cvAcc": "76.43%", "cvStd": "0.0633", "cvF1": "0.7619", "holdAcc": "78.57%", "holdF1": "0.7990"},
        {"stage": "3_Ensemble", "model": "SoftVoting_Ensemble", "cvAcc": "77.14%", "cvStd": "0.0522", "cvF1": "0.7655", "holdAcc": "76.79%", "holdF1": "0.7721", "winner": True},
    ],
    "Readiness_Binary": [
        {"stage": "1_Baseline", "model": "RandomForest", "cvAcc": "83.21%", "cvStd": "0.0182", "cvF1": "0.7021", "holdAcc": "80.36%", "holdF1": "0.7168"},
        {"stage": "1_Baseline", "model": "XGBoost", "cvAcc": "86.07%", "cvStd": "0.0208", "cvF1": "0.7150", "holdAcc": "87.50%", "holdF1": "0.7573"},
        {"stage": "1_Baseline", "model": "LightGBM", "cvAcc": "83.93%", "cvStd": "0.0226", "cvF1": "0.6809", "holdAcc": "85.71%", "holdF1": "0.7352"},
        {"stage": "2_Tuned", "model": "RandomForest_Tuned", "cvAcc": "84.64%", "cvStd": "0.0182", "cvF1": "0.7268", "holdAcc": "83.93%", "holdF1": "0.7363"},
        {"stage": "2_Tuned", "model": "XGBoost_Tuned", "cvAcc": "86.79%", "cvStd": "0.0143", "cvF1": "0.7302", "holdAcc": "85.71%", "holdF1": "0.7352", "winner": True},
        {"stage": "2_Tuned", "model": "LightGBM_Tuned", "cvAcc": "85.36%", "cvStd": "0.0262", "cvF1": "0.6975", "holdAcc": "85.71%", "holdF1": "0.7352"},
        {"stage": "3_Ensemble", "model": "SoftVoting_Ensemble", "cvAcc": "85.00%", "cvStd": "0.0182", "cvF1": "0.7092", "holdAcc": "85.71%", "holdF1": "0.7352"},
    ],
}

HOLDOUT_CLASS_REPORT = [
    {"cls": "harsh_glare", "f1": 1.0},
    {"cls": "incomplete_orbit", "f1": 0.933},
    {"cls": "low_light", "f1": 0.933},
    {"cls": "motion_blur", "f1": 0.857},
    {"cls": "exposure_flicker", "f1": 0.824},
    {"cls": "none (Clean)", "f1": 0.571},
    {"cls": "low_overlap_fast_move", "f1": 0.286},
]

TOP_FEATURES = [
    {"name": "dark_pixel_ratio", "importance": 0.1574, "role": "Underexposure & shadow clipping"},
    {"name": "brightness_mean", "importance": 0.1503, "role": "Global exposure & highlight blowout"},
    {"name": "laplacian_var_mean", "importance": 0.0898, "role": "Mean subject sharpness vs. motion blur"},
    {"name": "duration_sec", "importance": 0.0842, "role": "Capture pacing & truncated orbits"},
    {"name": "brightness_step_diff_mean", "importance": 0.0795, "role": "Auto-exposure flicker & pumping"},
    {"name": "specular_glare_ratio", "importance": 0.0794, "role": "Glare hotspots (V>240 & S<45)"},
    {"name": "sampled_frames", "importance": 0.0629, "role": "Temporal frame coverage"},
    {"name": "brightness_std", "importance": 0.0604, "role": "Luminance variance across orbit"},
    {"name": "laplacian_var_p10", "importance": 0.0499, "role": "Worst-decile frame blur"},
    {"name": "blurry_frame_ratio", "importance": 0.0418, "role": "Share of frames below sharpness floor"},
]

FEATURE_CLASS_MEANS = [
    {"defect": "none (Clean)", "duration": "31.73", "laplacian": "655.00", "blurry": "0.005", "brightness": "119.45", "stepDiff": "3.99", "dark": "0.094", "glare": "0.007", "orb": "452.84", "loop": "0.533"},
    {"defect": "motion_blur", "duration": "31.73", "laplacian": "47.99", "blurry": "0.530", "brightness": "117.58", "stepDiff": "3.98", "dark": "0.085", "glare": "0.005", "orb": "139.14", "loop": "0.355"},
    {"defect": "low_light", "duration": "31.73", "laplacian": "449.13", "blurry": "0.000", "brightness": "57.71", "stepDiff": "3.42", "dark": "0.452", "glare": "0.004", "orb": "385.48", "loop": "0.512"},
    {"defect": "harsh_glare", "duration": "31.73", "laplacian": "173.53", "blurry": "0.239", "brightness": "223.55", "stepDiff": "3.08", "dark": "0.000", "glare": "0.597", "orb": "251.80", "loop": "0.381"},
    {"defect": "exposure_flicker", "duration": "31.73", "laplacian": "431.26", "blurry": "0.079", "brightness": "123.49", "stepDiff": "76.58", "dark": "0.126", "glare": "0.110", "orb": "395.43", "loop": "0.532"},
    {"defect": "low_overlap_fast_move", "duration": "31.76", "laplacian": "364.79", "blurry": "0.021", "brightness": "117.63", "stepDiff": "3.97", "dark": "0.099", "glare": "0.007", "orb": "436.04", "loop": "0.521"},
    {"defect": "incomplete_orbit", "duration": "11.12", "laplacian": "395.63", "blurry": "0.020", "brightness": "117.11", "stepDiff": "3.71", "dark": "0.102", "glare": "0.007", "orb": "444.03", "loop": "0.095"},
]

FINAL_OUTPUT_GALLERY = [
    {
        "path": "frontend/public/img/57939c0f-a5c6-4f4f-a697-8239f34f55d0.png",
        "file": "1-34403-89911.mp4.down.mp4",
        "score": 54,
        "verdict": "USABLE WITH WARNINGS",
        "tier": "warn",
        "trigger": "Geometric Guardrail (72%)",
        "summary": "Potted plant capture (28.7s). Even though the raw ML classifier leaned Clean (81.5%), Peak Optical Flow spiked to 25.49 px - the deterministic Geometric Guardrail caught insufficient frame overlap and flagged fast panning.",
    },
    {
        "path": "frontend/public/img/a3122948-cbd7-4800-af0d-147b602b41b8.png",
        "file": "1-41035-27122.mp4.down.mp4",
        "score": 68,
        "verdict": "USABLE WITH WARNINGS",
        "tier": "warn",
        "trigger": "ML Model (26%)",
        "summary": "Textured woven object capture (28.7s, 88.6% RANSAC inliers). The Soft-Voting Ensemble detected 26.1% Low Overlap Fast Move (Peak Optical Flow 18.63 px) and recommended maintaining >70% frame overlap.",
    },
    {
        "path": "frontend/public/img/ddd3a8f2-305d-417a-881e-74e3b66e599c.png",
        "file": "WhatsApp Video 2026-11-19 at 23.12.35.mp4",
        "score": 29,
        "verdict": "RECAPTURE RECOMMENDED",
        "tier": "bad",
        "trigger": "ML Model (33%)",
        "summary": "Real smartphone capture (2.83s, 0.32 MB). Dark shadows pushed Mean Brightness down to 75.2/255 with 22.3% Underexposed Pixels, triggering a High-Severity Low-Light warning and actionable lighting fixes.",
    },
]

# ---------------------------------------------------------------------------
# Dynamic Dark / Light Theme Stylesheet Generator
# ---------------------------------------------------------------------------


def build_theme_css(theme: str) -> str:
    if theme == "light":
        vars_block = """
:root {
  --bg-primary: #f1f5f9;
  --bg-card: #ffffff;
  --bg-elevated: #f8fafc;
  --border-subtle: #dbe3ef;
  --text-primary: #0f172a;
  --text-secondary: #475569;
  --text-muted: #64748b;
  --accent-blue: #2563eb;
  --accent-blue-hover: #1d4ed8;
  --status-good: #059669;
  --status-good-bg: rgba(16, 185, 129, 0.12);
  --status-good-text: #047857;
  --status-warn: #d97706;
  --status-warn-bg: rgba(245, 158, 11, 0.14);
  --status-warn-text: #b45309;
  --status-bad: #dc2626;
  --status-bad-bg: rgba(239, 68, 68, 0.11);
  --status-bad-text: #b91c1c;
  --accent-chip-text: #1d4ed8;
  --accent-chip-bg: rgba(37, 99, 235, 0.09);
  --card-shadow: 0 4px 18px rgba(15, 23, 42, 0.06);
  --hero-grad: linear-gradient(145deg, #ffffff, #eff6ff);
  --topbar-bg: rgba(255, 255, 255, 0.94);
  --app-bg: radial-gradient(circle at 15% 0%, rgba(37, 99, 235, 0.07), transparent 42%),
            radial-gradient(circle at 85% 10%, rgba(6, 182, 212, 0.06), transparent 36%),
            #f1f5f9;
}
"""
    else:
        vars_block = """
:root {
  --bg-primary: #0b0f19;
  --bg-card: #111827;
  --bg-elevated: #1a2235;
  --border-subtle: #263147;
  --text-primary: #f3f4f6;
  --text-secondary: #9ca3af;
  --text-muted: #6b7280;
  --accent-blue: #3b82f6;
  --accent-blue-hover: #2563eb;
  --status-good: #10b981;
  --status-good-bg: rgba(16, 185, 129, 0.12);
  --status-good-text: #34d399;
  --status-warn: #f59e0b;
  --status-warn-bg: rgba(245, 158, 11, 0.12);
  --status-warn-text: #fbbf24;
  --status-bad: #ef4444;
  --status-bad-bg: rgba(239, 68, 68, 0.12);
  --status-bad-text: #f87171;
  --accent-chip-text: #93c5fd;
  --accent-chip-bg: rgba(59, 130, 246, 0.12);
  --card-shadow: 0 8px 24px rgba(0, 0, 0, 0.22);
  --hero-grad: linear-gradient(160deg, rgba(26, 34, 53, 0.92), rgba(17, 24, 39, 0.96));
  --topbar-bg: rgba(17, 24, 39, 0.9);
  --app-bg: radial-gradient(circle at 15% 0%, rgba(37, 99, 235, 0.12), transparent 40%),
            radial-gradient(circle at 85% 10%, rgba(6, 182, 212, 0.09), transparent 35%),
            #0b0f19;
}
"""

    base_css = textwrap.dedent("""
#MainMenu, footer, header[data-testid="stHeader"] {
  visibility: hidden;
  height: 0;
}

.stApp {
  background: var(--app-bg) !important;
  color: var(--text-primary) !important;
}

.stApp p, .stApp label, .stApp span, .stApp div {
  color: inherit;
}

.block-container {
  max-width: 1380px;
  padding-top: 1.1rem !important;
  padding-bottom: 3rem !important;
}

/* Topbar */
.pf-topbar {
  background: var(--topbar-bg);
  backdrop-filter: blur(12px);
  border: 1px solid var(--border-subtle);
  border-radius: 14px;
  padding: 0.85rem 1.4rem;
  margin-bottom: 0.85rem;
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-wrap: wrap;
  gap: 1rem;
  box-shadow: var(--card-shadow);
}

.pf-brand {
  display: flex;
  align-items: center;
  gap: 0.8rem;
}

.pf-brand-icon {
  background: linear-gradient(135deg, #2563eb, #06b6d4);
  width: 40px;
  height: 40px;
  border-radius: 10px;
  display: flex;
  align-items: center;
  justify-content: center;
  color: #ffffff;
  box-shadow: 0 4px 14px rgba(37, 99, 235, 0.3);
}

.pf-brand h1 {
  margin: 0;
  font-size: 1.18rem;
  font-weight: 700;
  letter-spacing: -0.02em;
  color: var(--text-primary);
  line-height: 1.2;
}

.pf-brand span {
  font-size: 0.78rem;
  color: var(--text-secondary);
}

.pf-health-pill {
  display: inline-flex;
  align-items: center;
  gap: 0.5rem;
  background: var(--bg-elevated);
  border: 1px solid var(--border-subtle);
  padding: 0.42rem 0.9rem;
  border-radius: 999px;
  font-size: 0.78rem;
  color: var(--text-secondary);
}

.pf-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: var(--status-good);
  box-shadow: 0 0 8px var(--status-good);
}

/* Generic Card */
.pf-card {
  background: var(--bg-card);
  border: 1px solid var(--border-subtle);
  border-radius: 14px;
  padding: 1.35rem;
  box-shadow: var(--card-shadow);
  margin-bottom: 1.2rem;
}

.pf-card-title {
  font-size: 0.96rem;
  font-weight: 600;
  margin-bottom: 1rem;
  display: flex;
  align-items: center;
  gap: 0.5rem;
  color: var(--text-primary);
}

.pf-icon-wrap {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  color: var(--accent-blue);
}

/* Empty Hero */
.empty-hero {
  padding: 2.5rem 2.2rem;
  background: var(--hero-grad);
  border: 1px solid var(--border-subtle);
  border-radius: 14px;
  box-shadow: var(--card-shadow);
}

.empty-badge {
  display: inline-flex;
  align-items: center;
  gap: 0.4rem;
  font-size: 0.76rem;
  font-weight: 600;
  color: var(--accent-chip-text);
  background: var(--accent-chip-bg);
  border: 1px solid rgba(59, 130, 246, 0.3);
  padding: 0.28rem 0.75rem;
  border-radius: 999px;
  margin-bottom: 1rem;
}

.empty-hero h2 {
  font-size: 1.5rem;
  letter-spacing: -0.02em;
  line-height: 1.25;
  margin: 0 0 0.75rem 0;
  color: var(--text-primary);
}

.empty-hero p {
  color: var(--text-secondary);
  font-size: 0.94rem;
  max-width: 66ch;
  margin: 0;
}

.empty-steps {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 0.85rem;
  margin: 1.65rem 0 0.5rem 0;
}

@media (max-width: 900px) {
  .empty-steps {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}

.es-step {
  background: var(--bg-primary);
  border: 1px solid var(--border-subtle);
  border-radius: 12px;
  padding: 0.95rem;
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
}

.es-step span {
  font-size: 0.72rem;
  font-weight: 700;
  color: var(--accent-blue);
}

.es-step strong {
  font-size: 0.9rem;
  color: var(--text-primary);
}

.es-step small {
  font-size: 0.77rem;
  color: var(--text-secondary);
}

/* Radial SVG Score Ring */
.verdict-card {
  display: flex;
  align-items: center;
  gap: 1.5rem;
  flex-wrap: wrap;
}

.radial-wrap {
  position: relative;
  width: 118px;
  height: 118px;
  flex-shrink: 0;
}

.radial-svg {
  width: 100%;
  height: 100%;
  transform: rotate(-90deg);
}

.radial-track {
  fill: none;
  stroke: var(--bg-elevated);
  stroke-width: 9;
}

.radial-progress {
  fill: none;
  stroke-width: 9;
  stroke-linecap: round;
}

.radial-progress.good { stroke: var(--status-good); }
.radial-progress.warn { stroke: var(--status-warn); }
.radial-progress.bad { stroke: var(--status-bad); }

.radial-center {
  position: absolute;
  inset: 0;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
}

.score-num {
  font-size: 1.9rem;
  font-weight: 800;
  line-height: 1;
  color: var(--text-primary);
}

.score-label {
  font-size: 0.66rem;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--text-secondary);
  margin-top: 0.25rem;
}

.verdict-details {
  flex: 1;
  min-width: 240px;
}

.verdict-badge {
  display: inline-flex;
  align-items: center;
  gap: 0.4rem;
  padding: 0.3rem 0.75rem;
  border-radius: 999px;
  font-size: 0.78rem;
  font-weight: 700;
  letter-spacing: 0.03em;
  margin-bottom: 0.5rem;
}

.verdict-badge.good { background: var(--status-good-bg); color: var(--status-good-text); }
.verdict-badge.warn { background: var(--status-warn-bg); color: var(--status-warn-text); }
.verdict-badge.bad { background: var(--status-bad-bg); color: var(--status-bad-text); }

.verdict-details h2 {
  font-size: 1.05rem;
  margin: 0 0 0.35rem 0;
  word-break: break-all;
  color: var(--text-primary);
}

.verdict-details p {
  font-size: 0.9rem;
  color: var(--text-secondary);
  margin: 0;
}

/* 4-Pillar Grid */
.pillars-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 0.85rem;
}

@media (max-width: 900px) {
  .pillars-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}

.pillar-card {
  background: var(--bg-elevated);
  border: 1px solid var(--border-subtle);
  border-radius: 10px;
  padding: 0.85rem;
}

.pillar-top {
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-size: 0.76rem;
  color: var(--text-secondary);
}

.pillar-top small { color: var(--text-muted); }

.pillar-val-row { margin: 0.35rem 0 0.5rem; }

.pillar-num {
  font-size: 1.25rem;
  font-weight: 700;
}

.pillar-num.good { color: var(--status-good-text); }
.pillar-num.warn { color: var(--status-warn-text); }
.pillar-num.bad { color: var(--status-bad-text); }

/* Pillar Legend Sidebar */
.pillar-legend {
  display: flex;
  flex-direction: column;
  gap: 0.65rem;
  font-size: 0.82rem;
  color: var(--text-secondary);
}

.pillar-legend-row {
  display: flex;
  align-items: center;
  gap: 0.55rem;
}

.pillar-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  flex-shrink: 0;
}

.pillar-legend strong { color: var(--text-primary); }

/* Recommendations */
.pass-banner {
  background: var(--status-good-bg);
  border: 1px solid rgba(16, 185, 129, 0.32);
  border-radius: 10px;
  padding: 1rem;
  color: var(--status-good-text);
  font-size: 0.9rem;
  display: flex;
  align-items: center;
  gap: 0.65rem;
}

.rec-item {
  background: var(--bg-elevated);
  border: 1px solid var(--border-subtle);
  border-left: 4px solid var(--status-bad);
  border-radius: 10px;
  padding: 1rem 1.1rem;
  margin-bottom: 0.9rem;
}

.rec-item.MEDIUM { border-left-color: var(--status-warn); }

.rec-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 0.75rem;
  flex-wrap: wrap;
  margin-bottom: 0.4rem;
}

.rec-header h4 {
  margin: 0;
  font-size: 0.96rem;
  font-weight: 600;
  color: var(--text-primary);
}

.tag {
  font-size: 0.72rem;
  padding: 0.2rem 0.55rem;
  border-radius: 6px;
  background: var(--bg-primary);
  color: var(--text-secondary);
  border: 1px solid var(--border-subtle);
  margin-left: 0.35rem;
}

.rec-impact {
  font-size: 0.85rem;
  color: var(--text-secondary);
  margin: 0 0 0.65rem 0;
}

.rec-steps {
  margin: 0;
  padding-left: 1.2rem;
  font-size: 0.86rem;
  color: var(--text-primary);
}

.rec-steps li { margin-bottom: 0.3rem; }

/* Probability Bars */
.prob-row {
  display: grid;
  grid-template-columns: 185px 1fr 56px;
  align-items: center;
  gap: 0.75rem;
  font-size: 0.83rem;
  margin-bottom: 0.62rem;
}

.prob-name {
  color: var(--text-secondary);
  text-transform: capitalize;
}

.mono-feat {
  text-transform: none;
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
  font-size: 0.78rem;
}

.prob-track {
  height: 9px;
  background: var(--bg-primary);
  border: 1px solid var(--border-subtle);
  border-radius: 999px;
  overflow: hidden;
}

.prob-fill {
  height: 100%;
  border-radius: 999px;
  background: var(--accent-blue);
}

.prob-fill.clean { background: var(--status-good); }
.prob-fill.warn-fill { background: var(--status-warn); }
.prob-fill.alert { background: var(--status-bad); }

.prob-val {
  text-align: right;
  font-variant-numeric: tabular-nums;
  color: var(--text-primary);
  font-weight: 500;
}

/* Telemetry Grid */
.metrics-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 0.75rem;
}

@media (max-width: 900px) {
  .metrics-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}

.metric-tile {
  background: var(--bg-elevated);
  border: 1px solid var(--border-subtle);
  border-radius: 10px;
  padding: 0.75rem 0.85rem;
}

.metric-top {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 0.4rem;
  margin-bottom: 0.25rem;
}

.metric-tile span {
  font-size: 0.74rem;
  color: var(--text-secondary);
}

.metric-cat {
  font-size: 0.65rem;
  color: var(--accent-chip-text);
  background: var(--accent-chip-bg);
  padding: 0.08rem 0.38rem;
  border-radius: 4px;
  font-weight: 600;
}

.metric-tile strong {
  font-size: 1.02rem;
  font-variant-numeric: tabular-nums;
  color: var(--text-primary);
}

/* How It Works Styles */
.hiw-hero {
  background: var(--hero-grad);
  border: 1px solid var(--border-subtle);
  border-radius: 14px;
  padding: 2rem;
  margin-bottom: 1.25rem;
  box-shadow: var(--card-shadow);
}

.hiw-badge {
  display: inline-flex;
  align-items: center;
  gap: 0.45rem;
  padding: 0.3rem 0.75rem;
  border-radius: 999px;
  background: var(--accent-chip-bg);
  border: 1px solid rgba(59, 130, 246, 0.35);
  color: var(--accent-chip-text);
  font-size: 0.76rem;
  font-weight: 600;
  margin-bottom: 0.9rem;
}

.hiw-hero h1 {
  font-size: 2rem;
  letter-spacing: -0.025em;
  line-height: 1.2;
  margin: 0 0 0.75rem 0;
  color: var(--text-primary);
}

.gradient-text {
  background: linear-gradient(90deg, #2563eb, #0891b2);
  -webkit-background-clip: text;
  background-clip: text;
  color: transparent;
}

.hiw-hero-sub {
  color: var(--text-secondary);
  font-size: 0.96rem;
  max-width: 78ch;
  margin: 0;
}

.hiw-kpi-strip {
  display: grid;
  grid-template-columns: repeat(5, minmax(0, 1fr));
  gap: 0.85rem;
  margin-top: 1.5rem;
  padding-top: 1.4rem;
  border-top: 1px solid var(--border-subtle);
}

@media (max-width: 960px) {
  .hiw-kpi-strip {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}

.kpi-tile {
  background: var(--bg-primary);
  border: 1px solid var(--border-subtle);
  border-radius: 12px;
  padding: 0.9rem 1rem;
}

.kpi-label { display: block; font-size: 0.75rem; color: var(--text-secondary); }
.kpi-value { display: block; font-size: 1.45rem; font-weight: 800; margin: 0.15rem 0; color: var(--text-primary); }
.kpi-sub { font-size: 0.74rem; color: var(--accent-blue); font-weight: 500; }

.ps-card {
  background: var(--bg-card);
  border: 1px solid var(--border-subtle);
  border-radius: 14px;
  padding: 1.35rem;
  height: 100%;
  box-shadow: var(--card-shadow);
}

.ps-card.problem { border-top: 3px solid var(--status-bad); }
.ps-card.solution { border-top: 3px solid var(--status-good); }

.ps-eyebrow {
  display: inline-flex;
  align-items: center;
  gap: 0.4rem;
  font-size: 0.74rem;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--text-secondary);
  font-weight: 600;
}

.ps-card h2 {
  font-size: 1.12rem;
  margin: 0.25rem 0 1rem 0;
  color: var(--text-primary);
}

.ps-item {
  background: var(--bg-elevated);
  border: 1px solid var(--border-subtle);
  border-radius: 10px;
  padding: 0.9rem 1rem;
  margin-bottom: 0.8rem;
}

.ps-item h4 { font-size: 0.92rem; margin: 0 0 0.3rem 0; color: var(--text-primary); }
.ps-item p { font-size: 0.83rem; color: var(--text-secondary); margin: 0; }

.section-desc {
  font-size: 0.85rem;
  color: var(--text-secondary);
  margin: -0.5rem 0 1rem 0;
}

.gif-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1.15rem;
}

@media (max-width: 900px) {
  .gif-grid { grid-template-columns: 1fr; }
}

.gif-card {
  background: var(--bg-elevated);
  border: 1px solid var(--border-subtle);
  border-radius: 12px;
  overflow: hidden;
}

.gif-card img {
  width: 100%;
  display: block;
  background: #0a0e17;
}

.gif-caption {
  padding: 0.9rem 1rem;
}

.gif-caption strong {
  display: block;
  font-size: 0.9rem;
  color: var(--text-primary);
  margin-bottom: 0.25rem;
}

.gif-caption span {
  font-size: 0.8rem;
  color: var(--text-secondary);
}

.stage-detail-card {
  background: var(--bg-elevated);
  border: 1px solid rgba(59, 130, 246, 0.35);
  border-radius: 12px;
  padding: 1.2rem;
  margin: 0.75rem 0 1.15rem 0;
}

.stage-detail-head {
  display: flex;
  align-items: center;
  gap: 0.9rem;
  flex-wrap: wrap;
  margin-bottom: 0.7rem;
}

.stage-badge-lg {
  background: linear-gradient(135deg, #2563eb, #0284c7);
  color: #ffffff;
  font-size: 0.78rem;
  font-weight: 700;
  padding: 0.35rem 0.75rem;
  border-radius: 8px;
}

.stage-detail-head h3 {
  font-size: 1.06rem;
  margin: 0 0 0.2rem 0;
  color: var(--text-primary);
}

.script-chip {
  font-size: 0.76rem;
  color: var(--accent-chip-text);
  background: var(--bg-primary);
  padding: 0.15rem 0.5rem;
  border-radius: 6px;
  border: 1px solid var(--border-subtle);
}

.stage-detail-summary {
  font-size: 0.88rem;
  color: var(--text-secondary);
  margin: 0 0 0.85rem 0;
}

.stage-highlights {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
}

.hl-chip {
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
  background: var(--bg-primary);
  border: 1px solid var(--border-subtle);
  padding: 0.3rem 0.65rem;
  border-radius: 999px;
  font-size: 0.76rem;
  color: var(--status-good-text);
  font-weight: 500;
}

.timeline-all-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 0.85rem;
}

@media (max-width: 900px) {
  .timeline-all-grid { grid-template-columns: 1fr; }
}

.timeline-mini-card {
  background: var(--bg-primary);
  border: 1px solid var(--border-subtle);
  border-radius: 10px;
  padding: 0.95rem;
}

.timeline-mini-card.selected {
  border-color: var(--accent-blue);
  background: var(--accent-chip-bg);
}

.tm-top {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 0.45rem;
}

.tm-id { font-size: 0.74rem; font-weight: 700; color: var(--accent-blue); }
.timeline-mini-card h4 { font-size: 0.88rem; margin: 0 0 0.35rem 0; color: var(--text-primary); }
.timeline-mini-card p { font-size: 0.78rem; color: var(--text-secondary); margin: 0; }

.taxonomy-grid {
  display: grid;
  grid-template-columns: repeat(5, minmax(0, 1fr));
  gap: 0.85rem;
  margin-bottom: 1.35rem;
}

@media (max-width: 1100px) {
  .taxonomy-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}

.tax-card {
  background: var(--bg-elevated);
  border: 1px solid var(--border-subtle);
  border-radius: 10px;
  padding: 0.9rem;
}

.tax-card h4 { font-size: 0.85rem; margin: 0 0 0.35rem 0; color: var(--accent-chip-text); }
.tax-items { font-size: 0.78rem; color: var(--text-primary); margin: 0 0 0.45rem 0; }
.tax-challenge { display: block; font-size: 0.74rem; color: var(--text-secondary); }

.table-scroll {
  overflow-x: auto;
  border: 1px solid var(--border-subtle);
  border-radius: 10px;
}

.data-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.8rem;
  text-align: left;
}

.data-table th {
  background: var(--bg-elevated);
  color: var(--text-secondary);
  font-weight: 600;
  padding: 0.65rem 0.8rem;
  border-bottom: 1px solid var(--border-subtle);
  white-space: nowrap;
}

.data-table td {
  padding: 0.6rem 0.8rem;
  border-bottom: 1px solid var(--border-subtle);
  color: var(--text-secondary);
  font-variant-numeric: tabular-nums;
  white-space: nowrap;
}

.data-table tr.winner-row { background: var(--status-good-bg); }
.mono-strong { color: var(--text-primary) !important; font-weight: 600; }

.win-pill {
  margin-left: 0.45rem;
  font-size: 0.66rem;
  padding: 0.14rem 0.42rem;
  border-radius: 999px;
  background: var(--status-good-bg);
  color: var(--status-good-text);
  border: 1px solid rgba(16, 185, 129, 0.35);
}

.winner-cards-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1rem;
  margin-bottom: 1.15rem;
}

@media (max-width: 820px) {
  .winner-cards-grid { grid-template-columns: 1fr; }
}

.winner-card {
  background: var(--bg-elevated);
  border: 1px solid var(--border-subtle);
  border-radius: 12px;
  padding: 1.1rem;
}

.winner-tag {
  font-size: 0.72rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  color: var(--accent-blue);
}

.winner-tag.green { color: var(--status-good-text); }
.winner-card h3 { font-size: 1.15rem; margin: 0.25rem 0 0.3rem 0; color: var(--text-primary); }
.winner-sub { font-size: 0.79rem; color: var(--text-secondary); margin: 0 0 0.85rem 0; }

.winner-metrics {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 0.5rem;
  padding-top: 0.75rem;
  border-top: 1px solid var(--border-subtle);
}

.winner-metrics span { display: block; font-size: 0.7rem; color: var(--text-secondary); }
.winner-metrics strong { font-size: 0.95rem; color: var(--text-primary); }

.output-card {
  background: var(--bg-elevated);
  border: 1px solid var(--border-subtle);
  border-radius: 12px;
  overflow: hidden;
  margin-bottom: 0.75rem;
}

.output-card img {
  width: 100%;
  height: 220px;
  object-fit: cover;
  object-position: top;
  display: block;
  border-bottom: 1px solid var(--border-subtle);
}

.output-body { padding: 1rem; }
.output-body h4 { font-size: 0.88rem; margin: 0.45rem 0; color: var(--text-primary); word-break: break-all; }
.output-body p { font-size: 0.8rem; color: var(--text-secondary); margin: 0; }

/* Streamlit Native Widget Overrides (adapts to both Dark & Light variables) */
div[data-testid="stButton"] > button[kind="primary"] {
  background: linear-gradient(135deg, #2563eb, #0284c7) !important;
  border: none !important;
  border-radius: 10px !important;
  font-weight: 600 !important;
  padding: 0.62rem 1rem !important;
  color: #ffffff !important;
}

div[data-testid="stButton"] > button[kind="secondary"] {
  background-color: var(--bg-card) !important;
  border: 1px solid var(--border-subtle) !important;
  border-radius: 10px !important;
  color: var(--text-primary) !important;
  font-weight: 500 !important;
}

div[data-testid="stButton"] > button[kind="secondary"]:hover {
  border-color: var(--accent-blue) !important;
  color: var(--accent-blue) !important;
}

div[data-testid="stFileUploader"] section {
  background-color: var(--bg-card) !important;
  border: 2px dashed var(--border-subtle) !important;
  border-radius: 12px !important;
  color: var(--text-primary) !important;
}

div[data-baseweb="select"] > div {
  background-color: var(--bg-card) !important;
  border-color: var(--border-subtle) !important;
  color: var(--text-primary) !important;
}

div[data-baseweb="tab-list"] {
  background-color: var(--bg-card) !important;
  border: 1px solid var(--border-subtle) !important;
  border-radius: 10px !important;
  padding: 0.25rem !important;
}

button[data-baseweb="tab"] {
  color: var(--text-secondary) !important;
}

button[data-baseweb="tab"][aria-selected="true"] {
  color: var(--text-primary) !important;
}

div[data-testid="stExpander"] {
  background-color: var(--bg-card) !important;
  border: 1px solid var(--border-subtle) !important;
  border-radius: 10px !important;
}
""").strip()

    return f"<style>\n{vars_block}\n{base_css}\n</style>"


# ---------------------------------------------------------------------------
# Helpers & Cached Loaders
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="Loading PreFlight-3D ML Ensemble...")
def load_analyzer() -> PreFlightAnalyzer:
    return PreFlightAnalyzer()


@st.cache_data
def encode_image_data_uri(file_path: str) -> str:
    """Encode local GIF/PNG from frontend/public/ into a data URI for HTML cards."""
    if not os.path.exists(file_path):
        return ""
    ext = os.path.splitext(file_path)[1].lower().lstrip(".")
    mime = "image/gif" if ext == "gif" else ("image/jpeg" if ext in ("jpg", "jpeg") else "image/png")
    with open(file_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def get_available_samples() -> list[dict]:
    """Load sample videos from sample_videos/ and local dataset_manifest.csv."""
    items = []
    sample_dir = "sample_videos"
    if os.path.isdir(sample_dir):
        for fname in sorted(os.listdir(sample_dir)):
            if fname.lower().endswith((".mp4", ".mov", ".mkv", ".avi", ".webm")):
                defect = "none" if "clean" in fname.lower() else fname.replace("defect_", "").replace(".mp4", "")
                items.append({
                    "id": f"demo_{fname}",
                    "label": f"[Demo Clip] {fname}",
                    "path": os.path.join(sample_dir, fname),
                    "defect_type": defect,
                })

    manifest_path = "data/processed/quality_measurements/dataset_manifest.csv"
    if os.path.exists(manifest_path):
        df = pd.read_csv(manifest_path)
        subset = df.groupby("defect_type", group_keys=False).head(2)
        for _, row in subset.iterrows():
            if os.path.exists(row["video_path"]):
                items.append({
                    "id": str(row["sample_id"]),
                    "label": f"[{row['readiness_label']}] {row['category']} - {row['defect_type']} ({row['sample_id']})",
                    "path": row["video_path"],
                    "defect_type": str(row["defect_type"]),
                })
    return items


def compute_pillar_scores(m: dict) -> list[dict]:
    """Exact port of computePillarScores() from frontend/src/App.jsx."""
    if not m:
        return []
    sharpness = round(
        min(1.0, (m.get("laplacian_var_mean", 0) / 150.0))
        * (1.0 - min(1.0, m.get("blurry_frame_ratio", 0)))
        * 100
    )
    tracking = round(
        (
            0.65 * min(1.0, m.get("pair_inlier_ratio_mean", 0) / 0.65)
            + 0.35 * min(1.0, m.get("orb_keypoints_mean", 0) / 250.0)
        )
        * 100
    )
    b = m.get("brightness_mean", 0)
    if 75 <= b <= 190:
        bright_comfort = 1.0
    elif b < 75:
        bright_comfort = max(0.0, b / 75.0)
    else:
        bright_comfort = max(0.0, (255.0 - b) / 65.0)

    glare_comfort = max(0.0, 1.0 - m.get("specular_glare_ratio", 0) * 3.0)
    flicker_comfort = max(0.0, 1.0 - max(0.0, (m.get("brightness_step_diff_mean", 0) - 8.0) / 28.0))
    lighting = round(bright_comfort * glare_comfort * flicker_comfort * 100)

    duration_factor = min(1.0, m.get("duration_sec", 0) / 12.0)
    coverage_factor = max(
        min(1.0, m.get("loop_closure_inlier_ratio", 0) / 0.45),
        min(1.0, m.get("cumulative_pan_px", 0) / 30.0),
    )
    orbit_raw = 0.5 * duration_factor + 0.5 * coverage_factor
    if m.get("loop_closure_inlier_ratio", 0) >= 0.5 and m.get("optical_flow_mean", 0) >= 2.0:
        orbit_raw = max(orbit_raw, coverage_factor * 0.95)
    orbit = round(orbit_raw * 100)

    return [
        {"key": "Focus", "label": "Focus & Sharpness", "score": sharpness, "weight": "25%"},
        {"key": "Track", "label": "Tracking & Overlap", "score": tracking, "weight": "30%"},
        {"key": "Light", "label": "Lighting & Glare", "score": lighting, "weight": "25%"},
        {"key": "Orbit", "label": "360° Orbit Coverage", "score": orbit, "weight": "20%"},
    ]


def get_verdict_tier(verdict: str) -> str:
    if verdict == "READY_FOR_RECONSTRUCTION":
        return "good"
    if verdict == "USABLE_WITH_WARNINGS":
        return "warn"
    return "bad"


def format_defect_name(code: str) -> str:
    return "Clean Capture (No Defect)" if code == "none" else code.replace("_", " ")


# ---------------------------------------------------------------------------
# View 1: Pre-Flight Analyzer (Matches App.jsx)
# ---------------------------------------------------------------------------


def render_analyzer_results(result: dict):
    tier = get_verdict_tier(result["verdict"])
    badge_icon = ICONS["check"] if tier == "good" else (ICONS["alert"] if tier == "warn" else ICONS["xcircle"])
    verdict_label = html.escape(result["verdict"].replace("_", " "))
    video_name = html.escape(result["video_file"])
    summary_text = html.escape(result["summary"])

    score = max(0, min(100, int(result["readiness_score"])))
    radius = 48
    circumference = 2 * 3.14159265 * radius
    offset = circumference - (score / 100.0) * circumference

    st.markdown(
        f'<div class="pf-card verdict-card">'
        f'<div class="radial-wrap {tier}">'
        f'<svg class="radial-svg" viewBox="0 0 120 120">'
        f'<circle class="radial-track" cx="60" cy="60" r="{radius}" />'
        f'<circle class="radial-progress {tier}" cx="60" cy="60" r="{radius}" '
        f'stroke-dasharray="{circumference:.2f}" stroke-dashoffset="{offset:.2f}" />'
        f'</svg>'
        f'<div class="radial-center">'
        f'<span class="score-num">{score}</span>'
        f'<span class="score-label">Readiness</span>'
        f'</div></div>'
        f'<div class="verdict-details">'
        f'<div class="verdict-badge {tier}">{badge_icon}<span>{verdict_label}</span></div>'
        f'<h2>{video_name}</h2>'
        f'<p>{summary_text}</p>'
        f'</div></div>',
        unsafe_allow_html=True,
    )

    # 4-Pillar Capture Health Breakdown
    pillars = compute_pillar_scores(result.get("extracted_metrics", {}))
    pillar_cards = []
    for p in pillars:
        p_tier = "good" if p["score"] >= 70 else ("warn" if p["score"] >= 45 else "bad")
        fill_cls = "clean" if p_tier == "good" else ("alert" if p_tier == "bad" else "warn-fill")
        pillar_cards.append(
            f'<div class="pillar-card">'
            f'<div class="pillar-top"><span>{html.escape(p["label"])}</span><small>Wt {p["weight"]}</small></div>'
            f'<div class="pillar-val-row"><strong class="pillar-num {p_tier}">{p["score"]}%</strong></div>'
            f'<div class="prob-track"><div class="prob-fill {fill_cls}" style="width:{p["score"]}%;"></div></div>'
            f'</div>'
        )
    st.markdown(
        f'<div class="pf-card">'
        f'<div class="pf-card-title"><span class="pf-icon-wrap">{ICONS["gauge"]}</span><span>4-Pillar Capture Health Breakdown</span></div>'
        f'<div class="pillars-grid">{"".join(pillar_cards)}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Actionable Capture Recommendations
    recs = result.get("recommendations", [])
    if not recs:
        rec_body = (
            f'<div class="pass-banner">'
            f'{ICONS["check"]}'
            f'<span>All quality gates passed. Frame sharpness, uniform lighting, and 360° orbit overlap are strong enough for 3D reconstruction.</span>'
            f'</div>'
        )
    else:
        items_html = []
        for rec in recs:
            sev = html.escape(rec["severity"])
            title = html.escape(rec["title"])
            cat = html.escape(rec["category"])
            det = html.escape(rec["detected_by"])
            conf_pct = round(rec["confidence"] * 100)
            impact = html.escape(rec["impact"])
            steps = "".join(f"<li>{html.escape(s)}</li>" for s in rec["action_steps"])
            items_html.append(
                f'<div class="rec-item {sev}">'
                f'<div class="rec-header"><h4>{title}</h4>'
                f'<div><span class="tag">{cat}</span><span class="tag">{det} ({conf_pct}%)</span></div></div>'
                f'<p class="rec-impact">{impact}</p>'
                f'<ol class="rec-steps">{steps}</ol>'
                f'</div>'
            )
        rec_body = "".join(items_html)

    st.markdown(
        f'<div class="pf-card">'
        f'<div class="pf-card-title"><span class="pf-icon-wrap">{ICONS["alert"]}</span><span>Actionable Capture Recommendations</span></div>'
        f'{rec_body}'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Soft-Voting Ensemble Diagnosis Probabilities
    sorted_probs = sorted(result["defect_probabilities"].items(), key=lambda x: x[1], reverse=True)
    prob_rows = []
    for cls, prob in sorted_probs:
        pct = round(prob * 1000) / 10
        fill_cls = "clean" if cls == "none" else ("alert" if prob >= 0.25 else "")
        prob_rows.append(
            f'<div class="prob-row">'
            f'<span class="prob-name">{html.escape(format_defect_name(cls))}</span>'
            f'<div class="prob-track"><div class="prob-fill {fill_cls}" style="width:{min(pct, 100)}%;"></div></div>'
            f'<span class="prob-val">{pct:.1f}%</span>'
            f'</div>'
        )
    st.markdown(
        f'<div class="pf-card">'
        f'<div class="pf-card-title"><span class="pf-icon-wrap">{ICONS["activity"]}</span><span>Soft-Voting Ensemble Diagnosis Probabilities</span></div>'
        f'{"".join(prob_rows)}'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Extracted CV & Trajectory Telemetry Grid
    metrics = result.get("extracted_metrics", {})
    tiles = []
    for key, label, unit, is_pct, cat in METRIC_LABELS:
        val = metrics.get(key)
        if val is None:
            disp = "-"
        elif is_pct:
            disp = f"{val * 100:.1f}{unit}"
        else:
            disp = f"{val}{unit}"
        tiles.append(
            f'<div class="metric-tile">'
            f'<div class="metric-top"><span>{html.escape(label)}</span><small class="metric-cat">{html.escape(cat)}</small></div>'
            f'<strong>{html.escape(disp)}</strong>'
            f'</div>'
        )
    st.markdown(
        f'<div class="pf-card">'
        f'<div class="pf-card-title"><span class="pf-icon-wrap">{ICONS["sliders"]}</span><span>Extracted Computer Vision &amp; Trajectory Telemetry</span></div>'
        f'<div class="metrics-grid">{"".join(tiles)}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_analyzer_page(analyzer: PreFlightAnalyzer):
    samples = get_available_samples()
    left_col, right_col = st.columns([1.05, 2.15], gap="large")

    with left_col:
        st.markdown(
            f'<div class="pf-card" style="margin-bottom:0.55rem; padding:0.95rem 1.2rem;">'
            f'<div class="pf-card-title" style="margin-bottom:0;"><span class="pf-icon-wrap">{ICONS["film"]}</span><span>Capture Input</span></div></div>',
            unsafe_allow_html=True,
        )
        tab_upload, tab_sample = st.tabs(["Upload Video", "Test Dataset"])

        with tab_upload:
            uploaded_file = st.file_uploader(
                "Click to select or drag & drop a capture video",
                type=["mp4", "mov", "mkv", "avi", "webm"],
                help="Supports MP4, MOV, MKV, AVI, WEBM",
            )
            if uploaded_file is not None:
                st.video(uploaded_file)
                size_mb = uploaded_file.size / (1024 * 1024)
                st.caption(f"File: `{uploaded_file.name}` ({size_mb:.2f} MB)")

                if st.button("Run Pre-Flight Check", type="primary", use_container_width=True):
                    ext = os.path.splitext(uploaded_file.name)[1].lower() or ".mp4"
                    tmp_path = None
                    try:
                        with st.spinner("Analyzing Frames & Geometry..."):
                            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                                tmp.write(uploaded_file.getbuffer())
                                tmp_path = tmp.name
                            res = analyzer.analyze_video(tmp_path)
                            res["video_file"] = uploaded_file.name
                            st.session_state.analysis_result = res
                    except Exception as exc:
                        st.error(f"Failed to analyze video: {exc}")
                    finally:
                        if tmp_path and os.path.exists(tmp_path):
                            os.remove(tmp_path)

        with tab_sample:
            if not samples:
                st.info("No benchmark samples found. Add `.mp4` clips to `sample_videos/`.")
            else:
                labels = [s["label"] for s in samples]
                chosen_idx = st.selectbox(
                    "Select a uCO3D Benchmark Video (Clean or Synthetic Defect):",
                    options=range(len(samples)),
                    format_func=lambda i: labels[i],
                )
                chosen_sample = samples[chosen_idx]
                st.video(chosen_sample["path"])
                st.caption(
                    "Runs live 19-feature CV extraction and Soft-Voting Ensemble + Physics Gate inference."
                )
                if st.button("Analyze Benchmark Sample", type="primary", use_container_width=True):
                    try:
                        with st.spinner("Running Live Inference..."):
                            st.session_state.analysis_result = analyzer.analyze_video(chosen_sample["path"])
                    except Exception as exc:
                        st.error(f"Failed to analyze sample: {exc}")

        # Instant 1-Click Benchmark Presets
        preset_map = {}
        for s in samples:
            dt = s["defect_type"]
            if dt in ("none", "motion_blur", "harsh_glare", "incomplete_orbit") and dt not in preset_map:
                preset_map[dt] = s

        if preset_map:
            st.markdown(
                '<div style="font-size:0.78rem; color:var(--text-secondary); margin:0.9rem 0 0.4rem 0; font-weight:500;">'
                'Instant 1-Click Benchmark Test:</div>',
                unsafe_allow_html=True,
            )
            p_cols = st.columns(2)
            ordered_keys = [k for k in ("none", "motion_blur", "harsh_glare", "incomplete_orbit") if k in preset_map]
            for idx, dt in enumerate(ordered_keys):
                s = preset_map[dt]
                btn_label = "Clean Sample" if dt == "none" else dt.replace("_", " ").title()
                with p_cols[idx % 2]:
                    if st.button(btn_label, key=f"preset_{dt}", use_container_width=True):
                        with st.spinner(f"Running {btn_label}..."):
                            st.session_state.analysis_result = analyzer.analyze_video(s["path"])
                        st.rerun()

        # 4 Inspection Pillars Sidebar Card
        st.markdown(
            f'<div class="pf-card" style="margin-top:1rem;">'
            f'<div class="pf-card-title"><span class="pf-icon-wrap">{ICONS["cpu"]}</span><span>4 Inspection Pillars</span></div>'
            f'<div class="pillar-legend">'
            f'<div class="pillar-legend-row"><span class="pillar-dot" style="background:#3b82f6;"></span><span><strong>Sharpness:</strong> Laplacian &amp; Canny edges</span></div>'
            f'<div class="pillar-legend-row"><span class="pillar-dot" style="background:#f59e0b;"></span><span><strong>Exposure:</strong> Brightness, flicker &amp; glare</span></div>'
            f'<div class="pillar-legend-row"><span class="pillar-dot" style="background:#10b981;"></span><span><strong>Texture:</strong> ORB keypoints &amp; RANSAC matches</span></div>'
            f'<div class="pillar-legend-row"><span class="pillar-dot" style="background:#8b5cf6;"></span><span><strong>360° Orbit:</strong> Optical flow &amp; loop closure</span></div>'
            f'</div></div>',
            unsafe_allow_html=True,
        )
        if st.button("Read Full A-Z Architecture & Benchmarks", use_container_width=True):
            st.session_state.page = "how-it-works"
            st.rerun()

    with right_col:
        if st.session_state.analysis_result is None:
            st.markdown(
                f'<div class="empty-hero">'
                f'<div class="empty-badge">{ICONS["sparkles"]}<span>Ready for Pre-Flight Inspection</span></div>'
                f'<h2>Catch 3D Reconstruction Failures Before Running COLMAP or 3DGS</h2>'
                f'<p>Upload an object-centric orbit video or click any benchmark preset on the left to inspect motion blur, lighting stability, specular glare, and 360° loop closure in under a second.</p>'
                f'<div class="empty-steps">'
                f'<div class="es-step"><span>01</span><strong>Sample @ 2.5 FPS</strong><small>Elliptical subject ROI masking at 480px</small></div>'
                f'<div class="es-step"><span>02</span><strong>19 CV Features</strong><small>Sharpness, glare, ORB, Farneback &amp; RANSAC</small></div>'
                f'<div class="es-step"><span>03</span><strong>Ensemble + Physics</strong><small>SoftVoting + XGBoost_Tuned + guardrails</small></div>'
                f'<div class="es-step"><span>04</span><strong>0-100 Readiness</strong><small>Actionable fixes before GPU compute</small></div>'
                f'</div></div>',
                unsafe_allow_html=True,
            )
            st.markdown("<div style='height:0.65rem;'></div>", unsafe_allow_html=True)
            if st.button("Explore How It Works (A-Z) & Model Results", type="secondary"):
                st.session_state.page = "how-it-works"
                st.rerun()
        else:
            render_analyzer_results(st.session_state.analysis_result)


# ---------------------------------------------------------------------------
# View 2: How It Works (A-to-Z Architecture & Benchmarks — Matches HowItWorks.jsx)
# ---------------------------------------------------------------------------


def render_how_it_works_page():
    kpi_html = "".join(
        f'<div class="kpi-tile">'
        f'<span class="kpi-label">{html.escape(s["label"])}</span>'
        f'<strong class="kpi-value">{html.escape(s["value"])}</strong>'
        f'<small class="kpi-sub">{html.escape(s["sub"])}</small>'
        f'</div>'
        for s in HERO_STATS
    )
    st.markdown(
        f'<div class="hiw-hero">'
        f'<div class="hiw-badge">{ICONS["sparkles"]}<span>A-to-Z Engineering Architecture &amp; Benchmark Report</span></div>'
        f'<h1>Pre-Flight Quality Gate for <span class="gradient-text">3D Reconstruction</span></h1>'
        f'<p class="hiw-hero-sub">PreFlight-3D evaluates smartphone and turntable object-capture videos in ~0.5 seconds on CPU — catching motion blur, underexposure, specular glare, exposure flicker, sparse overlap, and incomplete 360° orbits before you commit heavy compute to COLMAP, NeRF, or 3D Gaussian Splatting.</p>'
        f'<div class="hiw-kpi-strip">{kpi_html}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    if st.button("Try Live Analyzer", type="primary"):
        st.session_state.page = "analyzer"
        st.rerun()

    st.markdown("<div style='height:0.75rem;'></div>", unsafe_allow_html=True)

    # Problem vs Solution 2-Column Grid
    col_prob, col_sol = st.columns(2, gap="large")
    with col_prob:
        prob_items = "".join(
            f'<div class="ps-item"><h4>{html.escape(i["title"])}</h4><p>{html.escape(i["detail"])}</p></div>'
            for i in PROBLEMS_LIST
        )
        st.markdown(
            f'<div class="ps-card problem">'
            f'<span class="ps-eyebrow" style="color:var(--status-bad-text);">{ICONS["alert"]}<span>The Problem</span></span>'
            f'<h2>Why 3D Reconstructions Fail Silently</h2>'
            f'{prob_items}'
            f'</div>',
            unsafe_allow_html=True,
        )
    with col_sol:
        sol_items = "".join(
            f'<div class="ps-item"><h4>{html.escape(i["title"])}</h4><p>{html.escape(i["detail"])}</p></div>'
            for i in SOLUTIONS_LIST
        )
        st.markdown(
            f'<div class="ps-card solution">'
            f'<span class="ps-eyebrow" style="color:var(--status-good-text);">{ICONS["shield"]}<span>Our Solution</span></span>'
            f'<h2>Hybrid CV + Ensemble + Physics Gate</h2>'
            f'{sol_items}'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:1.1rem;'></div>", unsafe_allow_html=True)

    # Visual Dataset & Failure Simulation Showcase (GIFs from frontend/public/)
    gif_samples_uri = encode_image_data_uri("frontend/public/selected_samples_showcase.gif")
    gif_defects_uri = encode_image_data_uri("frontend/public/defect_modes_comparison.gif")
    if gif_samples_uri and gif_defects_uri:
        st.markdown(
            f'<div class="pf-card">'
            f'<div class="pf-card-title"><span class="pf-icon-wrap">{ICONS["eye"]}</span><span>Visual Dataset &amp; Failure Simulation Showcase</span></div>'
            f'<p class="section-desc">Real 360° captures curated from Meta\'s uCO3D dataset alongside live PreFlight-3D inference scores across all 7 diagnostic classes.</p>'
            f'<div class="gif-grid">'
            f'<div class="gif-card">'
            f'<img src="{gif_samples_uri}" alt="Selected uCO3D Benchmark Captures Showcase" />'
            f'<div class="gif-caption"><strong>Selected uCO3D Object Taxonomy (6 Clean Samples)</strong>'
            f'<span>Covers Glossy (Vase), Metallic (Fire Extinguisher), Complex (Helicopter), Machinery (Motor), Matte (Handbag), and Rigid (Battery) captures — scoring 95-98/100 READY.</span></div>'
            f'</div>'
            f'<div class="gif-card">'
            f'<img src="{gif_defects_uri}" alt="Clean vs 6 Synthetic Capture Defects Comparison" />'
            f'<div class="gif-caption"><strong>Clean vs. 6 Synthetic Defects (Vase 47433-58842-22217)</strong>'
            f'<span>Live hybrid inference separates clean capture (98/100 PASS) from Motion Blur, Low Light, Harsh Glare, Exposure Flicker, Low Overlap (23/100), and Incomplete Orbit (stopped at 35%).</span></div>'
            f'</div>'
            f'</div></div>',
            unsafe_allow_html=True,
        )

    # Interactive A-to-Z End-to-End Pipeline (Stages A -> I)
    st.markdown(
        f'<div class="pf-card" style="margin-bottom:0.65rem;">'
        f'<div class="pf-card-title" style="margin-bottom:0.35rem;"><span class="pf-icon-wrap">{ICONS["branch"]}</span><span>A-to-Z End-to-End Pipeline (Stages A to I)</span></div>'
        f'<p class="section-desc" style="margin:0;">Click any stage button below to inspect the exact scripts, design decisions, and verified outputs built across the project.</p>'
        f'</div>',
        unsafe_allow_html=True,
    )

    if "active_stage_id" not in st.session_state:
        st.session_state.active_stage_id = "A"

    stage_cols = st.columns(len(PIPELINE_STAGES))
    for idx, stg in enumerate(PIPELINE_STAGES):
        with stage_cols[idx]:
            is_active = st.session_state.active_stage_id == stg["id"]
            if st.button(
                f"{stg['id']}: {stg['tag']}",
                key=f"stage_btn_{stg['id']}",
                type="primary" if is_active else "secondary",
                use_container_width=True,
            ):
                st.session_state.active_stage_id = stg["id"]
                st.rerun()

    active_stage = next(
        (s for s in PIPELINE_STAGES if s["id"] == st.session_state.active_stage_id),
        PIPELINE_STAGES[0],
    )
    hl_chips = "".join(f'<span class="hl-chip">{ICONS["check"]}<span>{html.escape(h)}</span></span>' for h in active_stage["highlights"])
    mini_cards = "".join(
        f'<div class="timeline-mini-card {"selected" if s["id"] == active_stage["id"] else ""}">'
        f'<div class="tm-top"><span class="tm-id">Stage {s["id"]}</span><span class="tag">{html.escape(s["tag"])}</span></div>'
        f'<h4>{html.escape(s["title"])}</h4>'
        f'<p>{html.escape(s["summary"])}</p>'
        f'</div>'
        for s in PIPELINE_STAGES
    )
    st.markdown(
        f'<div class="pf-card">'
        f'<div class="stage-detail-card">'
        f'<div class="stage-detail-head">'
        f'<div class="stage-badge-lg">Stage {active_stage["id"]}</div>'
        f'<div><h3>{html.escape(active_stage["title"])}</h3><code class="script-chip">{html.escape(active_stage["script"])}</code></div>'
        f'</div>'
        f'<p class="stage-detail-summary">{html.escape(active_stage["summary"])}</p>'
        f'<div class="stage-highlights">{hl_chips}</div>'
        f'</div>'
        f'<div class="timeline-all-grid">{mini_cards}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # 40-Category Taxonomy & Feature Means Table
    tax_cards = "".join(
        f'<div class="tax-card"><h4>{html.escape(g["group"])}</h4>'
        f'<p class="tax-items">{html.escape(g["items"])}</p>'
        f'<span class="tax-challenge">Edge case: {html.escape(g["challenge"])}</span></div>'
        for g in TAXONOMY_GROUPS
    )
    mean_rows = "".join(
        f'<tr><td class="mono-strong">{html.escape(r["defect"])}</td>'
        f'<td>{r["duration"]}</td><td>{r["laplacian"]}</td><td>{r["blurry"]}</td>'
        f'<td>{r["brightness"]}</td><td>{r["stepDiff"]}</td><td>{r["dark"]}</td>'
        f'<td>{r["glare"]}</td><td>{r["orb"]}</td><td>{r["loop"]}</td></tr>'
        for r in FEATURE_CLASS_MEANS
    )
    st.markdown(
        f'<div class="pf-card">'
        f'<div class="pf-card-title"><span class="pf-icon-wrap">{ICONS["db"]}</span><span>uCO3D 40-Category Curation &amp; Extracted Feature Signatures</span></div>'
        f'<p class="section-desc">40 clean object sequences (1.000 visibility ratio, 18,361 mean 3D points) expanded into 280 labeled clips across 7 classes.</p>'
        f'<div class="taxonomy-grid">{tax_cards}</div>'
        f'<h4 style="font-size:0.88rem; margin:0 0 0.65rem 0;">Mean Extracted CV Features Across All 7 Defect Categories (280 Videos)</h4>'
        f'<div class="table-scroll"><table class="data-table"><thead><tr>'
        f'<th>Defect Class</th><th>Duration (s)</th><th>Laplacian Var</th><th>Blurry Ratio</th>'
        f'<th>Brightness</th><th>Step Diff</th><th>Dark Ratio</th><th>Glare Ratio</th>'
        f'<th>ORB KPs</th><th>Loop Closure</th></tr></thead><tbody>{mean_rows}</tbody></table></div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Model Benchmarks & Final Production Models
    st.markdown(
        f'<div class="pf-card" style="margin-bottom:0.65rem;">'
        f'<div class="pf-card-title" style="margin-bottom:0.35rem;"><span class="pf-icon-wrap">{ICONS["award"]}</span><span>Model Benchmarks &amp; Final Production Models</span></div>'
        f'<p class="section-desc" style="margin:0;">Evaluated with 5-Fold StratifiedGroupKFold grouped by sequence_id (32 training objects / 8 unseen holdout objects = 56 test clips).</p>'
        f'</div>',
        unsafe_allow_html=True,
    )
    bench_task = st.radio(
        "Select Benchmark Task:",
        options=["Defect_Diagnosis_7Class", "Readiness_Binary"],
        format_func=lambda k: "7-Class Defect Diagnosis" if k == "Defect_Diagnosis_7Class" else "Binary Readiness Gate",
        horizontal=True,
        label_visibility="collapsed",
    )
    bench_rows_list = []
    for r in BENCHMARK_RESULTS[bench_task]:
        row_cls = "winner-row" if r.get("winner") else ""
        win_badge = '<span class="win-pill">SELECTED</span>' if r.get("winner") else ""
        bench_rows_list.append(
            f'<tr class="{row_cls}">'
            f'<td>{r["stage"]}</td>'
            f'<td class="mono-strong">{r["model"]}{win_badge}</td>'
            f'<td>{r["cvAcc"]}</td><td>{r["cvStd"]}</td><td>{r["cvF1"]}</td>'
            f'<td>{r["holdAcc"]}</td><td>{r["holdF1"]}</td></tr>'
        )
    bench_rows_html = "".join(bench_rows_list)

    st.markdown(
        f'<div class="pf-card">'
        f'<div class="winner-cards-grid">'
        f'<div class="winner-card"><span class="winner-tag">Final 7-Class Diagnosis Model</span>'
        f'<h3>SoftVoting_Ensemble</h3>'
        f'<p class="winner-sub">Tuned RandomForest (350 trees) + XGBoost (lr=0.03, depth=4) + LightGBM (leaves=23, depth=3)</p>'
        f'<div class="winner-metrics"><div><span>5-Fold CV Acc</span><strong>77.14%</strong></div>'
        f'<div><span>CV Macro F1</span><strong>0.7655</strong></div>'
        f'<div><span>Holdout Acc</span><strong>76.79%</strong></div>'
        f'<div><span>Holdout F1</span><strong>0.7721</strong></div></div></div>'
        f'<div class="winner-card"><span class="winner-tag green">Final Binary Readiness Gate</span>'
        f'<h3>XGBoost_Tuned</h3>'
        f'<p class="winner-sub">220 trees, max_depth=5, lr=0.08, subsample=0.75, reg_alpha=0.5, scale_pos_weight=6.0</p>'
        f'<div class="winner-metrics"><div><span>5-Fold CV Acc</span><strong>86.79%</strong></div>'
        f'<div><span>CV Std</span><strong>±0.0143</strong></div>'
        f'<div><span>CV Macro F1</span><strong>0.7302</strong></div>'
        f'<div><span>Holdout Acc</span><strong>85.71%</strong></div></div></div>'
        f'</div>'
        f'<div class="table-scroll"><table class="data-table"><thead><tr>'
        f'<th>Stage</th><th>Model</th><th>5-Fold CV Acc</th><th>CV Std</th>'
        f'<th>CV Macro F1</th><th>Holdout Acc</th><th>Holdout Macro F1</th>'
        f'</tr></thead><tbody>{bench_rows_html}</tbody></table></div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Holdout Per-Class F1 & Top 10 Consensus Feature Importances
    col_f1, col_imp = st.columns(2, gap="large")
    with col_f1:
        f1_bars = "".join(
            f'<div class="prob-row">'
            f'<span class="prob-name">{html.escape(item["cls"])}</span>'
            f'<div class="prob-track"><div class="prob-fill {"clean" if item["f1"] >= 0.8 else ("alert" if item["f1"] < 0.5 else "")}" style="width:{item["f1"] * 100:.1f}%;"></div></div>'
            f'<span class="prob-val">{item["f1"]:.3f}</span>'
            f'</div>'
            for item in HOLDOUT_CLASS_REPORT
        )
        st.markdown(
            f'<div class="pf-card">'
            f'<div class="pf-card-title"><span class="pf-icon-wrap">{ICONS["chart"]}</span><span>Unseen Holdout Per-Class F1 (56 Test Videos)</span></div>'
            f'<p class="section-desc">Strong precision/recall on glare, orbit cutoff, low light, blur, and flicker. Deterministic Geometric Guardrails backstop low_overlap_fast_move where 2.5 FPS downsampling compresses frame-drop gaps.</p>'
            f'{f1_bars}'
            f'</div>',
            unsafe_allow_html=True,
        )
    with col_imp:
        imp_bars = "".join(
            f'<div class="prob-row">'
            f'<span class="prob-name mono-feat" title="{html.escape(f["role"])}">{html.escape(f["name"])}</span>'
            f'<div class="prob-track"><div class="prob-fill" style="width:{(f["importance"] / 0.16) * 100:.1f}%;"></div></div>'
            f'<span class="prob-val">{f["importance"] * 100:.1f}%</span>'
            f'</div>'
            for f in TOP_FEATURES
        )
        st.markdown(
            f'<div class="pf-card">'
            f'<div class="pf-card-title"><span class="pf-icon-wrap">{ICONS["layers"]}</span><span>Top 10 Consensus Feature Importances</span></div>'
            f'<p class="section-desc">Normalized feature importances averaged across Tuned Random Forest, XGBoost, and LightGBM inside the Soft-Voting Ensemble.</p>'
            f'{imp_bars}'
            f'</div>',
            unsafe_allow_html=True,
        )

    # Completed Final Output — Live Dashboard Results Gallery
    st.markdown(
        f'<div class="pf-card" style="margin-bottom:0.75rem;">'
        f'<div class="pf-card-title" style="margin-bottom:0.35rem;"><span class="pf-icon-wrap">{ICONS["monitor"]}</span><span>Completed Final Output — Live Dashboard Results</span></div>'
        f'<p class="section-desc" style="margin:0;">End-to-end production runs captured on real uploaded test &amp; smartphone videos, demonstrating both the Soft-Voting Ensemble and deterministic Geometric Guardrail in action.</p>'
        f'</div>',
        unsafe_allow_html=True,
    )
    gal_cols = st.columns(3, gap="medium")
    for idx, item in enumerate(FINAL_OUTPUT_GALLERY):
        with gal_cols[idx]:
            img_uri = encode_image_data_uri(item["path"])
            img_tag = f'<img src="{img_uri}" alt="{html.escape(item["file"])}" />' if img_uri else ""
            st.markdown(
                f'<div class="output-card">'
                f'{img_tag}'
                f'<div class="output-body">'
                f'<div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:0.4rem;">'
                f'<span class="verdict-badge {item["tier"]}" style="margin-bottom:0; font-size:0.7rem;">Score {item["score"]}/100 &bull; {html.escape(item["verdict"])}</span>'
                f'<span class="tag">{html.escape(item["trigger"])}</span>'
                f'</div>'
                f'<h4>{html.escape(item["file"])}</h4>'
                f'<p>{html.escape(item["summary"])}</p>'
                f'</div></div>',
                unsafe_allow_html=True,
            )
            if os.path.exists(item["path"]):
                with st.expander("Inspect Full Screenshot"):
                    st.image(item["path"], use_container_width=True)


# ---------------------------------------------------------------------------
# App Entry Point
# ---------------------------------------------------------------------------


def main():
    if "theme" not in st.session_state:
        st.session_state.theme = "dark"
    if "page" not in st.session_state:
        st.session_state.page = "analyzer"
    if "analysis_result" not in st.session_state:
        st.session_state.analysis_result = None

    st.markdown(build_theme_css(st.session_state.theme), unsafe_allow_html=True)
    analyzer = load_analyzer()

    defect_model = html.escape(analyzer.bundle.get("defect_model_name", "SoftVoting_Ensemble"))
    binary_model = html.escape(analyzer.bundle.get("binary_model_name", "XGBoost_Tuned"))
    num_features = len(analyzer.feature_cols)

    # Topbar matching App.jsx
    st.markdown(
        f'<div class="pf-topbar">'
        f'<div class="pf-brand">'
        f'<div class="pf-brand-icon">{ICONS["box"]}</div>'
        f'<div><h1>PreFlight-3D</h1><span>Photogrammetry &amp; 3DGS Video Quality Gate</span></div>'
        f'</div>'
        f'<div class="pf-health-pill">'
        f'<span class="pf-dot"></span>'
        f'<span>Engine Online &bull; {defect_model} + {binary_model} ({num_features} CV features)</span>'
        f'</div></div>',
        unsafe_allow_html=True,
    )

    # Navigation Bar + Theme Switcher
    _, nav_mid, nav_theme = st.columns([0.9, 2.3, 1.0])
    with nav_mid:
        n1, n2 = st.columns(2)
        with n1:
            if st.button(
                "Pre-Flight Analyzer",
                type="primary" if st.session_state.page == "analyzer" else "secondary",
                use_container_width=True,
            ):
                st.session_state.page = "analyzer"
                st.rerun()
        with n2:
            if st.button(
                "How It Works (A-Z)",
                type="primary" if st.session_state.page == "how-it-works" else "secondary",
                use_container_width=True,
            ):
                st.session_state.page = "how-it-works"
                st.rerun()

    with nav_theme:
        toggle_label = "Light Theme" if st.session_state.theme == "dark" else "Dark Theme"
        if st.button(toggle_label, key="theme_toggle_btn", type="secondary", use_container_width=True):
            st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
            st.rerun()

    st.markdown("<div style='height:0.65rem;'></div>", unsafe_allow_html=True)

    if st.session_state.page == "how-it-works":
        render_how_it_works_page()
    else:
        render_analyzer_page(analyzer)


if __name__ == "__main__":
    main()
