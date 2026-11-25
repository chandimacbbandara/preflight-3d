import os
import sys
import joblib
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath("."))
from ml_pipeline.feature_extraction.extractor import extract_video_features

BUNDLE_PATH = "models/preflight_model_bundle.joblib"

RECOMMENDATION_CATALOG = {
    "motion_blur": {
        "title": "Excessive Motion Blur Detected",
        "category": "Camera Movement & Shutter",
        "severity": "HIGH",
        "impact": "Blurry frames break SIFT/ORB keypoint matching and produce melted or hollow 3D surfaces.",
        "action_steps": [
            "Walk 2x slower around the object and pause briefly if repositioning.",
            "Increase ambient room lighting so your camera uses a faster shutter speed.",
            "Record at 60 FPS if supported by your device to minimize motion smear.",
        ],
    },
    "low_light": {
        "title": "Underexposed / Low-Light Capture",
        "category": "Lighting & Exposure",
        "severity": "HIGH",
        "impact": "Dark shadows introduce sensor grain and hide surface texture needed for depth triangulation.",
        "action_steps": [
            "Add diffuse front/side lighting (e.g., desk lamps or soft window light) to illuminate shadows.",
            "Avoid filming against bright windows or backlighting that darkens the object.",
            "Tap the object on your screen and increase exposure slider slightly before recording.",
        ],
    },
    "harsh_glare": {
        "title": "Specular Glare & Highlight Clipping",
        "category": "Lighting & Surface Reflection",
        "severity": "HIGH",
        "impact": "View-dependent bright reflections shift across frames and confuse multi-view geometry.",
        "action_steps": [
            "Turn off camera flash and move away from direct spotlights or harsh sunlight.",
            "Use diffused lighting (bounce light off a wall or white paper).",
            "For glossy/metallic objects, lightly dust with matte scanning spray or talcum powder.",
        ],
    },
    "exposure_flicker": {
        "title": "Unstable Auto-Exposure / Brightness Pumping",
        "category": "Camera Settings & Lighting",
        "severity": "MEDIUM",
        "impact": "Sudden brightness shifts between angles cause color banding and inconsistent feature descriptors.",
        "action_steps": [
            "Enable AE/AF Lock (press and hold on the object) before starting the recording.",
            "Keep lighting uniform on all sides of the object so the sensor does not adjust mid-orbit.",
        ],
    },
    "low_overlap_fast_move": {
        "title": "Insufficient Frame Overlap / Fast Panning",
        "category": "Camera Trajectory & Pacing",
        "severity": "MEDIUM",
        "impact": "Large jumps between consecutive viewpoints cause Structure-from-Motion tracking loss.",
        "action_steps": [
            "Maintain at least 70% visual overlap between consecutive frames.",
            "Complete a full 360° orbit smoothly over 15–25 seconds rather than rushing.",
        ],
    },
    "incomplete_orbit": {
        "title": "Incomplete 360° Viewpoint Coverage",
        "category": "Camera Angle & Orbit Coverage",
        "severity": "HIGH",
        "impact": "Missing back or top views leaves unmodeled holes and distorted back-facing geometry.",
        "action_steps": [
            "Walk a complete closed 360° circle around the object until you return to your starting angle.",
            "Record at least two elevation loops: one at eye level and one from a higher ~45° top-down angle.",
        ],
    },
}


class PreFlightAnalyzer:
    def __init__(self, bundle_path: str = BUNDLE_PATH):
        if not os.path.exists(bundle_path):
            raise FileNotFoundError(f"Model bundle not found at {bundle_path}. Run train_models.py first.")
        self.bundle = joblib.load(bundle_path)
        self.feature_cols = self.bundle["feature_columns"]
        self.defect_model = self.bundle["defect_model"]
        self.defect_le = self.bundle["defect_label_encoder"]
        self.binary_model = self.bundle["binary_model"]

    def analyze_video(self, video_path: str) -> dict:
        features = extract_video_features(video_path)
        X_input = pd.DataFrame([features])[self.feature_cols]

        # 1. ML Predictions (Soft-Voting Ensemble + Tuned XGBoost Readiness)
        defect_probs = self.defect_model.predict_proba(X_input)[0]
        class_names = list(self.defect_le.classes_)
        prob_map = {cls: round(float(p), 4) for cls, p in zip(class_names, defect_probs)}

        binary_probs = self.binary_model.predict_proba(X_input)[0]
        good_prob = float(binary_probs[1])
        clean_class_prob = float(prob_map.get("none", 0.0))

        # Calibrated 0-100 Readiness Score blending binary confidence & clean-class probability
        readiness_score = int(round((0.55 * good_prob + 0.45 * clean_class_prob) * 100))

        # 2. Hybrid Diagnosis: combine ML probabilities + deterministic physics guardrails
        detected_issues = []

        # Primary ML-predicted defect (if not 'none' and probability >= 0.25)
        sorted_defects = sorted(
            [(cls, p) for cls, p in prob_map.items() if cls != "none"],
            key=lambda x: x[1],
            reverse=True,
        )
        for cls, prob in sorted_defects:
            if prob >= 0.25:
                detected_issues.append((cls, prob, "ML Model"))

        # Physics guardrails for edge cases (e.g., low overlap / fast movement or low texture)
        existing_codes = {code for code, _, _ in detected_issues}
        if "low_overlap_fast_move" not in existing_codes and (
            features["pair_inlier_ratio_mean"] < 0.22 or features["optical_flow_max"] > 22.0
        ):
            detected_issues.append(("low_overlap_fast_move", 0.72, "Geometric Guardrail"))

        if "incomplete_orbit" not in existing_codes and (
            features["cumulative_pan_px"] < 20.0 and features["duration_sec"] < 8.0
        ):
            detected_issues.append(("incomplete_orbit", 0.70, "Trajectory Guardrail"))

        # Determine final verdict
        if readiness_score >= 70 and not detected_issues:
            verdict = "READY_FOR_RECONSTRUCTION"
            summary = "Capture quality looks solid. Sharpness, lighting consistency, and 360° coverage are suitable for COLMAP / 3DGS."
        elif readiness_score >= 45 and len(detected_issues) <= 1:
            verdict = "USABLE_WITH_WARNINGS"
            summary = "Reconstruction may succeed, but addressing the flagged capture warning below will noticeably improve mesh completeness."
        else:
            verdict = "RECAPTURE_RECOMMENDED"
            summary = "High risk of failed camera alignment or holes in the 3D model. Fix the issues below before running heavy reconstruction."

        recommendations = []
        for code, conf, source in detected_issues:
            template = RECOMMENDATION_CATALOG[code]
            recommendations.append({
                "issue_code": code,
                "confidence": round(conf, 3),
                "detected_by": source,
                **template,
            })

        # Extra surface-texture advisory if object has very few keypoints even when sharp
        if features["orb_keypoints_mean"] < 35 and features["laplacian_var_mean"] > 80:
            recommendations.append({
                "issue_code": "low_surface_texture",
                "confidence": 0.80,
                "detected_by": "Feature Density Check",
                "title": "Low Surface Texture / Featureless Object",
                "category": "Object Surface & Background",
                "severity": "MEDIUM",
                "impact": "Smooth, solid-colored surfaces give photogrammetry very few anchor points to track.",
                "action_steps": [
                    "Place the object on a textured surface (e.g., patterned mat or newspaper) to anchor camera pose tracking.",
                    "Avoid plain white tables or blank walls behind the object.",
                ],
            })

        return {
            "video_file": os.path.basename(video_path),
            "verdict": verdict,
            "readiness_score": readiness_score,
            "summary": summary,
            "defect_probabilities": prob_map,
            "recommendations": recommendations,
            "extracted_metrics": features,
        }


if __name__ == "__main__":
    import json
    analyzer = PreFlightAnalyzer()
    df_manifest = pd.read_csv("data/processed/quality_measurements/dataset_manifest.csv")

    # Quick smoke test on 1 clean video and 2 degraded videos
    test_samples = [
        df_manifest[df_manifest["defect_type"] == "none"].iloc[0]["video_path"],
        df_manifest[df_manifest["defect_type"] == "harsh_glare"].iloc[0]["video_path"],
        df_manifest[df_manifest["defect_type"] == "incomplete_orbit"].iloc[0]["video_path"],
    ]
    for path in test_samples:
        result = analyzer.analyze_video(path)
        print("\n" + "=" * 60)
        print(json.dumps({k: v for k, v in result.items() if k != "extracted_metrics"}, indent=2))