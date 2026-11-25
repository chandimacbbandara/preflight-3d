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


def _passes_physics_verification(defect_code: str, features: dict) -> bool:
    """
    Verify an ML-predicted defect against physical CV measurements to suppress
    false positives on real-world captures (e.g., turntable loops or indoor lighting gradients).
    """
    if defect_code == "incomplete_orbit":
        # Strong start-to-end loop closure with active motion proves a closed 360° orbit
        # (handles both slower handheld orbits and faster turntable captures)
        if features["loop_closure_inlier_ratio"] >= 0.50 and features["optical_flow_mean"] >= 2.0:
            return False
        if features["loop_closure_inlier_ratio"] >= 0.45 and features["duration_sec"] >= 10.0:
            return False
        return True

    if defect_code == "exposure_flicker":
        # Normal directional indoor lighting produces step diffs ~7-14; true flicker is >= 18
        return (
            features["brightness_step_diff_mean"] >= 18.0
            or features["brightness_std"] >= 32.0
        )

    if defect_code == "motion_blur":
        return (
            features["laplacian_var_mean"] < 120.0
            or features["blurry_frame_ratio"] > 0.20
        )

    if defect_code == "low_light":
        return (
            features["brightness_mean"] < 85.0
            or features["dark_pixel_ratio"] > 0.25
        )

    if defect_code == "harsh_glare":
        return (
            features["specular_glare_ratio"] > 0.12
            or features["brightness_mean"] > 195.0
        )

    if defect_code == "low_overlap_fast_move":
        return (
            features["pair_inlier_ratio_mean"] < 0.45
            or features["optical_flow_max"] > 18.0
        )

    return True


def _compute_cv_health_score(features: dict) -> float:
    """Compute a normalized [0, 1] reconstruction health score from deterministic CV telemetry."""
    # 1. Focus & sharpness (25%)
    sharpness_score = min(1.0, features["laplacian_var_mean"] / 150.0) * (
        1.0 - min(1.0, features["blurry_frame_ratio"])
    )

    # 2. Feature tracking & multi-view overlap (30%)
    tracking_score = (
        0.65 * min(1.0, features["pair_inlier_ratio_mean"] / 0.65)
        + 0.35 * min(1.0, features["orb_keypoints_mean"] / 250.0)
    )

    # 3. Lighting balance, glare & exposure stability (25%)
    bright = features["brightness_mean"]
    if 75.0 <= bright <= 190.0:
        bright_comfort = 1.0
    elif bright < 75.0:
        bright_comfort = max(0.0, bright / 75.0)
    else:
        bright_comfort = max(0.0, (255.0 - bright) / 65.0)

    glare_comfort = max(0.0, 1.0 - features["specular_glare_ratio"] * 3.0)
    flicker_comfort = max(
        0.0, 1.0 - max(0.0, (features["brightness_step_diff_mean"] - 8.0) / 28.0)
    )
    lighting_score = bright_comfort * glare_comfort * flicker_comfort

    # 4. Orbit duration & loop closure (20%)
    duration_factor = min(1.0, features["duration_sec"] / 12.0)
    coverage_factor = max(
        min(1.0, features["loop_closure_inlier_ratio"] / 0.45),
        min(1.0, features["cumulative_pan_px"] / 30.0),
    )
    orbit_score = 0.5 * duration_factor + 0.5 * coverage_factor
    if features["loop_closure_inlier_ratio"] >= 0.50 and features["optical_flow_mean"] >= 2.0:
        orbit_score = max(orbit_score, coverage_factor * 0.95)

    return float(
        0.25 * sharpness_score
        + 0.30 * tracking_score
        + 0.25 * lighting_score
        + 0.20 * orbit_score
    )


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
        raw_prob_map = {cls: float(p) for cls, p in zip(class_names, defect_probs)}

        binary_probs = self.binary_model.predict_proba(X_input)[0]
        good_prob = float(binary_probs[1])
        clean_class_prob = float(raw_prob_map.get("none", 0.0))

        # 2. Hybrid Diagnosis: ML probabilities gated by physical CV verification
        detected_issues = []
        suppressed_prob_mass = 0.0
        calibrated_prob_map = dict(raw_prob_map)

        sorted_defects = sorted(
            [(cls, p) for cls, p in raw_prob_map.items() if cls != "none"],
            key=lambda x: x[1],
            reverse=True,
        )
        for cls, prob in sorted_defects:
            if prob >= 0.25:
                if _passes_physics_verification(cls, features):
                    detected_issues.append((cls, prob, "ML Model"))
                else:
                    suppressed_prob_mass += prob
                    calibrated_prob_map[cls] = 0.0

        if suppressed_prob_mass > 0.0 and "none" in calibrated_prob_map:
            calibrated_prob_map["none"] = min(1.0, calibrated_prob_map["none"] + suppressed_prob_mass)

        prob_map = {cls: round(float(p), 4) for cls, p in calibrated_prob_map.items()}

        # Deterministic physics guardrails for edge cases missed by ML
        existing_codes = {code for code, _, _ in detected_issues}
        if "low_overlap_fast_move" not in existing_codes and (
            features["pair_inlier_ratio_mean"] < 0.25
            or (
                features["optical_flow_max"] > 22.0
                and features["pair_inlier_ratio_mean"] < 0.45
            )
        ):
            detected_issues.append(("low_overlap_fast_move", 0.72, "Geometric Guardrail"))

        if (
            "incomplete_orbit" not in existing_codes
            and features["loop_closure_inlier_ratio"] < 0.35
            and features["cumulative_pan_px"] < 20.0
            and features["duration_sec"] < 9.0
        ):
            detected_issues.append(("incomplete_orbit", 0.70, "Trajectory Guardrail"))

        # 3. Calibrated 0-100 Readiness Score
        cv_health = _compute_cv_health_score(features)
        effective_clean_prob = min(1.0, clean_class_prob + suppressed_prob_mass)
        ml_clean_conf = max(good_prob, effective_clean_prob)

        if not detected_issues:
            raw_score = (0.60 * cv_health + 0.40 * ml_clean_conf) * 100.0
            readiness_score = int(np.clip(round(raw_score), 65, 98))
        else:
            max_defect_conf = max(conf for _, conf, _ in detected_issues)
            has_high_severity = any(
                RECOMMENDATION_CATALOG[code]["severity"] == "HIGH"
                for code, _, _ in detected_issues
            )
            penalty = max_defect_conf * (65.0 if has_high_severity else 40.0)
            raw_score = (0.50 * cv_health + 0.50 * good_prob) * 100.0 - penalty
            upper_bound = 44 if has_high_severity else 68
            readiness_score = int(np.clip(round(raw_score), 1, upper_bound))

        # 4. Determine Final Verdict
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

    test_samples = [
        df_manifest[df_manifest["defect_type"] == "none"].iloc[0]["video_path"],
        df_manifest[df_manifest["defect_type"] == "harsh_glare"].iloc[0]["video_path"],
        df_manifest[df_manifest["defect_type"] == "incomplete_orbit"].iloc[0]["video_path"],
    ]
    for path in test_samples:
        result = analyzer.analyze_video(path)
        print("\n" + "=" * 60)
        print(json.dumps({k: v for k, v in result.items() if k != "extracted_metrics"}, indent=2))
