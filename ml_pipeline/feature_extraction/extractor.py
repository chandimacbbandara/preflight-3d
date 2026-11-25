import os
import cv2
import numpy as np


def _resize_standard(frame, max_dim=480):
    h, w = frame.shape[:2]
    scale = max_dim / max(h, w)
    if scale >= 1.0:
        return frame
    new_w = max(2, int(round(w * scale)) // 2 * 2)
    new_h = max(2, int(round(h * scale)) // 2 * 2)
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _create_center_subject_mask(h, w):
    """Elliptical center ROI covering the primary subject area (~65% width/height)."""
    mask = np.zeros((h, w), dtype=np.uint8)
    center = (w // 2, h // 2)
    axes = (int(w * 0.34), int(h * 0.36))
    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
    return mask


def _compute_pair_geometry(orb, bf, kp1, des1, kp2, des2):
    """Return (good_match_ratio, ransac_inlier_ratio) between two frames."""
    if des1 is None or des2 is None or len(kp1) < 12 or len(kp2) < 12:
        return 0.0, 0.0

    raw_matches = bf.knnMatch(des1, des2, k=2)
    good = []
    for pair in raw_matches:
        if len(pair) == 2:
            m, n = pair
            if m.distance < 0.75 * n.distance:
                good.append(m)

    match_ratio = len(good) / max(min(len(kp1), len(kp2)), 1)
    if len(good) < 8:
        return float(match_ratio), 0.0

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    _, inliers = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if inliers is None:
        return float(match_ratio), 0.0

    inlier_ratio = float(np.sum(inliers)) / len(good)
    return float(match_ratio), inlier_ratio


def extract_video_features(video_path: str, target_fps: float = 2.5) -> dict:
    """
    Extract 18 computer vision, lighting, texture, and camera trajectory features
    from an object capture video. Runs on CPU in ~0.3-0.5s per video.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps if fps > 0 else 0.0

    # Ensure at least ~12 sampled frames even on short clips
    step = max(1, int(round(fps / target_fps)))
    if total_frames > 0 and (total_frames // step) < 12:
        step = max(1, total_frames // 12)

    orb = cv2.ORB_create(nfeatures=600)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    laplacian_vars = []
    edge_densities = []
    brightness_means = []
    dark_ratios = []
    glare_ratios = []
    orb_counts = []

    flow_magnitudes = []
    pan_shifts_x = []
    pair_match_ratios = []
    pair_inlier_ratios = []

    prev_gray = None
    prev_kp, prev_des = None, None
    first_kp, first_des = None, None
    last_kp, last_des = None, None
    roi_mask = None

    frame_idx = 0
    sampled_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % step == 0:
            small = _resize_standard(frame, max_dim=480)
            h, w = small.shape[:2]
            if roi_mask is None:
                roi_mask = _create_center_subject_mask(h, w)

            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)

            roi_bool = roi_mask > 0
            roi_pixels = max(int(np.count_nonzero(roi_bool)), 1)

            # 1. Sharpness & edge density inside subject ROI
            lap = cv2.Laplacian(gray, cv2.CV_64F)
            laplacian_vars.append(float(lap[roi_bool].var()))

            edges = cv2.Canny(gray, 80, 160)
            edge_densities.append(float(np.count_nonzero(edges[roi_bool])) / roi_pixels)

            # 2. Brightness, underexposure & specular glare inside subject ROI
            roi_gray = gray[roi_bool]
            brightness_means.append(float(np.mean(roi_gray)))
            dark_ratios.append(float(np.mean(roi_gray < 40)))

            s_chan = hsv[:, :, 1][roi_bool]
            v_chan = hsv[:, :, 2][roi_bool]
            glare_pixels = np.logical_and(v_chan > 240, s_chan < 45)
            glare_ratios.append(float(np.mean(glare_pixels)))

            # 3. Texture richness (ORB keypoints in subject ROI)
            kp, des = orb.detectAndCompute(gray, roi_mask)
            orb_counts.append(len(kp))

            if first_kp is None:
                first_kp, first_des = kp, des
            last_kp, last_des = kp, des

            # 4. Temporal camera motion, overlap & geometric consistency
            if prev_gray is not None:
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
                )
                mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                flow_magnitudes.append(float(np.mean(mag)))
                pan_shifts_x.append(float(np.median(flow[..., 0])))

                m_ratio, inlier_ratio = _compute_pair_geometry(
                    orb, bf, prev_kp, prev_des, kp, des
                )
                pair_match_ratios.append(m_ratio)
                pair_inlier_ratios.append(inlier_ratio)

            prev_gray = gray
            prev_kp, prev_des = kp, des
            sampled_count += 1

        frame_idx += 1

    cap.release()

    if sampled_count == 0:
        raise ValueError(f"Could not read any frames from {video_path}")

    # 5. Loop-closure check between start and end of capture (360° orbit completeness)
    loop_match_ratio, loop_inlier_ratio = _compute_pair_geometry(
        orb, bf, first_kp, first_des, last_kp, last_des
    )

    lap_arr = np.array(laplacian_vars)
    bright_arr = np.array(brightness_means)
    bright_diffs = np.abs(np.diff(bright_arr)) if len(bright_arr) > 1 else np.array([0.0])

    return {
        "duration_sec": round(float(duration_sec), 3),
        "sampled_frames": int(sampled_count),
        # Sharpness / Blur features
        "laplacian_var_mean": round(float(np.mean(lap_arr)), 3),
        "laplacian_var_p10": round(float(np.percentile(lap_arr, 10)), 3),
        "blurry_frame_ratio": round(float(np.mean(lap_arr < 35.0)), 4),
        "edge_density_mean": round(float(np.mean(edge_densities)), 4),
        # Lighting / Exposure / Glare features
        "brightness_mean": round(float(np.mean(bright_arr)), 3),
        "brightness_std": round(float(np.std(bright_arr)), 3),
        "brightness_step_diff_mean": round(float(np.mean(bright_diffs)), 3),
        "dark_pixel_ratio": round(float(np.mean(dark_ratios)), 4),
        "specular_glare_ratio": round(float(np.mean(glare_ratios)), 4),
        # Texture features
        "orb_keypoints_mean": round(float(np.mean(orb_counts)), 2),
        "orb_keypoints_min": int(np.min(orb_counts)),
        # Camera motion, overlap & orbit coverage features
        "optical_flow_mean": round(float(np.mean(flow_magnitudes)) if flow_magnitudes else 0.0, 3),
        "optical_flow_max": round(float(np.max(flow_magnitudes)) if flow_magnitudes else 0.0, 3),
        "pair_match_ratio_mean": round(float(np.mean(pair_match_ratios)) if pair_match_ratios else 0.0, 4),
        "pair_inlier_ratio_mean": round(float(np.mean(pair_inlier_ratios)) if pair_inlier_ratios else 0.0, 4),
        "cumulative_pan_px": round(float(np.abs(np.sum(pan_shifts_x))), 2),
        "loop_closure_inlier_ratio": round(float(loop_inlier_ratio), 4),
    }