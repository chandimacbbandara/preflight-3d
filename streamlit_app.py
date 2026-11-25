"""
PreFlight-3D | Streamlit Cloud Entry Point
==========================================
Imports PreFlightAnalyzer directly — no FastAPI server required.
Runs on Streamlit Community Cloud against models/preflight_model_bundle.joblib.
"""

import os
import glob
import tempfile

import pandas as pd
import streamlit as st

# ── Page config (must be first Streamlit call) ─────────────────────────────
st.set_page_config(
    page_title="PreFlight-3D | 3D Reconstruction Quality Gate",
    page_icon="📦",
    layout="wide",
)

# ── Custom CSS — dark telemetry aesthetic ──────────────────────────────────
st.markdown(
    """
    <style>
    /* ── Global ── */
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #0b0f19;
        color: #e2e8f0;
        font-family: 'Inter', 'Segoe UI', sans-serif;
    }
    [data-testid="stSidebar"] { background-color: #111827; }

    /* ── Header strip ── */
    .pf-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #111827 60%, #0b0f19 100%);
        border: 1px solid #1e40af44;
        border-radius: 12px;
        padding: 24px 32px;
        margin-bottom: 20px;
    }
    .pf-header h1 { margin: 0; font-size: 2rem; color: #93c5fd; letter-spacing: -0.5px; }
    .pf-header p  { margin: 4px 0 0; color: #64748b; font-size: 0.9rem; }

    /* ── Metric pill badges ── */
    .pf-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 6px;
        vertical-align: middle;
    }
    .badge-green  { background:#052e16; color:#4ade80; border:1px solid #166534; }
    .badge-amber  { background:#1c1203; color:#fbbf24; border:1px solid #92400e; }
    .badge-red    { background:#1f0505; color:#f87171; border:1px solid #991b1b; }
    .badge-blue   { background:#0d1f3c; color:#60a5fa; border:1px solid #1e40af; }
    .badge-purple { background:#1a0533; color:#c084fc; border:1px solid #7e22ce; }

    /* ── Recommendation card ── */
    .rec-card {
        background: #111827;
        border-left: 4px solid #3b82f6;
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 10px;
    }
    .rec-card.sev-high   { border-left-color: #ef4444; }
    .rec-card.sev-medium { border-left-color: #f59e0b; }
    .rec-card.sev-low    { border-left-color: #22c55e; }
    .rec-card h4 { margin: 0 0 6px; color: #e2e8f0; font-size: 0.95rem; }
    .rec-card p  { margin: 3px 0; color: #94a3b8; font-size: 0.82rem; }
    .rec-card ol { margin: 8px 0 0 16px; color: #cbd5e1; font-size: 0.82rem; }

    /* ── Verdict banner ── */
    .verdict-ready   { background:#052e16; border:1px solid #166534; border-radius:10px; padding:16px 20px; }
    .verdict-warning { background:#1c1203; border:1px solid #92400e; border-radius:10px; padding:16px 20px; }
    .verdict-recapture { background:#1f0505; border:1px solid #991b1b; border-radius:10px; padding:16px 20px; }
    .verdict-banner h2 { margin:0; font-size:1.3rem; }
    .verdict-banner p  { margin:4px 0 0; font-size:0.85rem; color:#94a3b8; }

    /* ── Pillar card ── */
    .pillar-row { display:flex; gap:12px; flex-wrap:wrap; margin-top:12px; }
    .pillar-card {
        flex:1; min-width:120px;
        background:#111827; border:1px solid #1e293b;
        border-radius:8px; padding:12px 14px;
        text-align:center;
    }
    .pillar-card .icon { font-size:1.5rem; }
    .pillar-card .label { font-size:0.72rem; color:#64748b; margin-top:4px; }

    /* ── Probability bar ── */
    .prob-row { display:flex; align-items:center; gap:8px; margin-bottom:6px; }
    .prob-label { width:180px; font-size:0.78rem; color:#94a3b8; text-align:right; flex-shrink:0; }
    .prob-bar-bg { flex:1; background:#1e293b; border-radius:4px; height:12px; overflow:hidden; }
    .prob-bar-fill { height:100%; border-radius:4px; }
    .prob-pct { width:40px; font-size:0.78rem; color:#cbd5e1; text-align:right; flex-shrink:0; }

    /* ── Section divider ── */
    .pf-section-title {
        font-size:0.7rem; font-weight:700; letter-spacing:0.1em;
        color:#3b82f6; text-transform:uppercase;
        border-bottom:1px solid #1e293b;
        padding-bottom:4px; margin-bottom:12px; margin-top:20px;
    }

    /* ── Hide default Streamlit chrome ── */
    #MainMenu, footer { visibility: hidden; }
    .stDeployButton { display:none; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading PreFlight-3D analyzer…")
def load_analyzer():
    """Load the PreFlightAnalyzer once into memory and cache it."""
    from backend.app.services.recommendation_engine import PreFlightAnalyzer
    return PreFlightAnalyzer(bundle_path="models/preflight_model_bundle.joblib")


def _sample_video_paths() -> dict[str, str]:
    """Return {display_name: abs_path} for available sample videos."""
    paths: dict[str, str] = {}

    # 1. Tracked sample_videos/ folder (deployed on Streamlit Cloud)
    for p in sorted(glob.glob("sample_videos/*.mp4")):
        name = os.path.splitext(os.path.basename(p))[0].replace("_", " ").replace("  ", " — ")
        paths[name] = os.path.abspath(p)

    # 2. Fallback: local data/processed/quality_measurements/dataset_manifest.csv
    if not paths:
        manifest = "data/processed/quality_measurements/dataset_manifest.csv"
        if os.path.exists(manifest):
            df = pd.read_csv(manifest)
            if "video_path" in df.columns:
                for _, row in df.iterrows():
                    vp = row["video_path"]
                    if os.path.exists(vp):
                        label = os.path.splitext(os.path.basename(vp))[0]
                        paths[label] = os.path.abspath(vp)

    return paths


def _verdict_html(verdict: str, score: int, summary: str) -> str:
    css_cls = {
        "READY_FOR_RECONSTRUCTION": "verdict-ready",
        "USABLE_WITH_WARNINGS":     "verdict-warning",
        "RECAPTURE_RECOMMENDED":    "verdict-recapture",
    }.get(verdict, "verdict-ready")

    icon = {"READY_FOR_RECONSTRUCTION": "✅", "USABLE_WITH_WARNINGS": "⚠️",
            "RECAPTURE_RECOMMENDED": "❌"}.get(verdict, "📊")

    friendly = verdict.replace("_", " ").title()
    return f"""
    <div class="{css_cls} verdict-banner">
      <h2>{icon} {friendly}</h2>
      <p>{summary}</p>
    </div>
    """


def _severity_badge(sev: str) -> str:
    css = {"high": "badge-red", "medium": "badge-amber", "low": "badge-green"}.get(
        sev.lower(), "badge-blue"
    )
    return f'<span class="pf-badge {css}">{sev.upper()}</span>'


def _prob_bar_html(label: str, prob: float, color: str = "#3b82f6") -> str:
    pct = prob * 100
    return f"""
    <div class="prob-row">
      <div class="prob-label">{label}</div>
      <div class="prob-bar-bg">
        <div class="prob-bar-fill" style="width:{pct:.1f}%;background:{color};"></div>
      </div>
      <div class="prob-pct">{pct:.1f}%</div>
    </div>
    """


# Metric display configuration
_METRIC_CFG: list[tuple[str, str, str | None]] = [
    ("duration_sec",             "Duration",             "s"),
    ("sampled_frames",           "Sampled Frames",       None),
    ("laplacian_var_mean",       "Sharpness",            "Laplacian"),
    ("blurry_frame_ratio",       "Blurry Frames",        "%"),
    ("edge_density_mean",        "Edge Density",         "%"),
    ("brightness_mean",          "Brightness /255",      None),
    ("brightness_step_diff_mean","Exposure Jump",        None),
    ("dark_pixel_ratio",         "Underexposed Pixels",  "%"),
    ("specular_glare_ratio",     "Specular Glare",       "%"),
    ("orb_keypoints_mean",       "Avg ORB Keypoints",    None),
    ("optical_flow_mean",        "Mean Flow",            "px"),
    ("optical_flow_max",         "Peak Flow",            "px"),
    ("pair_match_ratio_mean",    "Match Ratio",          "%"),
    ("pair_inlier_ratio_mean",   "RANSAC Inlier",        "%"),
    ("cumulative_pan_px",        "Orbit Pan",            "px"),
    ("loop_closure_inlier_ratio","Loop Closure",         "%"),
]
_PCT_KEYS = {
    "blurry_frame_ratio", "edge_density_mean", "dark_pixel_ratio",
    "specular_glare_ratio", "pair_match_ratio_mean", "pair_inlier_ratio_mean",
    "loop_closure_inlier_ratio",
}

_DEFECT_COLORS = {
    "none":                 "#22c55e",
    "motion_blur":          "#ef4444",
    "harsh_glare":          "#f59e0b",
    "incomplete_orbit":     "#8b5cf6",
    "low_overlap_fast_move":"#f97316",
    "low_light":            "#6366f1",
    "exposure_flicker":     "#ec4899",
}

PILLARS = [
    ("🔍", "Focus /\nSharpness"),
    ("💡", "Exposure /\nLighting"),
    ("🎯", "Texture /\nTracking"),
    ("🔄", "360°\nOrbit"),
]


# ══════════════════════════════════════════════════════════════════════════════
# Render helpers
# ══════════════════════════════════════════════════════════════════════════════

def render_header(analyzer) -> None:
    bundle = analyzer.bundle if hasattr(analyzer, "bundle") else {}
    n_feats      = len(bundle.get("feature_columns", [])) or 19
    defect_model = bundle.get("defect_model_name", "SoftVoting_Ensemble")
    binary_model = bundle.get("binary_model_name",  "XGBoost_Tuned")

    st.markdown(
        f"""
        <div class="pf-header">
          <h1>📦 PreFlight-3D</h1>
          <p>3D Reconstruction Capture Quality Gate</p>
          <div style="margin-top:12px; display:flex; gap:8px; flex-wrap:wrap;">
            <span class="pf-badge badge-blue">🧠 {defect_model}</span>
            <span class="pf-badge badge-purple">⚡ {binary_model}</span>
            <span class="pf-badge badge-blue">📐 {n_feats} CV features</span>
            <span class="pf-badge badge-green">🎥 Video Analysis</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_recommendations(recs: list[dict]) -> None:
    st.markdown('<div class="pf-section-title">Actionable Capture Recommendations</div>', unsafe_allow_html=True)

    if not recs:
        st.markdown(
            '<div style="background:#052e16;border:1px solid #166534;border-radius:8px;padding:14px 18px;">'
            '✅ <strong style="color:#4ade80;">All quality gates passed.</strong>'
            ' Your capture is optimised for 3D reconstruction.</div>',
            unsafe_allow_html=True,
        )
        return

    for rec in recs:
        sev     = rec.get("severity", "medium")
        sev_cls = f"sev-{sev.lower()}"
        conf    = rec.get("confidence", 0.0) * 100
        steps_html = "".join(f"<li>{s}</li>" for s in rec.get("action_steps", []))
        st.markdown(
            f"""
            <div class="rec-card {sev_cls}">
              <h4>{rec.get('title','Recommendation')}
                {_severity_badge(sev)}
                <span class="pf-badge badge-blue">{rec.get('category','')}</span>
              </h4>
              <p>🤖 Detected by <strong>{rec.get('detected_by','')}</strong>
                 &nbsp;·&nbsp; confidence <strong>{conf:.0f}%</strong></p>
              <p>⚠️ Impact: {rec.get('impact','')}</p>
              <ol>{steps_html}</ol>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_probabilities(defect_probs: dict[str, float]) -> None:
    st.markdown('<div class="pf-section-title">7-Class Defect Diagnosis</div>', unsafe_allow_html=True)
    sorted_probs = sorted(defect_probs.items(), key=lambda x: x[1], reverse=True)
    html = "".join(
        _prob_bar_html(
            label.replace("_", " ").title(),
            prob,
            _DEFECT_COLORS.get(label, "#3b82f6"),
        )
        for label, prob in sorted_probs
    )
    st.markdown(html, unsafe_allow_html=True)


def render_telemetry(metrics: dict) -> None:
    st.markdown('<div class="pf-section-title">CV & Trajectory Telemetry</div>', unsafe_allow_html=True)
    cols = st.columns(4)
    for idx, (key, label, unit) in enumerate(_METRIC_CFG):
        raw = metrics.get(key)
        if raw is None:
            continue
        if key in _PCT_KEYS:
            display = f"{raw * 100:.1f} %"
        elif key == "sampled_frames":
            display = str(int(raw))
        elif unit == "px":
            display = f"{raw:.1f} px"
        elif unit == "s":
            display = f"{raw:.2f} s"
        else:
            display = f"{raw:.3f}"
        with cols[idx % 4]:
            st.metric(label=label, value=display)


def render_diagnostic_report(result: dict) -> None:
    verdict  = result.get("verdict", "")
    score    = result.get("readiness_score", 0)
    summary  = result.get("summary", "")

    # Score + verdict
    st.markdown('<div class="pf-section-title">Readiness Score & Verdict</div>', unsafe_allow_html=True)
    score_col, _, verdict_col = st.columns([1, 0.1, 2.4])
    with score_col:
        st.metric("Readiness Score", f"{score} / 100")
    with verdict_col:
        st.markdown(_verdict_html(verdict, score, summary), unsafe_allow_html=True)

    st.divider()
    render_recommendations(result.get("recommendations", []))
    st.divider()
    render_probabilities(result.get("defect_probabilities", {}))
    st.divider()
    render_telemetry(result.get("extracted_metrics", {}))


# ══════════════════════════════════════════════════════════════════════════════
# Main layout
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    analyzer = load_analyzer()
    render_header(analyzer)

    left, right = st.columns([1, 1.6], gap="large")

    # ── Left column — Capture Input ────────────────────────────────────────
    with left:
        st.markdown('<div class="pf-section-title">Capture Input</div>', unsafe_allow_html=True)

        mode = st.radio(
            "Input Mode",
            ["📤 Upload Video", "🎬 Sample Video"],
            horizontal=True,
            label_visibility="collapsed",
        )

        video_path: str | None = None
        video_bytes = None

        if mode == "📤 Upload Video":
            uploaded = st.file_uploader(
                "Upload your capture video",
                type=["mp4", "mov", "mkv", "avi", "webm"],
                label_visibility="visible",
            )
            if uploaded:
                st.video(uploaded)
                video_bytes = uploaded.read()
                st.session_state["_pending_bytes"] = video_bytes
                st.session_state["_pending_ext"]   = os.path.splitext(uploaded.name)[-1] or ".mp4"
                video_path = "__uploaded__"

        else:  # Sample Video
            samples = _sample_video_paths()
            if not samples:
                st.warning("No sample videos found. Upload a video or run locally with the full dataset.")
            else:
                choice = st.selectbox(
                    "Choose a sample capture",
                    list(samples.keys()),
                    label_visibility="visible",
                )
                video_path = samples[choice]
                st.video(video_path)

        st.markdown('<div class="pf-section-title" style="margin-top:24px;">Inspection Pillars</div>',
                    unsafe_allow_html=True)
        pcols = st.columns(4)
        for col, (icon, label) in zip(pcols, PILLARS):
            with col:
                st.markdown(
                    f'<div class="pillar-card"><div class="icon">{icon}</div>'
                    f'<div class="label">{label}</div></div>',
                    unsafe_allow_html=True,
                )

        st.markdown("<br>", unsafe_allow_html=True)
        run_btn = st.button("🛫 Run Pre-Flight Check", use_container_width=True, type="primary")

    # ── Right column — Diagnostic Report ──────────────────────────────────
    with right:
        if run_btn:
            if video_path is None:
                st.warning("Please select or upload a video first.")
            elif video_path == "__uploaded__":
                # Write to temp file, analyse, clean up
                ext = st.session_state.get("_pending_ext", ".mp4")
                data = st.session_state.get("_pending_bytes", b"")
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
                try:
                    tmp.write(data)
                    tmp.flush()
                    tmp.close()
                    with st.spinner("Analysing uploaded video…"):
                        result = analyzer.analyze_video(tmp.name)
                    st.session_state["pf_result"] = result
                finally:
                    if os.path.exists(tmp.name):
                        os.unlink(tmp.name)
            else:
                with st.spinner("Analysing sample video…"):
                    result = analyzer.analyze_video(video_path)
                st.session_state["pf_result"] = result

        if "pf_result" in st.session_state:
            render_diagnostic_report(st.session_state["pf_result"])
        else:
            st.markdown(
                """
                <div style="display:flex;flex-direction:column;align-items:center;
                            justify-content:center;height:400px;color:#334155;
                            border:2px dashed #1e293b;border-radius:12px;gap:12px;">
                  <div style="font-size:3rem;">📦</div>
                  <div style="font-size:1.1rem;font-weight:600;">Awaiting capture video</div>
                  <div style="font-size:0.85rem;">Upload a video or choose a sample, then click <strong>Run Pre-Flight Check</strong></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ── Footer ─────────────────────────────────────────────────────────────
    st.markdown(
        """
        <div style="margin-top:40px;border-top:1px solid #1e293b;padding-top:16px;
                    text-align:center;color:#334155;font-size:0.78rem;">
          PreFlight-3D · SoftVoting_Ensemble + XGBoost_Tuned · UCO3D Dataset ·
          <a href="https://github.com/chandimacbbandara/preflight-3d"
             style="color:#3b82f6;text-decoration:none;">GitHub</a>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
