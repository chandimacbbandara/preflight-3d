"""
PreFlight-3D | Streamlit Cloud Entry Point
==========================================
Faithful port of the React frontend aesthetic.
Imports PreFlightAnalyzer directly — no FastAPI server required.
"""

import os
import glob
import tempfile
import math

import pandas as pd
import streamlit as st

# ── Page config (must be first Streamlit call) ─────────────────────────────
st.set_page_config(
    page_title="PreFlight-3D | 3D Reconstruction Quality Gate",
    page_icon="[3D]",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Design system CSS — mirrors React index.css + App.css ─────────────────
st.markdown(
    """
    <style>
    /* ══ Design tokens (mirrors React :root) ════════════════════════════════ */
    :root {
      --bg-primary:       #0b0f19;
      --bg-card:          #111827;
      --bg-elevated:      #1a2235;
      --border-subtle:    #263147;
      --text-primary:     #f3f4f6;
      --text-secondary:   #9ca3af;
      --text-muted:       #6b7280;
      --accent-blue:      #3b82f6;
      --accent-blue-dark: #2563eb;
      --status-good:      #10b981;
      --status-good-bg:   rgba(16,185,129,0.12);
      --status-warn:      #f59e0b;
      --status-warn-bg:   rgba(245,158,11,0.12);
      --status-bad:       #ef4444;
      --status-bad-bg:    rgba(239,68,68,0.12);
    }

    /* ══ Global reset ════════════════════════════════════════════════════════ */
    html, body, [data-testid="stAppViewContainer"],
    [data-testid="stMain"], [data-testid="block-container"] {
      background-color: var(--bg-primary) !important;
      color: var(--text-primary);
      font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      -webkit-font-smoothing: antialiased;
    }
    [data-testid="stSidebar"]  { display: none !important; }
    [data-testid="stToolbar"]  { display: none !important; }
    #MainMenu, footer          { visibility: hidden; }
    .stDeployButton            { display: none; }

    /* remove default Streamlit padding that fights our layout */
    [data-testid="block-container"] {
      padding: 0 !important;
      max-width: 100% !important;
    }

    /* ══ Topbar (sticky header) ══════════════════════════════════════════════ */
    .pf-topbar {
      background: rgba(17,24,39,0.92);
      backdrop-filter: blur(12px);
      border-bottom: 1px solid var(--border-subtle);
      padding: 0.85rem 2rem;
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 1rem;
      position: sticky;
      top: 0;
      z-index: 50;
    }
    .pf-brand {
      display: flex;
      align-items: center;
      gap: 0.75rem;
    }
    .pf-brand-icon {
      background: linear-gradient(135deg,#2563eb,#06b6d4);
      padding: 0.5rem;
      border-radius: 10px;
      display: flex;
      align-items: center;
      justify-content: center;
      color:#fff;
      box-shadow: 0 4px 14px rgba(37,99,235,0.35);
      font-size: 1.1rem;
      font-weight: 800;
      width: 36px;
      height: 36px;
    }
    .pf-brand h1 {
      font-size: 1.18rem;
      font-weight: 700;
      letter-spacing: -0.02em;
      margin: 0;
      color: var(--text-primary);
    }
    .pf-brand span {
      font-size: 0.78rem;
      color: var(--text-secondary);
      display: block;
    }
    .pf-nav {
      display: flex;
      align-items: center;
      gap: 0.4rem;
      background: var(--bg-primary);
      padding: 0.3rem;
      border-radius: 999px;
      border: 1px solid var(--border-subtle);
    }
    .pf-nav-btn {
      background: transparent;
      border: none;
      color: var(--text-secondary);
      padding: 0.45rem 1rem;
      border-radius: 999px;
      font-size: 0.84rem;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.16s ease;
      text-decoration: none;
    }
    .pf-nav-btn:hover { color: var(--text-primary); }
    .pf-nav-btn.active {
      background: linear-gradient(135deg,#2563eb,#0284c7);
      color: #fff;
      box-shadow: 0 2px 10px rgba(37,99,235,0.35);
    }
    .pf-model-pill {
      display: flex;
      align-items: center;
      gap: 0.5rem;
      background: var(--bg-elevated);
      border: 1px solid var(--border-subtle);
      padding: 0.4rem 0.85rem;
      border-radius: 999px;
      font-size: 0.78rem;
      color: var(--text-secondary);
    }
    .dot-online {
      width: 8px; height: 8px;
      border-radius: 50%;
      background: var(--status-good);
      box-shadow: 0 0 8px var(--status-good);
      display: inline-block;
    }

    /* ══ Page shells ═════════════════════════════════════════════════════════ */
    .pf-page-wrap {
      max-width: 1380px;
      margin: 0 auto;
      padding: 1.5rem 2rem 3.5rem;
    }
    @media(max-width:1024px) {
      .pf-page-wrap { padding: 1rem; }
      .pf-topbar    { padding: 0.85rem 1rem; }
    }

    /* ══ Card ════════════════════════════════════════════════════════════════ */
    .pf-card {
      background: rgba(17,24,39,0.92);
      border: 1px solid var(--border-subtle);
      border-radius: 14px;
      padding: 1.35rem;
      box-shadow: 0 8px 24px rgba(0,0,0,0.22);
      margin-bottom: 1.25rem;
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

    /* ══ Two-column main layout ══════════════════════════════════════════════ */
    .pf-main-grid {
      max-width: 1380px;
      margin: 0 auto;
      padding: 1.5rem 2rem 3rem;
      display: grid;
      grid-template-columns: 400px 1fr;
      gap: 1.5rem;
      align-items: start;
    }
    @media(max-width:1024px) {
      .pf-main-grid { grid-template-columns: 1fr; padding: 1rem; }
    }

    /* ══ Tab switcher ════════════════════════════════════════════════════════ */
    .pf-tabs {
      display: grid;
      grid-template-columns: 1fr 1fr;
      background: var(--bg-primary);
      padding: 0.25rem;
      border-radius: 10px;
      border: 1px solid var(--border-subtle);
      margin-bottom: 1.15rem;
      gap: 2px;
    }
    .pf-tab {
      background: transparent;
      border: none;
      color: var(--text-secondary);
      padding: 0.55rem 0.75rem;
      border-radius: 8px;
      font-size: 0.84rem;
      font-weight: 500;
      cursor: pointer;
      text-align: center;
      transition: all 0.15s ease;
    }
    .pf-tab.active {
      background: var(--bg-elevated);
      color: var(--text-primary);
      box-shadow: 0 1px 3px rgba(0,0,0,0.3);
    }

    /* ══ Dropzone ════════════════════════════════════════════════════════════ */
    .pf-dropzone {
      border: 2px dashed var(--border-subtle);
      border-radius: 12px;
      padding: 1.75rem 1rem;
      text-align: center;
      background: rgba(26,34,53,0.4);
      margin-bottom: 0.75rem;
    }
    .pf-dropzone p  { font-size:0.9rem; margin-top:0.6rem; font-weight:500; }
    .pf-dropzone small { display:block; margin-top:0.25rem; color:var(--text-muted); font-size:0.78rem; }

    /* ══ Buttons ═════════════════════════════════════════════════════════════ */
    .pf-primary-btn {
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 0.5rem;
      width: 100%;
      padding: 0.75rem 1rem;
      background: linear-gradient(135deg,#2563eb,#0284c7);
      color: #fff;
      border: none;
      border-radius: 10px;
      font-size: 0.92rem;
      font-weight: 600;
      cursor: pointer;
      transition: opacity 0.15s, transform 0.15s;
      text-decoration: none;
      margin-top: 1rem;
    }
    .pf-primary-btn:hover { opacity:0.95; transform:translateY(-1px); }

    .pf-secondary-btn {
      display: inline-flex;
      align-items: center;
      gap: 0.5rem;
      padding: 0.7rem 1.2rem;
      border-radius: 10px;
      border: 1px solid rgba(59,130,246,0.45);
      background: rgba(59,130,246,0.12);
      color: #93c5fd;
      font-size: 0.88rem;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.15s ease;
      text-decoration: none;
    }
    .pf-secondary-btn:hover { background:rgba(59,130,246,0.22); color:#fff; }

    .pf-ghost-btn {
      width: 100%;
      background: var(--bg-elevated);
      border: 1px solid var(--border-subtle);
      color: #93c5fd;
      padding: 0.58rem 0.85rem;
      border-radius: 9px;
      font-size: 0.81rem;
      font-weight: 600;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: space-between;
      text-decoration: none;
      margin-top: 1rem;
    }
    .pf-ghost-btn:hover { border-color:var(--accent-blue); color:#fff; }

    /* ══ Verdict card ════════════════════════════════════════════════════════ */
    .pf-verdict-card {
      display: flex;
      align-items: center;
      gap: 1.5rem;
      flex-wrap: wrap;
      background: rgba(17,24,39,0.92);
      border: 1px solid var(--border-subtle);
      border-radius: 14px;
      padding: 1.35rem;
      box-shadow: 0 8px 24px rgba(0,0,0,0.22);
      margin-bottom: 1.25rem;
    }
    /* Radial SVG ring */
    .pf-radial-wrap { position:relative; width:118px; height:118px; flex-shrink:0; }
    .pf-radial-svg  { width:100%; height:100%; transform:rotate(-90deg); }
    .pf-radial-track    { fill:none; stroke:var(--bg-elevated); stroke-width:9; }
    .pf-radial-progress { fill:none; stroke-width:9; stroke-linecap:round; }
    .pf-radial-progress.good { stroke:var(--status-good); }
    .pf-radial-progress.warn { stroke:var(--status-warn); }
    .pf-radial-progress.bad  { stroke:var(--status-bad);  }
    .pf-radial-center {
      position:absolute; inset:0;
      display:flex; flex-direction:column; align-items:center; justify-content:center;
    }
    .pf-score-num   { font-size:1.9rem; font-weight:800; line-height:1; }
    .pf-score-label {
      font-size:0.66rem; text-transform:uppercase; letter-spacing:0.06em;
      color:var(--text-secondary); margin-top:0.25rem;
    }

    /* verdict badge */
    .pf-verdict-badge {
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
    .pf-verdict-badge.good { background:var(--status-good-bg); color:#34d399; }
    .pf-verdict-badge.warn { background:var(--status-warn-bg); color:#fbbf24; }
    .pf-verdict-badge.bad  { background:var(--status-bad-bg);  color:#f87171; }

    .pf-verdict-details { flex:1; min-width:240px; }
    .pf-verdict-details h2 { font-size:1.05rem; margin-bottom:0.35rem; word-break:break-all; }
    .pf-verdict-details p  { font-size:0.9rem; color:var(--text-secondary); margin:0; }

    /* ══ 4-Pillar grid ═══════════════════════════════════════════════════════ */
    .pf-pillars-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(175px,1fr));
      gap: 0.85rem;
    }
    .pf-pillar-card {
      background: var(--bg-elevated);
      border: 1px solid var(--border-subtle);
      border-radius: 10px;
      padding: 0.85rem;
    }
    .pf-pillar-top {
      display: flex; justify-content:space-between; align-items:center;
      font-size:0.76rem; color:var(--text-secondary);
    }
    .pf-pillar-num { font-size:1.25rem; font-weight:700; display:block; margin:0.35rem 0 0.5rem; }
    .pf-pillar-num.good { color:#34d399; }
    .pf-pillar-num.warn { color:#fbbf24; }
    .pf-pillar-num.bad  { color:#f87171; }

    /* ══ Probability bars ════════════════════════════════════════════════════ */
    .pf-prob-list { display:flex; flex-direction:column; gap:0.65rem; }
    .pf-prob-row  {
      display: grid;
      grid-template-columns: 185px 1fr 56px;
      align-items: center;
      gap: 0.75rem;
      font-size: 0.83rem;
    }
    .pf-prob-name  { color:var(--text-secondary); text-transform:capitalize; }
    .pf-prob-track { height:9px; background:var(--bg-primary); border-radius:999px; overflow:hidden; }
    .pf-prob-fill  {
      height:100%; border-radius:999px;
      background:var(--accent-blue); transition:width 0.35s ease;
    }
    .pf-prob-fill.clean { background:var(--status-good); }
    .pf-prob-fill.alert { background:var(--status-bad);  }
    .pf-prob-val   { text-align:right; color:var(--text-secondary); }

    /* ══ Recommendations ═════════════════════════════════════════════════════ */
    .pf-rec-item {
      background: var(--bg-elevated);
      border: 1px solid var(--border-subtle);
      border-left: 4px solid var(--status-bad);
      border-radius: 10px;
      padding: 1rem 1.1rem;
      margin-bottom: 0.9rem;
    }
    .pf-rec-item.MEDIUM { border-left-color: var(--status-warn); }
    .pf-rec-header {
      display:flex; justify-content:space-between;
      align-items:flex-start; gap:0.75rem; flex-wrap:wrap; margin-bottom:0.4rem;
    }
    .pf-rec-header h4 { font-size:0.96rem; font-weight:600; margin:0; }
    .pf-rec-tags { display:flex; gap:0.45rem; flex-wrap:wrap; }
    .pf-tag {
      font-size:0.72rem; padding:0.2rem 0.55rem; border-radius:6px;
      background:var(--bg-primary); color:var(--text-secondary);
      border:1px solid var(--border-subtle);
    }
    .pf-rec-impact { font-size:0.85rem; color:var(--text-secondary); margin-bottom:0.65rem; }
    .pf-rec-steps  { padding-left:1.2rem; font-size:0.86rem; }
    .pf-rec-steps li { margin-bottom:0.3rem; }

    .pf-pass-banner {
      background: var(--status-good-bg);
      border: 1px solid rgba(16,185,129,0.3);
      border-radius: 10px; padding:1rem;
      display:flex; align-items:center; gap:0.75rem;
      color:#6ee7b7; font-size:0.9rem;
    }

    /* ══ Metrics grid ════════════════════════════════════════════════════════ */
    .pf-metrics-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(165px,1fr));
      gap: 0.75rem;
    }
    .pf-metric-tile {
      background: var(--bg-elevated);
      border: 1px solid var(--border-subtle);
      border-radius: 10px;
      padding: 0.85rem;
    }
    .pf-metric-top {
      display:flex; justify-content:space-between;
      align-items:center; margin-bottom:0.4rem;
      font-size:0.76rem; color:var(--text-secondary);
    }
    .pf-metric-cat { color:var(--text-muted); font-size:0.68rem; }
    .pf-metric-val { font-size:1.05rem; font-weight:700; color:var(--text-primary); }

    /* ══ Error / info boxes ══════════════════════════════════════════════════ */
    .pf-error-box {
      margin-top:0.9rem; padding:0.75rem; border-radius:10px;
      background:var(--status-bad-bg); border:1px solid rgba(239,68,68,0.35);
      color:#fca5a5; font-size:0.83rem;
    }
    .pf-info-box {
      background:var(--bg-elevated); border:1px solid var(--border-subtle);
      border-radius:10px; padding:0.75rem; font-size:0.8rem;
      color:var(--text-secondary); margin-bottom:1rem;
    }
    .pf-sample-hint {
      font-size:0.8rem; color:var(--text-secondary);
      background:var(--bg-elevated); padding:0.75rem;
      border-radius:10px; margin-bottom:1rem;
      border:1px solid var(--border-subtle);
    }

    /* ══ Pillar legend (sidebar) ═════════════════════════════════════════════ */
    .pf-pillar-legend { display:flex; flex-direction:column; gap:0.6rem;
                        font-size:0.82rem; color:var(--text-secondary); }
    .pf-pillar-legend > div { display:flex; align-items:center; gap:0.55rem; }
    .pf-pillar-legend strong { color:var(--text-primary); }

    /* ══ Empty hero state ════════════════════════════════════════════════════ */
    .pf-empty-hero {
      padding: 2.6rem 2.2rem;
      background: linear-gradient(160deg,rgba(26,34,53,0.9),rgba(17,24,39,0.96));
      border: 1px solid var(--border-subtle); border-radius:14px;
    }
    .pf-empty-badge {
      display:inline-flex; align-items:center; gap:0.4rem;
      font-size:0.76rem; font-weight:600; color:#60a5fa;
      background:rgba(59,130,246,0.12); border:1px solid rgba(59,130,246,0.3);
      padding:0.28rem 0.7rem; border-radius:999px; margin-bottom:1rem;
    }
    .pf-empty-hero h2 { font-size:1.5rem; letter-spacing:-0.02em; line-height:1.25; margin-bottom:0.75rem; }
    .pf-empty-hero p  { color:var(--text-secondary); font-size:0.94rem; max-width:66ch; }
    .pf-empty-steps {
      display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr));
      gap:0.85rem; margin:1.65rem 0;
    }
    .pf-es-step {
      background:var(--bg-primary); border:1px solid var(--border-subtle);
      border-radius:12px; padding:0.95rem;
      display:flex; flex-direction:column; gap:0.25rem;
    }
    .pf-es-step span   { font-size:0.72rem; font-weight:700; color:#60a5fa; }
    .pf-es-step strong { font-size:0.9rem; color:var(--text-primary); }
    .pf-es-step small  { font-size:0.77rem; color:var(--text-secondary); }

    /* ══ How It Works — Hero ════════════════════════════════════════════════ */
    .hiw-hero-top { margin-bottom: 1.5rem; }
    .hiw-badge {
      display:inline-flex; align-items:center; gap:0.4rem;
      font-size:0.76rem; font-weight:600; color:#60a5fa;
      background:rgba(59,130,246,0.12); border:1px solid rgba(59,130,246,0.3);
      padding:0.28rem 0.7rem; border-radius:999px; margin-bottom:1rem;
    }
    .hiw-hero-top h1 { font-size:1.9rem; letter-spacing:-0.03em; line-height:1.2; margin-bottom:0.75rem; }
    .gradient-text {
      background: linear-gradient(90deg,#3b82f6,#06b6d4);
      -webkit-background-clip: text; -webkit-text-fill-color:transparent;
    }
    .hiw-hero-sub { color:var(--text-secondary); font-size:1rem; max-width:72ch; margin-bottom:1.4rem; }
    .hiw-kpi-strip {
      display:grid; grid-template-columns:repeat(auto-fit,minmax(140px,1fr));
      gap:0.75rem; padding-top:1.25rem; border-top:1px solid var(--border-subtle);
    }
    .kpi-tile { text-align:center; }
    .kpi-label { font-size:0.72rem; color:var(--text-muted); display:block; }
    .kpi-value { font-size:1.6rem; font-weight:800; color:var(--accent-blue); display:block; margin:0.1rem 0; }
    .kpi-sub   { font-size:0.72rem; color:var(--text-secondary); }

    /* ══ Problem / Solution ══════════════════════════════════════════════════ */
    .pf-ps-grid {
      display:grid; grid-template-columns:1fr 1fr; gap:1.25rem; margin-bottom:1.25rem;
    }
    @media(max-width:800px) { .pf-ps-grid { grid-template-columns:1fr; } }
    .pf-ps-card { background:rgba(17,24,39,0.92); border:1px solid var(--border-subtle);
                  border-radius:14px; padding:1.35rem; }
    .pf-ps-header { display:flex; align-items:flex-start; gap:1rem; margin-bottom:1.1rem; }
    .pf-ps-icon {
      width:42px; height:42px; border-radius:10px; flex-shrink:0;
      display:flex; align-items:center; justify-content:center; font-size:1.1rem;
    }
    .pf-ps-icon.bad  { background:var(--status-bad-bg);  color:var(--status-bad);  border:1px solid rgba(239,68,68,0.3); }
    .pf-ps-icon.good { background:var(--status-good-bg); color:var(--status-good); border:1px solid rgba(16,185,129,0.3); }
    .pf-ps-eyebrow { font-size:0.72rem; font-weight:600; text-transform:uppercase;
                     letter-spacing:0.08em; color:var(--text-muted); display:block; }
    .pf-ps-card h2 { font-size:1.05rem; margin:0.25rem 0 0; }
    .pf-ps-item { margin-bottom:0.9rem; }
    .pf-ps-item h4 { font-size:0.9rem; font-weight:600; margin-bottom:0.25rem; }
    .pf-ps-item p  { font-size:0.83rem; color:var(--text-secondary); }

    /* ══ GIF showcase ════════════════════════════════════════════════════════ */
    .pf-gif-grid {
      display:grid; grid-template-columns:1fr 1fr; gap:1rem; margin-top:1rem;
    }
    @media(max-width:800px) { .pf-gif-grid { grid-template-columns:1fr; } }
    .pf-gif-card { border-radius:12px; overflow:hidden; border:1px solid var(--border-subtle); }
    .pf-gif-card img { width:100%; display:block; max-height:260px; object-fit:cover; }
    .pf-gif-caption { background:var(--bg-elevated); padding:0.75rem 1rem; }
    .pf-gif-caption strong { font-size:0.85rem; display:block; margin-bottom:0.25rem; }
    .pf-gif-caption span   { font-size:0.78rem; color:var(--text-secondary); }

    /* ══ Pipeline stage stepper ══════════════════════════════════════════════ */
    .pf-stage-stepper { display:flex; flex-wrap:wrap; gap:0.5rem; margin-bottom:1rem; }
    .pf-stage-pill {
      display:flex; align-items:center; gap:0.45rem;
      background:var(--bg-elevated); border:1px solid var(--border-subtle);
      border-radius:999px; padding:0.4rem 0.9rem;
      font-size:0.82rem; font-weight:600; cursor:pointer;
      transition: all 0.15s ease; color:var(--text-secondary);
    }
    .pf-stage-pill:hover { border-color:var(--accent-blue); color:var(--text-primary); }
    .pf-stage-pill.active {
      background:linear-gradient(135deg,#2563eb,#0284c7);
      color:#fff; border-color:transparent;
      box-shadow:0 2px 10px rgba(37,99,235,0.35);
    }
    .pf-stage-letter { font-size:0.78rem; font-weight:800; }

    .pf-stage-detail {
      background:linear-gradient(135deg,rgba(37,99,235,0.08),rgba(2,132,199,0.05));
      border:1px solid rgba(59,130,246,0.25); border-radius:12px;
      padding:1.25rem; margin-bottom:1rem;
    }
    .pf-stage-detail-head { display:flex; align-items:flex-start; gap:1rem; margin-bottom:0.75rem; }
    .pf-stage-badge-lg {
      background:linear-gradient(135deg,#2563eb,#0284c7);
      color:#fff; font-size:0.76rem; font-weight:800; border-radius:8px;
      padding:0.4rem 0.75rem; flex-shrink:0; white-space:nowrap;
    }
    .pf-stage-detail-head h3 { font-size:1.05rem; margin:0 0 0.25rem; }
    .pf-script-chip {
      font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,monospace;
      font-size:0.76rem; color:#93c5fd; background:rgba(59,130,246,0.12);
      padding:0.2rem 0.6rem; border-radius:6px; border:1px solid rgba(59,130,246,0.25);
    }
    .pf-stage-summary { font-size:0.88rem; color:var(--text-secondary); margin-bottom:0.9rem; }
    .pf-hl-chips { display:flex; flex-wrap:wrap; gap:0.5rem; }
    .pf-hl-chip {
      display:inline-flex; align-items:center; gap:0.35rem;
      font-size:0.76rem; font-weight:500; color:#34d399;
      background:rgba(16,185,129,0.1); border:1px solid rgba(16,185,129,0.25);
      padding:0.25rem 0.65rem; border-radius:999px;
    }

    .pf-timeline-grid {
      display:grid; grid-template-columns:repeat(auto-fill,minmax(220px,1fr));
      gap:0.85rem;
    }
    .pf-timeline-mini {
      background:var(--bg-elevated); border:1px solid var(--border-subtle);
      border-radius:10px; padding:0.9rem; cursor:pointer;
      transition: border-color 0.15s, background 0.15s;
    }
    .pf-timeline-mini:hover { border-color:var(--accent-blue); }
    .pf-timeline-mini.selected { border-color:var(--accent-blue); background:rgba(59,130,246,0.08); }
    .pf-tm-top { display:flex; justify-content:space-between; align-items:center; margin-bottom:0.4rem; }
    .pf-tm-id { font-size:0.72rem; font-weight:700; color:#60a5fa; }
    .pf-timeline-mini h4 { font-size:0.85rem; font-weight:600; margin-bottom:0.3rem; }
    .pf-timeline-mini p  { font-size:0.76rem; color:var(--text-secondary); line-height:1.4; }

    /* ══ Taxonomy grid ═══════════════════════════════════════════════════════ */
    .pf-taxonomy-grid {
      display:grid; grid-template-columns:repeat(auto-fill,minmax(200px,1fr));
      gap:0.75rem; margin-bottom:1.25rem;
    }
    .pf-tax-card {
      background:var(--bg-elevated); border:1px solid var(--border-subtle);
      border-radius:10px; padding:0.9rem;
    }
    .pf-tax-card h4 { font-size:0.85rem; font-weight:700; margin-bottom:0.4rem; }
    .pf-tax-items   { font-size:0.78rem; color:var(--text-secondary); margin-bottom:0.5rem; }
    .pf-tax-challenge { font-size:0.72rem; color:var(--text-muted); font-style:italic; }

    /* ══ Data table ══════════════════════════════════════════════════════════ */
    .pf-table-scroll { overflow-x:auto; border-radius:10px; border:1px solid var(--border-subtle); }
    .pf-data-table { width:100%; border-collapse:collapse; font-size:0.82rem; }
    .pf-data-table th {
      background:var(--bg-elevated); padding:0.65rem 0.85rem;
      text-align:left; color:var(--text-secondary); font-weight:600;
      border-bottom:1px solid var(--border-subtle); white-space:nowrap;
    }
    .pf-data-table td {
      padding:0.6rem 0.85rem; border-bottom:1px solid var(--border-subtle);
      color:var(--text-primary);
    }
    .pf-data-table tr:last-child td { border-bottom:none; }
    .pf-data-table tr.winner-row td { background:rgba(59,130,246,0.07); }
    .pf-mono-strong { font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,monospace;
                      font-weight:600; }
    .pf-win-pill {
      display:inline-block; font-size:0.68rem; font-weight:700;
      background:linear-gradient(135deg,#2563eb,#0284c7); color:#fff;
      padding:0.15rem 0.5rem; border-radius:999px; margin-left:0.4rem;
    }
    .pf-subtable-title { font-size:0.88rem; font-weight:600;
                          margin:1.25rem 0 0.75rem; color:var(--text-primary); }

    /* ══ Winner cards ════════════════════════════════════════════════════════ */
    .pf-winner-cards-grid { display:grid; grid-template-columns:1fr 1fr; gap:1rem; margin-bottom:1.25rem; }
    @media(max-width:700px) { .pf-winner-cards-grid { grid-template-columns:1fr; } }
    .pf-winner-card {
      background:var(--bg-elevated); border:1px solid var(--border-subtle);
      border-radius:12px; padding:1.1rem;
    }
    .pf-winner-head { display:flex; justify-content:space-between; align-items:center; margin-bottom:0.6rem; }
    .pf-winner-tag {
      font-size:0.72rem; font-weight:700; text-transform:uppercase; letter-spacing:0.06em;
      color:#60a5fa; background:rgba(59,130,246,0.12);
      padding:0.2rem 0.65rem; border-radius:999px;
    }
    .pf-winner-tag.green { color:#34d399; background:rgba(16,185,129,0.12); }
    .pf-winner-card h3 { font-size:1.1rem; font-weight:700; margin-bottom:0.3rem; }
    .pf-winner-sub { font-size:0.78rem; color:var(--text-secondary); margin-bottom:0.75rem; }
    .pf-winner-metrics { display:grid; grid-template-columns:1fr 1fr; gap:0.5rem; }
    .pf-winner-metrics > div { background:var(--bg-primary); border-radius:8px; padding:0.6rem 0.8rem; }
    .pf-winner-metrics span   { font-size:0.72rem; color:var(--text-secondary); display:block; }
    .pf-winner-metrics strong { font-size:0.95rem; font-weight:700; display:block; }

    /* ══ Two-col section ═════════════════════════════════════════════════════ */
    .pf-two-col { display:grid; grid-template-columns:1fr 1fr; gap:1.25rem; margin-bottom:1.25rem; }
    @media(max-width:800px) { .pf-two-col { grid-template-columns:1fr; } }

    /* ══ Output gallery ══════════════════════════════════════════════════════ */
    .pf-output-gallery {
      display:grid; grid-template-columns:repeat(auto-fill,minmax(280px,1fr));
      gap:1rem;
    }
    .pf-output-card {
      background:var(--bg-elevated); border:1px solid var(--border-subtle);
      border-radius:12px; overflow:hidden;
    }
    .pf-output-card img { width:100%; display:block; max-height:200px; object-fit:cover; }
    .pf-output-body { padding:0.9rem; }
    .pf-output-meta-row { display:flex; align-items:center; gap:0.5rem;
                           flex-wrap:wrap; margin-bottom:0.5rem; }
    .pf-output-card h4 { font-size:0.85rem; margin-bottom:0.3rem; }
    .pf-output-card p  { font-size:0.78rem; color:var(--text-secondary); }

    /* ══ Benchmark tab switcher (compact) ════════════════════════════════════ */
    .pf-compact-tabs {
      display:flex; gap:0.4rem; flex-wrap:wrap;
    }
    .pf-compact-tab {
      background:var(--bg-elevated); border:1px solid var(--border-subtle);
      color:var(--text-secondary); padding:0.45rem 0.85rem; border-radius:8px;
      font-size:0.82rem; font-weight:500; cursor:pointer; transition:all 0.15s;
    }
    .pf-compact-tab.active {
      background:var(--bg-primary); color:var(--text-primary);
      box-shadow:0 1px 3px rgba(0,0,0,0.3); border-color:var(--accent-blue);
    }

    /* ══ Section divider row ═════════════════════════════════════════════════ */
    .pf-section-head { margin-bottom:1rem; }
    .pf-section-head-row { display:flex; justify-content:space-between; align-items:flex-start;
                            flex-wrap:wrap; gap:0.75rem; margin-bottom:1rem; }
    .pf-section-desc { font-size:0.85rem; color:var(--text-secondary);
                        margin-top:0.4rem; max-width:70ch; }

    /* ══ hiw bar rows ════════════════════════════════════════════════════════ */
    .pf-hiw-bar-row { grid-template-columns:160px 1fr 56px; }

    /* ══ Streamlit widget overrides ══════════════════════════════════════════ */
    /* hide default st.radio label bullets + make it look like our tabs */
    div[data-testid="stRadio"] > div {
      display: flex !important;
      flex-direction: row !important;
      background: var(--bg-primary);
      padding: 0.25rem;
      border-radius: 10px;
      border: 1px solid var(--border-subtle);
      gap: 2px;
    }
    div[data-testid="stRadio"] label {
      flex: 1;
      text-align: center;
      padding: 0.55rem 0.75rem !important;
      border-radius: 8px;
      font-size: 0.84rem !important;
      font-weight: 500 !important;
      color: var(--text-secondary) !important;
      cursor: pointer;
      transition: all 0.15s ease;
    }
    div[data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) {
      background: var(--bg-elevated);
      color: var(--text-primary) !important;
      box-shadow: 0 1px 3px rgba(0,0,0,0.3);
    }
    /* radio dot — hide it */
    div[data-testid="stRadio"] [data-baseweb="radio"] > div:first-child { display: none !important; }

    /* selectbox */
    div[data-testid="stSelectbox"] > div > div {
      background: var(--bg-elevated) !important;
      border: 1px solid var(--border-subtle) !important;
      border-radius: 10px !important;
      color: var(--text-primary) !important;
    }

    /* file uploader */
    div[data-testid="stFileUploaderDropzone"] {
      background: rgba(26,34,53,0.4) !important;
      border: 2px dashed var(--border-subtle) !important;
      border-radius: 12px !important;
    }
    div[data-testid="stFileUploaderDropzone"]:hover {
      border-color: var(--accent-blue) !important;
      background: rgba(59,130,246,0.08) !important;
    }

    /* primary action button */
    div[data-testid="stButton"] > button[kind="primary"] {
      background: linear-gradient(135deg,#2563eb,#0284c7) !important;
      border: none !important;
      border-radius: 10px !important;
      font-weight: 600 !important;
      font-size: 0.92rem !important;
      padding: 0.75rem 1rem !important;
      color: #fff !important;
      width: 100% !important;
      transition: opacity 0.15s, transform 0.15s !important;
    }
    div[data-testid="stButton"] > button[kind="primary"]:hover {
      opacity: 0.95 !important;
      transform: translateY(-1px) !important;
    }

    /* secondary buttons */
    div[data-testid="stButton"] > button[kind="secondary"] {
      background: rgba(59,130,246,0.12) !important;
      border: 1px solid rgba(59,130,246,0.45) !important;
      border-radius: 10px !important;
      color: #93c5fd !important;
      font-weight: 600 !important;
    }
    div[data-testid="stButton"] > button[kind="secondary"]:hover {
      background: rgba(59,130,246,0.22) !important;
      color: #fff !important;
    }

    /* metric value color */
    [data-testid="stMetricValue"] { color: var(--text-primary) !important; }
    [data-testid="stMetricLabel"] { color: var(--text-secondary) !important; font-size:0.76rem !important; }

    /* spinner */
    div[data-testid="stSpinner"] p { color: var(--text-secondary) !important; }

    /* video */
    video { border-radius: 10px; }

    /* hide st.columns gap default backgrounds */
    [data-testid="column"] { background: transparent !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ══════════════════════════════════════════════════════════════════════════════
# Data — mirrors frontend/src/data/projectData.js
# ══════════════════════════════════════════════════════════════════════════════

HERO_STATS = [
    {"label": "uCO3D Objects",     "value": "40",     "sub": "40 diverse categories"},
    {"label": "Labeled Clips",     "value": "280",    "sub": "40 clean + 240 degraded"},
    {"label": "CV & Orbit Features","value": "19",    "sub": "~0.5s CPU extraction"},
    {"label": "7-Class CV Acc",    "value": "77.14%", "sub": "SoftVoting_Ensemble"},
    {"label": "Binary Gate CV Acc","value": "86.79%", "sub": "XGBoost_Tuned"},
]

PROBLEMS_LIST = [
    {
        "title": "Silent Compute Waste in 3D Pipelines",
        "detail": "COLMAP, NeRF, and 3D Gaussian Splatting often run for 20–90 minutes before failing or producing hollow meshes, floating floaters, and torn textures.",
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
        "detail": "Inspects sharpness (Laplacian variance & Canny edges), HSV glare & dark pixel ratio, ORB texture density, Farneback optical flow, and start-to-end RANSAC loop closure in ~0.3–0.5s on CPU.",
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
    {"id": "A", "tag": "Setup",      "title": "Toolkit & Environment Setup",
     "script": "scripts/setup_uco3d.sh",
     "summary": "Installs OpenCV, NumPy, Pandas, PyYAML, tqdm, and trimesh, and clones Meta's official facebookresearch/uco3d toolkit into external/uco3d.",
     "highlights": ["CPU-only Ubuntu environment", "12 GB RAM constraint", "Zero GPU required"]},
    {"id": "B", "tag": "Ingestion",  "title": "Selective uCO3D Subset Acquisition",
     "script": "ml_pipeline/dataset_builder/01_download_subset.py",
     "summary": "Downloads examples.zip (8.71 GB) from Hugging Face (facebook/uco3d), selectively extracts only .mp4, .mkv, .ply, .sqlite, and .json files, and deletes the archive.",
     "highlights": ["52 categories / 52 sequences", "~75% disk space saved", "available_categories.csv"]},
    {"id": "C", "tag": "Curation",   "title": "40-Category Curation & Extraction",
     "script": "ml_pipeline/dataset_builder/02_select_and_organize.py",
     "summary": "Selects 40 diverse sequences across 5 material/geometry groups (Glossy, Metallic, Thin/Complex, Organic Matte, Rigid), symlinks RGB videos, extracts 2 FPS frames & binary masks, and copies .ply point clouds.",
     "highlights": ["40 object categories", "7,999 camera annotation rows", "21.15 min total source footage"]},
    {"id": "D", "tag": "3D Audit",   "title": "Baseline 3D & Quality Audit",
     "script": "ml_pipeline/dataset_builder/03_inspect_and_measure.py",
     "summary": "Audits all 40 clean sequences against their foreground masks and .ply reconstructions to verify clean ground-truth baselines before synthetic degradation.",
     "highlights": ["1.000 object visible ratio", "18,360.7 mean segmented 3D pts", "16,047.5 mean sparse SfM pts"]},
    {"id": "E", "tag": "Simulation", "title": "Synthetic Capture Failure Generation",
     "script": "ml_pipeline/dataset_builder/04_generate_degraded_videos.py",
     "summary": "Generates 6 degraded variants per clean sequence at max_dim=640 (240 POOR clips + 40 clean GOOD clips = 280 labeled videos) covering blur, lighting, glare, flicker, overlap, and orbit completeness.",
     "highlights": ["motion_blur (19x19 kernel)", "low_light (gamma 2.5 + noise)", "harsh_glare (alpha=1.55 + mask hotspot)", "exposure_flicker (1.45x/0.55x oscillation)", "low_overlap_fast_move (67% frame drop)", "incomplete_orbit (35% cutoff)"]},
    {"id": "F", "tag": "Features",   "title": "19-Feature CV & Trajectory Extraction",
     "script": "ml_pipeline/feature_extraction/extractor.py",
     "summary": "Samples at 2.5 FPS (min 12 frames), resizes to 480px, and computes 19 sharpness, lighting, glare, ORB texture, Farneback optical flow, RANSAC match/inlier, and loop-closure features inside an elliptical subject ROI.",
     "highlights": ["19 standardized features", "~0.3-0.5s per video on CPU", "extracted_features.csv"]},
    {"id": "G", "tag": "Training",   "title": "Group-Aware Training, Tuning & Voting",
     "script": "ml_pipeline/training/train_models.py",
     "summary": "Evaluates Random Forest, XGBoost, and LightGBM (Baselines, RandomizedSearchCV Tuned, and SoftVoting_Ensemble) under 5-Fold StratifiedGroupKFold grouped by sequence_id across 7-class defect diagnosis and binary readiness.",
     "highlights": ["Zero object leakage (32 train / 8 holdout objects)", "7-Class Winner: SoftVoting_Ensemble (77.14% CV)", "Binary Winner: XGBoost_Tuned (86.79% CV)"]},
    {"id": "H", "tag": "Inference",  "title": "Hybrid Physics Gate & Calibration",
     "script": "backend/app/services/recommendation_engine.py",
     "summary": "Gates ML predictions with deterministic physics verification, geometric & trajectory guardrails, a 4-pillar CV health score, and calibrated 0-100 Readiness Verdicts with actionable fixes.",
     "highlights": ["Suppresses false turntable/indoor alerts", "Catches extreme low-overlap & short orbits", "READY / WARNINGS / RECAPTURE verdicts"]},
    {"id": "I", "tag": "Full-Stack", "title": "FastAPI Service & React 19 Dashboard",
     "script": "backend/app/main.py & frontend/src/App.jsx",
     "summary": "Exposes /api/health, /api/analyze, /api/samples, and /api/analyze-sample/{id} to power an interactive diagnostic studio with drag-and-drop video upload, benchmark sampling, probability bars, and live telemetry.",
     "highlights": ["FastAPI + Uvicorn (:8000)", "React 19 + Vite (:5173)", "14 curated benchmark samples"]},
]

TAXONOMY_GROUPS = [
    {"group": "Transparent & Glossy (5)", "items": "wineglass, water_bottle, perfume, vase, ashtray",
     "challenge": "Refraction & view-dependent specular highlights"},
    {"group": "Metallic & Reflective (8)", "items": "bass_horn, pot, flashlight, aerosol_can, fire_extinguisher, motor, binoculars, cooker",
     "challenge": "Glare clipping & mirror-like low-feature regions"},
    {"group": "Thin & Complex Geometry (6)", "items": "shopping_cart, chair, helicopter, speaker_stero_equipment, telephone, funnel",
     "challenge": "Thin wires/rotors causing SfM tracking dropouts"},
    {"group": "Organic & High-Texture Matte (12)", "items": "shoe, teddy_bear, pinecone, handbag, pillow, hand_towel, shirt, cap_headwear, muffin, bagel, onion, orange_fruit",
     "challenge": "Strong textured reconstruction baselines"},
    {"group": "Rigid Everyday Objects (9)", "items": "book, mouse_computer_equipment, flowerpot, sculpture, first-aid_kit, battery, dog_collar, dustpan, scraper",
     "challenge": "Varied scales, flat sides & everyday indoor lighting"},
]

BENCHMARK_RESULTS = {
    "Defect_Diagnosis_7Class": [
        {"stage": "1_Baseline", "model": "RandomForest",         "cvAcc": "73.93%", "cvStd": "0.0571", "cvF1": "0.7267", "holdAcc": "73.21%", "holdF1": "0.7430", "winner": False},
        {"stage": "1_Baseline", "model": "XGBoost",              "cvAcc": "76.07%", "cvStd": "0.0720", "cvF1": "0.7516", "holdAcc": "76.79%", "holdF1": "0.7640", "winner": False},
        {"stage": "1_Baseline", "model": "LightGBM",             "cvAcc": "75.36%", "cvStd": "0.0602", "cvF1": "0.7470", "holdAcc": "75.00%", "holdF1": "0.7577", "winner": False},
        {"stage": "2_Tuned",    "model": "RandomForest_Tuned",   "cvAcc": "76.07%", "cvStd": "0.0702", "cvF1": "0.7491", "holdAcc": "76.79%", "holdF1": "0.7617", "winner": False},
        {"stage": "2_Tuned",    "model": "XGBoost_Tuned",        "cvAcc": "76.43%", "cvStd": "0.0398", "cvF1": "0.7607", "holdAcc": "73.21%", "holdF1": "0.7287", "winner": False},
        {"stage": "2_Tuned",    "model": "LightGBM_Tuned",       "cvAcc": "76.43%", "cvStd": "0.0633", "cvF1": "0.7619", "holdAcc": "78.57%", "holdF1": "0.7990", "winner": False},
        {"stage": "3_Ensemble", "model": "SoftVoting_Ensemble",  "cvAcc": "77.14%", "cvStd": "0.0522", "cvF1": "0.7655", "holdAcc": "76.79%", "holdF1": "0.7721", "winner": True},
    ],
    "Readiness_Binary": [
        {"stage": "1_Baseline", "model": "RandomForest",         "cvAcc": "83.21%", "cvStd": "0.0182", "cvF1": "0.7021", "holdAcc": "80.36%", "holdF1": "0.7168", "winner": False},
        {"stage": "1_Baseline", "model": "XGBoost",              "cvAcc": "86.07%", "cvStd": "0.0208", "cvF1": "0.7150", "holdAcc": "87.50%", "holdF1": "0.7573", "winner": False},
        {"stage": "1_Baseline", "model": "LightGBM",             "cvAcc": "83.93%", "cvStd": "0.0226", "cvF1": "0.6809", "holdAcc": "85.71%", "holdF1": "0.7352", "winner": False},
        {"stage": "2_Tuned",    "model": "RandomForest_Tuned",   "cvAcc": "84.64%", "cvStd": "0.0182", "cvF1": "0.7268", "holdAcc": "83.93%", "holdF1": "0.7363", "winner": False},
        {"stage": "2_Tuned",    "model": "XGBoost_Tuned",        "cvAcc": "86.79%", "cvStd": "0.0143", "cvF1": "0.7302", "holdAcc": "85.71%", "holdF1": "0.7352", "winner": True},
        {"stage": "2_Tuned",    "model": "LightGBM_Tuned",       "cvAcc": "85.36%", "cvStd": "0.0262", "cvF1": "0.6975", "holdAcc": "85.71%", "holdF1": "0.7352", "winner": False},
        {"stage": "3_Ensemble", "model": "SoftVoting_Ensemble",  "cvAcc": "85.00%", "cvStd": "0.0182", "cvF1": "0.7092", "holdAcc": "85.71%", "holdF1": "0.7352", "winner": False},
    ],
}

HOLDOUT_CLASS_REPORT = [
    {"cls": "harsh_glare",          "f1": 1.000},
    {"cls": "incomplete_orbit",     "f1": 0.933},
    {"cls": "low_light",            "f1": 0.933},
    {"cls": "motion_blur",          "f1": 0.857},
    {"cls": "exposure_flicker",     "f1": 0.824},
    {"cls": "none (Clean)",         "f1": 0.571},
    {"cls": "low_overlap_fast_move","f1": 0.286},
]

TOP_FEATURES = [
    {"name": "dark_pixel_ratio",          "importance": 0.1574, "role": "Underexposure & shadow clipping"},
    {"name": "brightness_mean",           "importance": 0.1503, "role": "Global exposure & highlight blowout"},
    {"name": "laplacian_var_mean",        "importance": 0.0898, "role": "Mean subject sharpness vs. motion blur"},
    {"name": "duration_sec",              "importance": 0.0842, "role": "Capture pacing & truncated orbits"},
    {"name": "brightness_step_diff_mean", "importance": 0.0795, "role": "Auto-exposure flicker & pumping"},
    {"name": "specular_glare_ratio",      "importance": 0.0794, "role": "Glare hotspots (V>240 & S<45)"},
    {"name": "sampled_frames",            "importance": 0.0629, "role": "Temporal frame coverage"},
    {"name": "brightness_std",            "importance": 0.0604, "role": "Luminance variance across orbit"},
    {"name": "laplacian_var_p10",         "importance": 0.0499, "role": "Worst-decile frame blur"},
    {"name": "blurry_frame_ratio",        "importance": 0.0418, "role": "Share of frames below sharpness floor"},
]

FEATURE_CLASS_MEANS = [
    {"defect": "none (Clean)",          "duration": "31.73", "laplacian": "655.00", "blurry": "0.005", "brightness": "119.45", "stepDiff": "3.99",  "dark": "0.094", "glare": "0.007", "orb": "452.84", "loop": "0.533"},
    {"defect": "motion_blur",           "duration": "31.73", "laplacian": "47.99",  "blurry": "0.530", "brightness": "117.58", "stepDiff": "3.98",  "dark": "0.085", "glare": "0.005", "orb": "139.14", "loop": "0.355"},
    {"defect": "low_light",             "duration": "31.73", "laplacian": "449.13", "blurry": "0.000", "brightness": "57.71",  "stepDiff": "3.42",  "dark": "0.452", "glare": "0.004", "orb": "385.48", "loop": "0.512"},
    {"defect": "harsh_glare",           "duration": "31.73", "laplacian": "173.53", "blurry": "0.239", "brightness": "223.55", "stepDiff": "3.08",  "dark": "0.000", "glare": "0.597", "orb": "251.80", "loop": "0.381"},
    {"defect": "exposure_flicker",      "duration": "31.73", "laplacian": "431.26", "blurry": "0.079", "brightness": "123.49", "stepDiff": "76.58", "dark": "0.126", "glare": "0.110", "orb": "395.43", "loop": "0.532"},
    {"defect": "low_overlap_fast_move", "duration": "31.76", "laplacian": "364.79", "blurry": "0.021", "brightness": "117.63", "stepDiff": "3.97",  "dark": "0.099", "glare": "0.007", "orb": "436.04", "loop": "0.521"},
    {"defect": "incomplete_orbit",      "duration": "11.12", "laplacian": "395.63", "blurry": "0.020", "brightness": "117.11", "stepDiff": "3.71",  "dark": "0.102", "glare": "0.007", "orb": "444.03", "loop": "0.095"},
]

FINAL_OUTPUT_GALLERY = [
    {"src": "/img/57939c0f-a5c6-4f4f-a697-8239f34f55d0.png",
     "file": "1-34403-89911.mp4.down.mp4", "score": 54, "verdict": "USABLE WITH WARNINGS", "tier": "warn",
     "trigger": "Geometric Guardrail (72%)",
     "summary": "Potted plant capture (28.7s). Even though the raw ML classifier leaned Clean (81.5%), Peak Optical Flow spiked to 25.49 px — the deterministic Geometric Guardrail caught insufficient frame overlap and flagged fast panning."},
    {"src": "/img/a3122948-cbd7-4800-af0d-147b602b41b8.png",
     "file": "1-41035-27122.mp4.down.mp4", "score": 68, "verdict": "USABLE WITH WARNINGS", "tier": "warn",
     "trigger": "ML Model (26%)",
     "summary": "Textured woven object capture (28.7s, 88.6% RANSAC inliers). The Soft-Voting Ensemble detected 26.1% Low Overlap Fast Move (Peak Optical Flow 18.63 px) and recommended maintaining >70% frame overlap."},
    {"src": "/img/ddd3a8f2-305d-417a-881e-74e3b66e599c.png",
     "file": "WhatsApp Video 2026-11-19 at 23.12.35.mp4", "score": 29, "verdict": "RECAPTURE RECOMMENDED", "tier": "bad",
     "trigger": "ML Model (33%)",
     "summary": "Real smartphone capture (2.83s, 0.32 MB). Dark shadows pushed Mean Brightness down to 75.2/255 with 22.3% Underexposed Pixels, triggering a High-Severity Low-Light warning and actionable lighting fixes."},
]

METRIC_LABELS = [
    {"key": "duration_sec",              "label": "Duration (sec)",         "unit": "s",   "pct": False, "cat": "Orbit"},
    {"key": "sampled_frames",            "label": "Sampled Frames",         "unit": "",    "pct": False, "cat": "Orbit"},
    {"key": "laplacian_var_mean",        "label": "Sharpness (Laplacian)",  "unit": "",    "pct": False, "cat": "Focus"},
    {"key": "blurry_frame_ratio",        "label": "Blurry Frame Ratio",     "unit": "%",   "pct": True,  "cat": "Focus"},
    {"key": "edge_density_mean",         "label": "Edge Density",           "unit": "%",   "pct": True,  "cat": "Focus"},
    {"key": "brightness_mean",           "label": "Mean Brightness",        "unit": "/255","pct": False, "cat": "Light"},
    {"key": "brightness_step_diff_mean", "label": "Exposure Jump",          "unit": "",    "pct": False, "cat": "Light"},
    {"key": "dark_pixel_ratio",          "label": "Underexposed Pixels",    "unit": "%",   "pct": True,  "cat": "Light"},
    {"key": "specular_glare_ratio",      "label": "Specular Glare Ratio",   "unit": "%",   "pct": True,  "cat": "Light"},
    {"key": "orb_keypoints_mean",        "label": "Avg ORB Keypoints",      "unit": "",    "pct": False, "cat": "Track"},
    {"key": "optical_flow_mean",         "label": "Mean Optical Flow",      "unit": " px", "pct": False, "cat": "Track"},
    {"key": "optical_flow_max",          "label": "Peak Optical Flow",      "unit": " px", "pct": False, "cat": "Track"},
    {"key": "pair_match_ratio_mean",     "label": "Feature Match Ratio",    "unit": "%",   "pct": True,  "cat": "Track"},
    {"key": "pair_inlier_ratio_mean",    "label": "RANSAC Inlier Ratio",    "unit": "%",   "pct": True,  "cat": "Track"},
    {"key": "cumulative_pan_px",         "label": "Cumulative Orbit Pan",   "unit": " px", "pct": False, "cat": "Orbit"},
    {"key": "loop_closure_inlier_ratio", "label": "Loop Closure Inliers",   "unit": "%",   "pct": True,  "cat": "Orbit"},
]


# ══════════════════════════════════════════════════════════════════════════════
# Utilities
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading PreFlight-3D analyzer…")
def load_analyzer():
    from backend.app.services.recommendation_engine import PreFlightAnalyzer
    return PreFlightAnalyzer(bundle_path="models/preflight_model_bundle.joblib")


def sample_video_paths() -> dict[str, str]:
    paths: dict[str, str] = {}
    for p in sorted(glob.glob("sample_videos/*.mp4")):
        name = os.path.splitext(os.path.basename(p))[0].replace("_", " ").replace("  ", " — ")
        paths[name] = os.path.abspath(p)
    if not paths:
        manifest = "data/processed/quality_measurements/dataset_manifest.csv"
        if os.path.exists(manifest):
            df = pd.read_csv(manifest)
            if "video_path" in df.columns:
                for _, row in df.iterrows():
                    vp = row["video_path"]
                    if os.path.exists(vp):
                        paths[os.path.splitext(os.path.basename(vp))[0]] = os.path.abspath(vp)
    return paths


def verdict_tier(verdict: str) -> str:
    if verdict == "READY_FOR_RECONSTRUCTION":  return "good"
    if verdict == "USABLE_WITH_WARNINGS":       return "warn"
    return "bad"


def format_defect(code: str) -> str:
    return "Clean Capture (No Defect)" if code == "none" else code.replace("_", " ").title()


def radial_svg(score: int, tier: str) -> str:
    r = 48
    circ = 2 * math.pi * r
    offset = circ - (max(0, min(100, score)) / 100) * circ
    return f"""
    <div class="pf-radial-wrap">
      <svg class="pf-radial-svg" viewBox="0 0 120 120">
        <circle class="pf-radial-track" cx="60" cy="60" r="{r}"/>
        <circle class="pf-radial-progress {tier}" cx="60" cy="60" r="{r}"
          stroke-dasharray="{circ:.3f}" stroke-dashoffset="{offset:.3f}"/>
      </svg>
      <div class="pf-radial-center">
        <span class="pf-score-num">{score}</span>
        <span class="pf-score-label">Readiness</span>
      </div>
    </div>"""


def compute_pillar_scores(m: dict) -> list[dict]:
    sharpness = round(
        min(1, (m.get("laplacian_var_mean", 0) or 0) / 150) *
        (1 - min(1, m.get("blurry_frame_ratio", 0) or 0)) * 100
    )
    tracking = round(
        (0.65 * min(1, (m.get("pair_inlier_ratio_mean", 0) or 0) / 0.65) +
         0.35 * min(1, (m.get("orb_keypoints_mean", 0) or 0) / 250)) * 100
    )
    b = m.get("brightness_mean", 0) or 0
    if 75 <= b <= 190:    bright_c = 1.0
    elif b < 75:          bright_c = max(0, b / 75)
    else:                 bright_c = max(0, (255 - b) / 65)
    glare_c   = max(0, 1 - (m.get("specular_glare_ratio", 0) or 0) * 3)
    flicker_c = max(0, 1 - max(0, ((m.get("brightness_step_diff_mean", 0) or 0) - 8) / 28))
    lighting  = round(bright_c * glare_c * flicker_c * 100)

    dur_f = min(1, (m.get("duration_sec", 0) or 0) / 12)
    cov_f = max(
        min(1, (m.get("loop_closure_inlier_ratio", 0) or 0) / 0.45),
        min(1, (m.get("cumulative_pan_px", 0) or 0) / 30)
    )
    orbit_raw = 0.5 * dur_f + 0.5 * cov_f
    if (m.get("loop_closure_inlier_ratio", 0) or 0) >= 0.5 and (m.get("optical_flow_mean", 0) or 0) >= 2:
        orbit_raw = max(orbit_raw, cov_f * 0.95)
    orbit = round(orbit_raw * 100)

    return [
        {"key": "Focus", "label": "Focus & Sharpness",   "score": sharpness, "weight": "25%"},
        {"key": "Track", "label": "Tracking & Overlap",  "score": tracking,  "weight": "30%"},
        {"key": "Light", "label": "Lighting & Glare",    "score": lighting,  "weight": "25%"},
        {"key": "Orbit", "label": "360 Orbit Coverage",  "score": orbit,     "weight": "20%"},
    ]


def metric_display(raw, cfg: dict) -> str:
    if raw is None:
        return "—"
    if cfg["pct"]:
        return f"{raw * 100:.1f}{cfg['unit']}"
    if cfg["key"] == "sampled_frames":
        return str(int(raw))
    return f"{raw}{cfg['unit']}"


# ══════════════════════════════════════════════════════════════════════════════
# Topbar (page navigation)
# ══════════════════════════════════════════════════════════════════════════════

if "page" not in st.session_state:
    st.session_state["page"] = "analyzer"

analyzer = load_analyzer()
bundle      = analyzer.bundle if hasattr(analyzer, "bundle") else {}
n_feats     = len(bundle.get("feature_columns", [])) or 19
defect_model = bundle.get("defect_model_name", "SoftVoting_Ensemble")
binary_model = bundle.get("binary_model_name",  "XGBoost_Tuned")

page = st.session_state["page"]

ana_active = "active" if page == "analyzer"    else ""
hiw_active = "active" if page == "how-it-works" else ""

st.markdown(
    f"""
    <div class="pf-topbar">
      <div class="pf-brand">
        <div class="pf-brand-icon">3D</div>
        <div>
          <h1>PreFlight-3D</h1>
          <span>Photogrammetry &amp; 3DGS Video Quality Gate</span>
        </div>
      </div>
      <div class="pf-nav">
        <span class="pf-nav-btn {ana_active}" id="nav-analyzer">Pre-Flight Analyzer</span>
        <span class="pf-nav-btn {hiw_active}" id="nav-hiw">How It Works (A-Z)</span>
      </div>
      <div class="pf-model-pill">
        <span class="dot-online"></span>
        <span>{defect_model} + {binary_model} &bull; {n_feats} CV features</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Real navigation buttons hidden behind a visual overlay
nav_col1, nav_col2 = st.columns([1, 1])
with nav_col1:
    if st.button("Pre-Flight Analyzer", key="nav_analyzer_btn", use_container_width=True):
        st.session_state["page"] = "analyzer"
        st.rerun()
with nav_col2:
    if st.button("How It Works (A-Z)", key="nav_hiw_btn", use_container_width=True):
        st.session_state["page"] = "how-it-works"
        st.rerun()

# Hide the real nav buttons (they just handle routing; topbar CSS handles visuals)
st.markdown(
    """
    <style>
    /* hide the routing buttons — only the topbar HTML nav is shown */
    div[data-testid="stHorizontalBlock"]:nth-of-type(1) button {
      position: absolute !important;
      opacity: 0 !important;
      pointer-events: none !important;
      height: 1px !important;
      overflow: hidden !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Pre-Flight Analyzer
# ══════════════════════════════════════════════════════════════════════════════

def render_analyzer() -> None:
    st.markdown('<div class="pf-main-grid">', unsafe_allow_html=True)

    left, right = st.columns([4, 6], gap="large")

    # ── Left sidebar ─────────────────────────────────────────────────────────
    with left:
        # Capture Input card
        st.markdown('<div class="pf-card">', unsafe_allow_html=True)
        st.markdown(
            '<div class="pf-card-title">'
            '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#9ca3af" stroke-width="2"><rect x="2" y="2" width="20" height="20" rx="3"/><path d="M7 2v20M17 2v20M2 12h20M2 7h5M2 17h5M17 7h5M17 17h5"/></svg>'
            '<span>Capture Input</span></div>',
            unsafe_allow_html=True,
        )

        mode = st.radio(
            "Input Mode",
            ["Upload Video", "Test Dataset"],
            horizontal=True,
            label_visibility="collapsed",
        )

        video_path: str | None = None

        if mode == "Upload Video":
            st.markdown(
                '<div class="pf-dropzone">'
                '<p>Click to select or drag & drop a capture video</p>'
                '<small>Supports MP4, MOV, MKV, AVI, WEBM</small>'
                '</div>',
                unsafe_allow_html=True,
            )
            uploaded = st.file_uploader(
                "Upload capture video",
                type=["mp4", "mov", "mkv", "avi", "webm"],
                label_visibility="collapsed",
            )
            if uploaded:
                st.video(uploaded)
                st.session_state["_pending_bytes"] = uploaded.read()
                st.session_state["_pending_ext"]   = os.path.splitext(uploaded.name)[-1] or ".mp4"
                st.session_state["_pending_name"]  = uploaded.name
                video_path = "__uploaded__"

        else:  # Test Dataset
            samples = sample_video_paths()
            if not samples:
                st.markdown(
                    '<div class="pf-error-box">No sample videos found. Upload a video or run locally with the full dataset.</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<p style="font-size:0.82rem;color:var(--text-secondary);margin-bottom:0.45rem;">'
                    'Select a sample capture video:</p>',
                    unsafe_allow_html=True,
                )
                choice = st.selectbox(
                    "Sample video",
                    list(samples.keys()),
                    label_visibility="collapsed",
                )
                video_path = samples[choice]
                st.video(video_path)
                st.markdown(
                    '<div class="pf-sample-hint">Runs live 19-feature CV extraction and Soft-Voting Ensemble + Physics Gate inference.</div>',
                    unsafe_allow_html=True,
                )

        run_btn = st.button(
            "Run Pre-Flight Check",
            type="primary",
            use_container_width=True,
        )

        if st.session_state.get("_pf_error"):
            st.markdown(
                f'<div class="pf-error-box">{st.session_state["_pf_error"]}</div>',
                unsafe_allow_html=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)

        # 4-Pillar legend card
        st.markdown(
            """
            <div class="pf-card">
              <div class="pf-card-title">
                <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="#9ca3af" stroke-width="2"><rect x="2" y="2" width="8" height="8" rx="1"/><rect x="14" y="2" width="8" height="8" rx="1"/><rect x="2" y="14" width="8" height="8" rx="1"/><rect x="14" y="14" width="8" height="8" rx="1"/></svg>
                <span>4 Inspection Pillars</span>
              </div>
              <div class="pf-pillar-legend">
                <div>
                  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#60a5fa" stroke-width="2"><circle cx="12" cy="12" r="3"/><path d="M12 2v3M12 19v3M2 12h3M19 12h3"/></svg>
                  <span><strong>Sharpness:</strong> Laplacian &amp; Canny edges</span>
                </div>
                <div>
                  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#fbbf24" stroke-width="2"><circle cx="12" cy="12" r="5"/><path d="M12 2v2M12 20v2M2 12h2M20 12h2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M4.93 19.07l1.41-1.41M17.66 6.34l1.41-1.41"/></svg>
                  <span><strong>Exposure:</strong> Brightness, flicker &amp; glare</span>
                </div>
                <div>
                  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#34d399" stroke-width="2"><path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z"/></svg>
                  <span><strong>Texture:</strong> ORB keypoints &amp; RANSAC matches</span>
                </div>
                <div>
                  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#a78bfa" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"/><path d="M2 12h20"/></svg>
                  <span><strong>360 Orbit:</strong> Optical flow &amp; loop closure</span>
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Right results panel ───────────────────────────────────────────────────
    with right:
        # Handle run
        if run_btn:
            st.session_state.pop("_pf_error", None)
            if video_path is None:
                st.session_state["_pf_error"] = "Please select or upload a video first."
            elif video_path == "__uploaded__":
                data = st.session_state.get("_pending_bytes", b"")
                ext  = st.session_state.get("_pending_ext", ".mp4")
                tmp  = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
                try:
                    tmp.write(data); tmp.flush(); tmp.close()
                    with st.spinner("Analysing frames & geometry…"):
                        result = analyzer.analyze_video(tmp.name)
                    # Store original filename
                    result["video_file"] = st.session_state.get("_pending_name", result.get("video_file", ""))
                    st.session_state["pf_result"] = result
                except Exception as exc:
                    st.session_state["_pf_error"] = str(exc)
                finally:
                    if os.path.exists(tmp.name):
                        os.unlink(tmp.name)
            else:
                try:
                    with st.spinner("Running live inference…"):
                        result = analyzer.analyze_video(video_path)
                    st.session_state["pf_result"] = result
                except Exception as exc:
                    st.session_state["_pf_error"] = str(exc)
            st.rerun()

        result = st.session_state.get("pf_result")

        if not result:
            # Empty hero state
            st.markdown(
                """
                <div class="pf-empty-hero">
                  <div class="pf-empty-badge">Ready for Pre-Flight Inspection</div>
                  <h2>Catch 3D Reconstruction Failures Before Running COLMAP or 3DGS</h2>
                  <p>Upload an object-centric orbit video or choose a sample to inspect motion blur,
                     lighting stability, specular glare, and 360 loop closure in under a second.</p>
                  <div class="pf-empty-steps">
                    <div class="pf-es-step">
                      <span>01</span><strong>Sample @ 2.5 FPS</strong>
                      <small>Elliptical subject ROI masking at 480px</small>
                    </div>
                    <div class="pf-es-step">
                      <span>02</span><strong>19 CV Features</strong>
                      <small>Sharpness, glare, ORB, Farneback & RANSAC</small>
                    </div>
                    <div class="pf-es-step">
                      <span>03</span><strong>Ensemble + Physics</strong>
                      <small>SoftVoting + XGBoost_Tuned + guardrails</small>
                    </div>
                    <div class="pf-es-step">
                      <span>04</span><strong>0-100 Readiness</strong>
                      <small>Actionable fixes before GPU compute</small>
                    </div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            tier = verdict_tier(result.get("verdict", ""))

            # ── Verdict card ──────────────────────────────────────────────────
            icon_map = {"good": "&#10003;", "warn": "&#9888;", "bad": "&#10007;"}
            icon = icon_map.get(tier, "")
            friendly = result.get("verdict", "").replace("_", " ")
            st.markdown(
                f"""
                <div class="pf-verdict-card">
                  {radial_svg(result.get("readiness_score", 0), tier)}
                  <div class="pf-verdict-details">
                    <div class="pf-verdict-badge {tier}">{icon} {friendly}</div>
                    <h2>{result.get("video_file", "")}</h2>
                    <p>{result.get("summary", "")}</p>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # ── 4-Pillar breakdown ────────────────────────────────────────────
            pillars = compute_pillar_scores(result.get("extracted_metrics", {}))
            pillar_html = ""
            for p in pillars:
                pt = "good" if p["score"] >= 70 else "warn" if p["score"] >= 45 else "bad"
                fill_cls = "clean" if pt == "good" else "alert" if pt == "bad" else "warn-fill"
                pillar_html += f"""
                <div class="pf-pillar-card">
                  <div class="pf-pillar-top"><span>{p["label"]}</span><small>Wt {p["weight"]}</small></div>
                  <span class="pf-pillar-num {pt}">{p["score"]}%</span>
                  <div class="pf-prob-track">
                    <div class="pf-prob-fill {fill_cls}" style="width:{p['score']}%"></div>
                  </div>
                </div>"""
            st.markdown(
                f"""
                <div class="pf-card">
                  <div class="pf-card-title">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#9ca3af" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="M12 8v4l3 3"/></svg>
                    <span>4-Pillar Capture Health Breakdown</span>
                  </div>
                  <div class="pf-pillars-grid">{pillar_html}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # ── Recommendations ───────────────────────────────────────────────
            recs = result.get("recommendations", [])
            recs_html = ""
            if not recs:
                recs_html = (
                    '<div class="pf-pass-banner">'
                    '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#6ee7b7" stroke-width="2.5"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>'
                    '<span>All quality gates passed. Frame sharpness, uniform lighting, and 360 orbit overlap are strong enough for 3D reconstruction.</span>'
                    '</div>'
                )
            else:
                for rec in recs:
                    sev_cls = "MEDIUM" if rec.get("severity", "HIGH").upper() == "MEDIUM" else ""
                    conf = round((rec.get("confidence", 0.0)) * 100)
                    steps = "".join(f"<li>{s}</li>" for s in rec.get("action_steps", []))
                    recs_html += f"""
                    <div class="pf-rec-item {sev_cls}">
                      <div class="pf-rec-header">
                        <h4>{rec.get("title","")}</h4>
                        <div class="pf-rec-tags">
                          <span class="pf-tag">{rec.get("category","")}</span>
                          <span class="pf-tag">{rec.get("detected_by","")} ({conf}%)</span>
                        </div>
                      </div>
                      <p class="pf-rec-impact">{rec.get("impact","")}</p>
                      <ol class="pf-rec-steps">{steps}</ol>
                    </div>"""
            st.markdown(
                f"""
                <div class="pf-card">
                  <div class="pf-card-title">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#9ca3af" stroke-width="2"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>
                    <span>Actionable Capture Recommendations</span>
                  </div>
                  {recs_html}
                </div>
                """,
                unsafe_allow_html=True,
            )

            # ── Defect probabilities ──────────────────────────────────────────
            probs = sorted(result.get("defect_probabilities", {}).items(), key=lambda x: x[1], reverse=True)
            prob_rows_html = ""
            for cls, prob in probs:
                pct = round(prob * 1000) / 10
                fill_cls = "clean" if cls == "none" else ("alert" if prob >= 0.25 else "")
                prob_rows_html += f"""
                <div class="pf-prob-row">
                  <span class="pf-prob-name">{format_defect(cls)}</span>
                  <div class="pf-prob-track">
                    <div class="pf-prob-fill {fill_cls}" style="width:{min(pct,100)}%"></div>
                  </div>
                  <span class="pf-prob-val">{pct:.1f}%</span>
                </div>"""
            st.markdown(
                f"""
                <div class="pf-card">
                  <div class="pf-card-title">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#9ca3af" stroke-width="2"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>
                    <span>Soft-Voting Ensemble Diagnosis Probabilities</span>
                  </div>
                  <div class="pf-prob-list">{prob_rows_html}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # ── CV Telemetry grid ─────────────────────────────────────────────
            m = result.get("extracted_metrics", {})
            tiles_html = ""
            for cfg in METRIC_LABELS:
                raw = m.get(cfg["key"])
                tiles_html += f"""
                <div class="pf-metric-tile">
                  <div class="pf-metric-top">
                    <span>{cfg["label"]}</span>
                    <small class="pf-metric-cat">{cfg["cat"]}</small>
                  </div>
                  <div class="pf-metric-val">{metric_display(raw, cfg)}</div>
                </div>"""
            st.markdown(
                f"""
                <div class="pf-card">
                  <div class="pf-card-title">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#9ca3af" stroke-width="2"><line x1="4" y1="21" x2="4" y2="14"/><line x1="4" y1="10" x2="4" y2="3"/><line x1="12" y1="21" x2="12" y2="12"/><line x1="12" y1="8" x2="12" y2="3"/><line x1="20" y1="21" x2="20" y2="16"/><line x1="20" y1="12" x2="20" y2="3"/><line x1="1" y1="14" x2="7" y2="14"/><line x1="9" y1="8" x2="15" y2="8"/><line x1="17" y1="16" x2="23" y2="16"/></svg>
                    <span>Extracted Computer Vision &amp; Trajectory Telemetry</span>
                  </div>
                  <div class="pf-metrics-grid">{tiles_html}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: How It Works
# ══════════════════════════════════════════════════════════════════════════════

def render_how_it_works() -> None:
    st.markdown('<div class="pf-page-wrap">', unsafe_allow_html=True)

    # ── Hero ──────────────────────────────────────────────────────────────────
    kpi_html = "".join(
        f'<div class="kpi-tile"><span class="kpi-label">{s["label"]}</span>'
        f'<strong class="kpi-value">{s["value"]}</strong>'
        f'<small class="kpi-sub">{s["sub"]}</small></div>'
        for s in HERO_STATS
    )
    st.markdown(
        f"""
        <div class="pf-card">
          <div class="hiw-hero-top">
            <div class="hiw-badge">A-to-Z Engineering Architecture &amp; Benchmark Report</div>
            <h1>Pre-Flight Quality Gate for <span class="gradient-text">3D Reconstruction</span></h1>
            <p class="hiw-hero-sub">
              PreFlight-3D evaluates smartphone and turntable object-capture videos in ~0.5 seconds
              on CPU — catching motion blur, underexposure, specular glare, exposure flicker, sparse
              overlap, and incomplete 360 orbits before you commit heavy compute to COLMAP, NeRF, or
              3D Gaussian Splatting.
            </p>
          </div>
          <div class="hiw-kpi-strip">{kpi_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Try Analyzer button
    if st.button("Try Live Analyzer", key="hiw_goto_analyzer", type="secondary"):
        st.session_state["page"] = "analyzer"
        st.rerun()

    # ── Problem vs Solution ───────────────────────────────────────────────────
    prob_html = "".join(
        f'<div class="pf-ps-item"><h4>{p["title"]}</h4><p>{p["detail"]}</p></div>'
        for p in PROBLEMS_LIST
    )
    sol_html = "".join(
        f'<div class="pf-ps-item"><h4>{s["title"]}</h4><p>{s["detail"]}</p></div>'
        for s in SOLUTIONS_LIST
    )
    st.markdown(
        f"""
        <div class="pf-ps-grid">
          <div class="pf-ps-card">
            <div class="pf-ps-header">
              <div class="pf-ps-icon bad">&#9888;</div>
              <div>
                <span class="pf-ps-eyebrow">The Problem</span>
                <h2>Why 3D Reconstructions Fail Silently</h2>
              </div>
            </div>
            <div class="pf-ps-list">{prob_html}</div>
          </div>
          <div class="pf-ps-card">
            <div class="pf-ps-header">
              <div class="pf-ps-icon good">&#10003;</div>
              <div>
                <span class="pf-ps-eyebrow">Our Solution</span>
                <h2>Hybrid CV + Ensemble + Physics Gate</h2>
              </div>
            </div>
            <div class="pf-ps-list">{sol_html}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Visual GIF Showcase ───────────────────────────────────────────────────
    st.markdown(
        """
        <div class="pf-card">
          <div class="pf-section-head">
            <div class="pf-card-title">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#9ca3af" stroke-width="2"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/></svg>
              <span>Visual Dataset &amp; Failure Simulation Showcase</span>
            </div>
            <p class="pf-section-desc">Real 360 captures curated from Meta's uCO3D dataset alongside live PreFlight-3D inference scores across all 7 diagnostic classes.</p>
          </div>
          <div class="pf-gif-grid">
            <div class="pf-gif-card">
              <img src="/selected_samples_showcase.gif" alt="Selected uCO3D Benchmark Captures Showcase"/>
              <div class="pf-gif-caption">
                <strong>Selected uCO3D Object Taxonomy (6 Clean Samples)</strong>
                <span>Covers Glossy (Vase), Metallic (Fire Extinguisher), Complex (Helicopter), Machinery (Motor), Matte (Handbag), and Rigid (Battery) captures — scoring 95-98/100 READY.</span>
              </div>
            </div>
            <div class="pf-gif-card">
              <img src="/defect_modes_comparison.gif" alt="Clean vs 6 Synthetic Capture Defects Comparison"/>
              <div class="pf-gif-caption">
                <strong>Clean vs. 6 Synthetic Defects (Vase 47433-58842-22217)</strong>
                <span>Live hybrid inference separates clean capture (98/100 PASS) from Motion Blur, Low Light, Harsh Glare, Exposure Flicker, Low Overlap (23/100), and Incomplete Orbit (stopped at 35%).</span>
              </div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── A-to-Z Pipeline stepper ───────────────────────────────────────────────
    if "active_stage" not in st.session_state:
        st.session_state["active_stage"] = "A"

    stage_pills_html = ""
    for stage in PIPELINE_STAGES:
        active_cls = "active" if st.session_state["active_stage"] == stage["id"] else ""
        stage_pills_html += (
            f'<button class="pf-stage-pill {active_cls}" '
            f'onclick="void(0)">'
            f'<span class="pf-stage-letter">{stage["id"]}</span>'
            f'<span>{stage["tag"]}</span></button>'
        )

    st.markdown(
        f"""
        <div class="pf-card">
          <div class="pf-section-head">
            <div class="pf-card-title">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#9ca3af" stroke-width="2"><line x1="6" y1="3" x2="6" y2="15"/><circle cx="18" cy="6" r="3"/><circle cx="6" cy="18" r="3"/><path d="M18 9a9 9 0 0 1-9 9"/></svg>
              <span>A-to-Z End-to-End Pipeline (Stages A to I)</span>
            </div>
            <p class="pf-section-desc">Select any stage below to inspect the exact scripts, design decisions, and verified outputs built across the project.</p>
          </div>
          <div class="pf-stage-stepper">{stage_pills_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Stage selector (Streamlit-native for interactivity)
    stage_ids = [s["id"] for s in PIPELINE_STAGES]
    stage_labels = [f"{s['id']}  {s['tag']}" for s in PIPELINE_STAGES]
    sel_idx = stage_ids.index(st.session_state["active_stage"]) if st.session_state["active_stage"] in stage_ids else 0
    chosen = st.selectbox(
        "Select Pipeline Stage",
        stage_labels,
        index=sel_idx,
        label_visibility="collapsed",
    )
    chosen_id = chosen.split()[0]
    st.session_state["active_stage"] = chosen_id
    active_stage = next(s for s in PIPELINE_STAGES if s["id"] == chosen_id)

    hl_html = "".join(
        f'<span class="pf-hl-chip">&#10003; {hl}</span>'
        for hl in active_stage["highlights"]
    )
    st.markdown(
        f"""
        <div class="pf-stage-detail">
          <div class="pf-stage-detail-head">
            <div class="pf-stage-badge-lg">Stage {active_stage["id"]}</div>
            <div>
              <h3>{active_stage["title"]}</h3>
              <span class="pf-script-chip">{active_stage["script"]}</span>
            </div>
          </div>
          <p class="pf-stage-summary">{active_stage["summary"]}</p>
          <div class="pf-hl-chips">{hl_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Mini cards grid for all stages
    mini_html = ""
    for stage in PIPELINE_STAGES:
        sel_cls = "selected" if stage["id"] == chosen_id else ""
        mini_html += f"""
        <div class="pf-timeline-mini {sel_cls}">
          <div class="pf-tm-top">
            <span class="pf-tm-id">Stage {stage["id"]}</span>
            <span class="pf-tag">{stage["tag"]}</span>
          </div>
          <h4>{stage["title"]}</h4>
          <p>{stage["summary"][:100]}...</p>
        </div>"""
    st.markdown(
        f'<div class="pf-timeline-grid">{mini_html}</div>',
        unsafe_allow_html=True,
    )

    # ── 40-Category Taxonomy ──────────────────────────────────────────────────
    tax_html = "".join(
        f'<div class="pf-tax-card"><h4>{g["group"]}</h4>'
        f'<p class="pf-tax-items">{g["items"]}</p>'
        f'<span class="pf-tax-challenge">Edge case: {g["challenge"]}</span></div>'
        for g in TAXONOMY_GROUPS
    )
    # Feature class means table
    means_rows = "".join(
        f'<tr><td class="pf-mono-strong">{r["defect"]}</td>'
        f'<td>{r["duration"]}</td><td>{r["laplacian"]}</td><td>{r["blurry"]}</td>'
        f'<td>{r["brightness"]}</td><td>{r["stepDiff"]}</td><td>{r["dark"]}</td>'
        f'<td>{r["glare"]}</td><td>{r["orb"]}</td><td>{r["loop"]}</td></tr>'
        for r in FEATURE_CLASS_MEANS
    )
    st.markdown(
        f"""
        <div class="pf-card">
          <div class="pf-section-head">
            <div class="pf-card-title">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#9ca3af" stroke-width="2"><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/></svg>
              <span>uCO3D 40-Category Curation &amp; Extracted Feature Signatures</span>
            </div>
            <p class="pf-section-desc">40 clean object sequences (1.000 visibility ratio, 18,361 mean 3D points) expanded into 280 labeled clips across 7 classes.</p>
          </div>
          <div class="pf-taxonomy-grid">{tax_html}</div>
          <p class="pf-subtable-title">Mean Extracted CV Features Across All 7 Defect Categories (280 Videos)</p>
          <div class="pf-table-scroll">
            <table class="pf-data-table">
              <thead><tr>
                <th>Defect Class</th><th>Duration (s)</th><th>Laplacian Var</th>
                <th>Blurry Ratio</th><th>Brightness</th><th>Step Diff</th>
                <th>Dark Ratio</th><th>Glare Ratio</th><th>ORB KPs</th><th>Loop Closure</th>
              </tr></thead>
              <tbody>{means_rows}</tbody>
            </table>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Model Benchmarks ──────────────────────────────────────────────────────
    if "bench_task" not in st.session_state:
        st.session_state["bench_task"] = "Defect_Diagnosis_7Class"

    bench_col1, bench_col2 = st.columns([3, 2])
    with bench_col1:
        st.markdown(
            """
            <div class="pf-card-title" style="margin-bottom:0.5rem;">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#9ca3af" stroke-width="2"><circle cx="12" cy="8" r="6"/><path d="M15.477 12.89L17 22l-5-3-5 3 1.523-9.11"/></svg>
              <span>Model Benchmarks &amp; Final Production Models</span>
            </div>
            <p class="pf-section-desc" style="margin-bottom:1rem;">Evaluated with 5-Fold StratifiedGroupKFold grouped by sequence_id (32 training objects / 8 unseen holdout objects = 56 test clips).</p>
            """,
            unsafe_allow_html=True,
        )
    with bench_col2:
        bench_opts = {"7-Class Defect Diagnosis": "Defect_Diagnosis_7Class",
                      "Binary Readiness Gate":     "Readiness_Binary"}
        bench_choice = st.radio("Benchmark Task", list(bench_opts.keys()),
                                 horizontal=True, label_visibility="collapsed",
                                 index=0 if st.session_state["bench_task"] == "Defect_Diagnosis_7Class" else 1)
        st.session_state["bench_task"] = bench_opts[bench_choice]

    bench_rows = BENCHMARK_RESULTS[st.session_state["bench_task"]]

    st.markdown(
        """
        <div class="pf-winner-cards-grid">
          <div class="pf-winner-card">
            <div class="pf-winner-head">
              <span class="pf-winner-tag">Final 7-Class Diagnosis Model</span>
            </div>
            <h3>SoftVoting_Ensemble</h3>
            <p class="pf-winner-sub">Tuned RandomForest (350 trees) + XGBoost (lr=0.03, depth=4) + LightGBM (leaves=23, depth=3)</p>
            <div class="pf-winner-metrics">
              <div><span>5-Fold CV Acc</span><strong>77.14%</strong></div>
              <div><span>CV Macro F1</span><strong>0.7655</strong></div>
              <div><span>Holdout Acc</span><strong>76.79%</strong></div>
              <div><span>Holdout F1</span><strong>0.7721</strong></div>
            </div>
          </div>
          <div class="pf-winner-card">
            <div class="pf-winner-head">
              <span class="pf-winner-tag green">Final Binary Readiness Gate</span>
            </div>
            <h3>XGBoost_Tuned</h3>
            <p class="pf-winner-sub">220 trees, max_depth=5, lr=0.08, subsample=0.75, reg_alpha=0.5, scale_pos_weight=6.0</p>
            <div class="pf-winner-metrics">
              <div><span>5-Fold CV Acc</span><strong>86.79%</strong></div>
              <div><span>CV Std</span><strong>+/-0.0143</strong></div>
              <div><span>CV Macro F1</span><strong>0.7302</strong></div>
              <div><span>Holdout Acc</span><strong>85.71%</strong></div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    table_rows = ""
    for row in bench_rows:
        win_cls = "winner-row" if row["winner"] else ""
        win_pill = '<span class="pf-win-pill">SELECTED</span>' if row["winner"] else ""
        table_rows += (
            f'<tr class="{win_cls}">'
            f'<td>{row["stage"]}</td>'
            f'<td class="pf-mono-strong">{row["model"]}{win_pill}</td>'
            f'<td>{row["cvAcc"]}</td><td>{row["cvStd"]}</td>'
            f'<td>{row["cvF1"]}</td><td>{row["holdAcc"]}</td><td>{row["holdF1"]}</td>'
            f'</tr>'
        )
    st.markdown(
        f"""
        <div class="pf-table-scroll" style="margin-bottom:1.25rem;">
          <table class="pf-data-table">
            <thead><tr>
              <th>Stage</th><th>Model</th><th>5-Fold CV Acc</th>
              <th>CV Std</th><th>CV Macro F1</th><th>Holdout Acc</th><th>Holdout Macro F1</th>
            </tr></thead>
            <tbody>{table_rows}</tbody>
          </table>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Holdout F1 + Top Features (two-col) ──────────────────────────────────
    holdout_bars = "".join(
        f'<div class="pf-prob-row pf-hiw-bar-row">'
        f'<span class="pf-prob-name">{item["cls"]}</span>'
        f'<div class="pf-prob-track">'
        f'<div class="pf-prob-fill {"clean" if item["f1"]>=0.8 else "alert" if item["f1"]<0.5 else ""}" style="width:{item["f1"]*100:.1f}%"></div>'
        f'</div>'
        f'<span class="pf-prob-val">{item["f1"]:.3f}</span>'
        f'</div>'
        for item in HOLDOUT_CLASS_REPORT
    )
    feat_bars = "".join(
        f'<div class="pf-prob-row pf-hiw-bar-row">'
        f'<span class="pf-prob-name pf-mono-strong" title="{feat["role"]}">{feat["name"]}</span>'
        f'<div class="pf-prob-track">'
        f'<div class="pf-prob-fill" style="width:{(feat["importance"]/0.16)*100:.1f}%"></div>'
        f'</div>'
        f'<span class="pf-prob-val">{feat["importance"]*100:.1f}%</span>'
        f'</div>'
        for feat in TOP_FEATURES
    )
    st.markdown(
        f"""
        <div class="pf-two-col">
          <div class="pf-card">
            <div class="pf-card-title">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#9ca3af" stroke-width="2"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/></svg>
              <span>Unseen Holdout Per-Class F1 (56 Test Videos)</span>
            </div>
            <p class="pf-section-desc">Strong precision/recall on glare, orbit cutoff, low light, blur, and flicker. Deterministic Geometric Guardrails backstop <code>low_overlap_fast_move</code> where 2.5 FPS downsampling compresses frame-drop gaps.</p>
            <div class="pf-prob-list" style="margin-top:1rem">{holdout_bars}</div>
          </div>
          <div class="pf-card">
            <div class="pf-card-title">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#9ca3af" stroke-width="2"><rect x="2" y="3" width="6" height="18" rx="1"/><rect x="9" y="8" width="6" height="13" rx="1"/><rect x="16" y="13" width="6" height="8" rx="1"/></svg>
              <span>Top 10 Consensus Feature Importances</span>
            </div>
            <p class="pf-section-desc">Normalized feature importances averaged across Tuned Random Forest, XGBoost, and LightGBM inside the Soft-Voting Ensemble.</p>
            <div class="pf-prob-list" style="margin-top:1rem">{feat_bars}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Final Output Gallery ──────────────────────────────────────────────────
    gallery_html = ""
    for item in FINAL_OUTPUT_GALLERY:
        gallery_html += f"""
        <div class="pf-output-card">
          <img src="{item["src"]}" alt="{item["file"]}"/>
          <div class="pf-output-body">
            <div class="pf-output-meta-row">
              <span class="pf-verdict-badge {item["tier"]}">Score {item["score"]}/100 &bull; {item["verdict"]}</span>
              <span class="pf-tag">{item["trigger"]}</span>
            </div>
            <h4 class="pf-mono-strong">{item["file"]}</h4>
            <p>{item["summary"]}</p>
          </div>
        </div>"""
    st.markdown(
        f"""
        <div class="pf-card">
          <div class="pf-section-head">
            <div class="pf-card-title">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#9ca3af" stroke-width="2"><rect x="2" y="3" width="20" height="14" rx="2"/><path d="M8 21h8M12 17v4"/></svg>
              <span>Completed Final Output — Live Dashboard Results</span>
            </div>
            <p class="pf-section-desc">End-to-end production runs captured on real uploaded test and smartphone videos, demonstrating both the Soft-Voting Ensemble and deterministic Geometric Guardrail in action.</p>
          </div>
          <div class="pf-output-gallery">{gallery_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("</div>", unsafe_allow_html=True)  # close pf-page-wrap

    # Footer
    st.markdown(
        """
        <div style="max-width:1380px;margin:0 auto;padding:0 2rem 2rem;text-align:center;
                    color:var(--text-muted);font-size:0.78rem;border-top:1px solid var(--border-subtle);
                    padding-top:1.5rem;">
          PreFlight-3D &bull; SoftVoting_Ensemble + XGBoost_Tuned &bull; UCO3D Dataset &bull;
          <a href="https://github.com/chandimacbbandara/preflight-3d" style="color:var(--accent-blue);text-decoration:none;">GitHub</a>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Router
# ══════════════════════════════════════════════════════════════════════════════

if st.session_state["page"] == "how-it-works":
    render_how_it_works()
else:
    render_analyzer()
