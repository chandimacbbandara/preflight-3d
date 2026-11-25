import { useEffect, useRef, useState } from "react";
import {
  Activity,
  AlertTriangle,
  ArrowRight,
  BookOpen,
  Box,
  CheckCircle2,
  Cpu,
  Database,
  Film,
  Focus,
  Gauge,
  Loader2,
  Orbit,
  Sliders,
  Sparkles,
  Sun,
  UploadCloud,
  XCircle,
} from "lucide-react";
import "./App.css";
import HowItWorks from "./pages/HowItWorks";

const API_BASE = "http://localhost:8000/api";

const METRIC_LABELS = [
  { key: "duration_sec", label: "Duration (sec)", unit: "s", cat: "Orbit" },
  { key: "sampled_frames", label: "Sampled Frames", unit: "", cat: "Orbit" },
  { key: "laplacian_var_mean", label: "Sharpness (Laplacian)", unit: "", cat: "Focus" },
  { key: "blurry_frame_ratio", label: "Blurry Frame Ratio", unit: "%", pct: true, cat: "Focus" },
  { key: "edge_density_mean", label: "Edge Density", unit: "%", pct: true, cat: "Focus" },
  { key: "brightness_mean", label: "Mean Brightness", unit: "/255", cat: "Light" },
  { key: "brightness_step_diff_mean", label: "Exposure Jump", unit: "", cat: "Light" },
  { key: "dark_pixel_ratio", label: "Underexposed Pixels", unit: "%", pct: true, cat: "Light" },
  { key: "specular_glare_ratio", label: "Specular Glare Ratio", unit: "%", pct: true, cat: "Light" },
  { key: "orb_keypoints_mean", label: "Avg ORB Keypoints", unit: "", cat: "Track" },
  { key: "optical_flow_mean", label: "Mean Optical Flow", unit: " px", cat: "Track" },
  { key: "optical_flow_max", label: "Peak Optical Flow", unit: " px", cat: "Track" },
  { key: "pair_match_ratio_mean", label: "Feature Match Ratio", unit: "%", pct: true, cat: "Track" },
  { key: "pair_inlier_ratio_mean", label: "RANSAC Inlier Ratio", unit: "%", pct: true, cat: "Track" },
  { key: "cumulative_pan_px", label: "Cumulative Orbit Pan", unit: " px", cat: "Orbit" },
  { key: "loop_closure_inlier_ratio", label: "Loop Closure Inliers", unit: "%", pct: true, cat: "Orbit" },
];

function computePillarScores(m) {
  if (!m) return [];
  const sharpness = Math.round(
    Math.min(1, (m.laplacian_var_mean || 0) / 150) *
      (1 - Math.min(1, m.blurry_frame_ratio || 0)) *
      100
  );

  const tracking = Math.round(
    (0.65 * Math.min(1, (m.pair_inlier_ratio_mean || 0) / 0.65) +
      0.35 * Math.min(1, (m.orb_keypoints_mean || 0) / 250)) *
      100
  );

  const b = m.brightness_mean || 0;
  let brightComfort = 0;
  if (b >= 75 && b <= 190) brightComfort = 1;
  else if (b < 75) brightComfort = Math.max(0, b / 75);
  else brightComfort = Math.max(0, (255 - b) / 65);

  const glareComfort = Math.max(0, 1 - (m.specular_glare_ratio || 0) * 3);
  const flickerComfort = Math.max(
    0,
    1 - Math.max(0, ((m.brightness_step_diff_mean || 0) - 8) / 28)
  );
  const lighting = Math.round(brightComfort * glareComfort * flickerComfort * 100);

  const durationFactor = Math.min(1, (m.duration_sec || 0) / 12);
  const coverageFactor = Math.max(
    Math.min(1, (m.loop_closure_inlier_ratio || 0) / 0.45),
    Math.min(1, (m.cumulative_pan_px || 0) / 30)
  );
  let orbitRaw = 0.5 * durationFactor + 0.5 * coverageFactor;
  if ((m.loop_closure_inlier_ratio || 0) >= 0.5 && (m.optical_flow_mean || 0) >= 2) {
    orbitRaw = Math.max(orbitRaw, coverageFactor * 0.95);
  }
  const orbit = Math.round(orbitRaw * 100);

  return [
    { key: "Focus", label: "Focus & Sharpness", score: sharpness, weight: "25%" },
    { key: "Track", label: "Tracking & Overlap", score: tracking, weight: "30%" },
    { key: "Light", label: "Lighting & Glare", score: lighting, weight: "25%" },
    { key: "Orbit", label: "360° Orbit Coverage", score: orbit, weight: "20%" },
  ];
}

function RadialScoreRing({ score, tier }) {
  const radius = 48;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - (Math.max(0, Math.min(100, score)) / 100) * circumference;

  return (
    <div className={`radial-wrap ${tier}`}>
      <svg className="radial-svg" viewBox="0 0 120 120">
        <circle className="radial-track" cx="60" cy="60" r={radius} />
        <circle
          className={`radial-progress ${tier}`}
          cx="60"
          cy="60"
          r={radius}
          strokeDasharray={circumference}
          strokeDashoffset={offset}
        />
      </svg>
      <div className="radial-center">
        <span className="score-num">{score}</span>
        <span className="score-label">Readiness</span>
      </div>
    </div>
  );
}

export default function App() {
  const [page, setPage] = useState("analyzer");
  const [health, setHealth] = useState(null);
  const [mode, setMode] = useState("upload");
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [dragging, setDragging] = useState(false);

  const [samples, setSamples] = useState([]);
  const [selectedSampleId, setSelectedSampleId] = useState("");

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);

  const fileInputRef = useRef(null);

  useEffect(() => {
    fetch(`${API_BASE}/health`)
      .then((res) => (res.ok ? res.json() : null))
      .then((data) => setHealth(data))
      .catch(() => setHealth(null));

    fetch(`${API_BASE}/samples`)
      .then((res) => (res.ok ? res.json() : []))
      .then((data) => {
        setSamples(data);
        if (data.length > 0) setSelectedSampleId(data[0].sample_id);
      })
      .catch(() => setSamples([]));
  }, []);

  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  const handleFileSelect = (file) => {
    if (!file) return;
    setError("");
    setSelectedFile(file);
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setPreviewUrl(URL.createObjectURL(file));
  };

  const handleAnalyzeUpload = async () => {
    if (!selectedFile) return;
    setLoading(true);
    setError("");
    try {
      const formData = new FormData();
      formData.append("file", selectedFile);
      const res = await fetch(`${API_BASE}/analyze`, {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Failed to analyze video.");
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const runSampleById = async (sampleId) => {
    if (!sampleId) return;
    setSelectedSampleId(sampleId);
    setMode("sample");
    setLoading(true);
    setError("");
    try {
      const res = await fetch(
        `${API_BASE}/analyze-sample/${encodeURIComponent(sampleId)}`,
        { method: "POST" }
      );
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Failed to analyze sample.");
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const getVerdictTier = (verdict) => {
    if (verdict === "READY_FOR_RECONSTRUCTION") return "good";
    if (verdict === "USABLE_WITH_WARNINGS") return "warn";
    return "bad";
  };

  const formatDefectName = (code) =>
    code === "none" ? "Clean Capture (No Defect)" : code.replaceAll("_", " ");

  // Quick-test presets (1 per key defect category from loaded /api/samples)
  const quickPresets = ["none", "motion_blur", "harsh_glare", "incomplete_orbit"]
    .map((dtype) => samples.find((s) => s.defect_type === dtype))
    .filter(Boolean);

  const pillars = computePillarScores(result?.extracted_metrics);

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand">
          <div className="brand-icon">
            <Box size={22} />
          </div>
          <div>
            <h1>PreFlight-3D</h1>
            <span>Photogrammetry &amp; 3DGS Video Quality Gate</span>
          </div>
        </div>

        <nav className="top-nav" aria-label="Primary Navigation">
          <button
            type="button"
            className={`nav-pill ${page === "analyzer" ? "active" : ""}`}
            onClick={() => setPage("analyzer")}
          >
            <Gauge size={15} />
            <span>Pre-Flight Analyzer</span>
          </button>
          <button
            type="button"
            className={`nav-pill ${page === "how-it-works" ? "active" : ""}`}
            onClick={() => setPage("how-it-works")}
          >
            <BookOpen size={15} />
            <span>How It Works (A–Z)</span>
          </button>
        </nav>

        <div className="health-pill">
          <span className={`dot ${health ? "online" : "offline"}`} />
          {health ? (
            <span>
              API Online &bull; {health.defect_model} + {health.binary_model} (
              {health.num_features} CV features)
            </span>
          ) : (
            <span>API Offline (Start FastAPI on :8000)</span>
          )}
        </div>
      </header>

      {page === "how-it-works" ? (
        <main className="page-wrap">
          <HowItWorks onGoToAnalyzer={() => setPage("analyzer")} />
        </main>
      ) : (
        <main className="main-grid">
          <aside className="sidebar-stack">
            <section className="card">
              <div className="card-title">
                <Film size={18} />
                <span>Capture Input</span>
              </div>

              <div className="tabs">
                <button
                  type="button"
                  className={`tab-btn ${mode === "upload" ? "active" : ""}`}
                  onClick={() => {
                    setMode("upload");
                    setError("");
                  }}
                >
                  <UploadCloud size={15} />
                  <span>Upload Video</span>
                </button>
                <button
                  type="button"
                  className={`tab-btn ${mode === "sample" ? "active" : ""}`}
                  onClick={() => {
                    setMode("sample");
                    setError("");
                  }}
                >
                  <Database size={15} />
                  <span>Test Dataset</span>
                </button>
              </div>

              {mode === "upload" ? (
                <>
                  <div
                    className={`dropzone ${dragging ? "dragging" : ""}`}
                    onClick={() => fileInputRef.current?.click()}
                    onDragOver={(e) => {
                      e.preventDefault();
                      setDragging(true);
                    }}
                    onDragLeave={() => setDragging(false)}
                    onDrop={(e) => {
                      e.preventDefault();
                      setDragging(false);
                      handleFileSelect(e.dataTransfer.files?.[0]);
                    }}
                  >
                    <UploadCloud
                      size={32}
                      color="#3b82f6"
                      style={{ margin: "0 auto" }}
                    />
                    <p>Click to select or drag &amp; drop a capture video</p>
                    <small>Supports MP4, MOV, MKV, AVI, WEBM</small>
                    <input
                      ref={fileInputRef}
                      type="file"
                      accept=".mp4,.mov,.mkv,.avi,.webm,video/*"
                      hidden
                      onChange={(e) => handleFileSelect(e.target.files?.[0])}
                    />
                  </div>

                  {selectedFile && previewUrl && (
                    <div className="video-preview">
                      <video src={previewUrl} controls muted />
                      <div className="file-meta">
                        <span>{selectedFile.name}</span>
                        <span>
                          {(selectedFile.size / (1024 * 1024)).toFixed(2)} MB
                        </span>
                      </div>
                    </div>
                  )}

                  <button
                    type="button"
                    className="primary-btn"
                    disabled={!selectedFile || loading}
                    onClick={handleAnalyzeUpload}
                  >
                    {loading ? (
                      <Loader2 size={18} className="spin" />
                    ) : (
                      <Activity size={18} />
                    )}
                    <span>
                      {loading
                        ? "Analyzing Frames & Geometry..."
                        : "Run Pre-Flight Check"}
                    </span>
                  </button>
                </>
              ) : (
                <>
                  <label
                    htmlFor="sample-picker"
                    style={{
                      display: "block",
                      fontSize: "0.82rem",
                      color: "var(--text-secondary)",
                      marginBottom: "0.45rem",
                    }}
                  >
                    Select a uCO3D Benchmark Video (Clean or Synthetic Defect):
                  </label>
                  <select
                    id="sample-picker"
                    className="sample-select"
                    value={selectedSampleId}
                    onChange={(e) => setSelectedSampleId(e.target.value)}
                  >
                    {samples.map((s) => (
                      <option key={s.sample_id} value={s.sample_id}>
                        [{s.readiness_label}] {s.category} — {s.defect_type} (
                        {s.sample_id})
                      </option>
                    ))}
                  </select>

                  <div className="sample-hint">
                    Runs live 19-feature CV extraction and Soft-Voting Ensemble
                    + Physics Gate inference directly on your local uCO3D
                    dataset.
                  </div>

                  <button
                    type="button"
                    className="primary-btn"
                    disabled={!selectedSampleId || loading}
                    onClick={() => runSampleById(selectedSampleId)}
                  >
                    {loading ? (
                      <Loader2 size={18} className="spin" />
                    ) : (
                      <Activity size={18} />
                    )}
                    <span>
                      {loading
                        ? "Running Live Inference..."
                        : "Analyze Benchmark Sample"}
                    </span>
                  </button>
                </>
              )}

              {quickPresets.length > 0 && (
                <div className="quick-presets">
                  <span className="qp-title">Instant 1-Click Benchmark Test:</span>
                  <div className="qp-grid">
                    {quickPresets.map((s) => (
                      <button
                        key={s.sample_id}
                        type="button"
                        disabled={loading}
                        className={`qp-chip ${
                          selectedSampleId === s.sample_id && mode === "sample"
                            ? "active"
                            : ""
                        }`}
                        onClick={() => runSampleById(s.sample_id)}
                      >
                        <span
                          className={`qp-dot ${
                            s.defect_type === "none" ? "good" : "bad"
                          }`}
                        />
                        <span>
                          {s.defect_type === "none"
                            ? "Clean Sample"
                            : s.defect_type.replaceAll("_", " ")}
                        </span>
                      </button>
                    ))}
                  </div>
                </div>
              )}

              {error && <div className="error-box">{error}</div>}
            </section>

            <section className="card mini-guide-card">
              <div className="card-title">
                <Cpu size={17} />
                <span>4 Inspection Pillars</span>
              </div>
              <div className="pillar-legend">
                <div>
                  <Focus size={15} color="#60a5fa" />
                  <span>
                    <strong>Sharpness:</strong> Laplacian &amp; Canny edges
                  </span>
                </div>
                <div>
                  <Sun size={15} color="#fbbf24" />
                  <span>
                    <strong>Exposure:</strong> Brightness, flicker &amp; glare
                  </span>
                </div>
                <div>
                  <Sparkles size={15} color="#34d399" />
                  <span>
                    <strong>Texture:</strong> ORB keypoints &amp; RANSAC matches
                  </span>
                </div>
                <div>
                  <Orbit size={15} color="#a78bfa" />
                  <span>
                    <strong>360° Orbit:</strong> Optical flow &amp; loop closure
                  </span>
                </div>
              </div>
              <button
                type="button"
                className="ghost-link-btn"
                onClick={() => setPage("how-it-works")}
              >
                <span>Read Full A–Z Architecture &amp; Benchmarks</span>
                <ArrowRight size={15} />
              </button>
            </section>
          </aside>

          <section className="results-stack">
            {!result ? (
              <div className="card empty-hero">
                <div className="empty-badge">
                  <Sparkles size={14} />
                  <span>Ready for Pre-Flight Inspection</span>
                </div>
                <h2>Catch 3D Reconstruction Failures Before Running COLMAP or 3DGS</h2>
                <p>
                  Upload an object-centric orbit video or click any benchmark
                  preset on the left to inspect motion blur, lighting stability,
                  specular glare, and 360° loop closure in under a second.
                </p>

                <div className="empty-steps">
                  <div className="es-step">
                    <span>01</span>
                    <strong>Sample @ 2.5 FPS</strong>
                    <small>Elliptical subject ROI masking at 480px</small>
                  </div>
                  <div className="es-step">
                    <span>02</span>
                    <strong>19 CV Features</strong>
                    <small>Sharpness, glare, ORB, Farneback &amp; RANSAC</small>
                  </div>
                  <div className="es-step">
                    <span>03</span>
                    <strong>Ensemble + Physics</strong>
                    <small>SoftVoting + XGBoost_Tuned + guardrails</small>
                  </div>
                  <div className="es-step">
                    <span>04</span>
                    <strong>0–100 Readiness</strong>
                    <small>Actionable fixes before GPU compute</small>
                  </div>
                </div>

                <div className="empty-actions">
                  <button
                    type="button"
                    className="secondary-btn"
                    onClick={() => setPage("how-it-works")}
                  >
                    <BookOpen size={16} />
                    <span>Explore How It Works (A–Z) &amp; Model Results</span>
                    <ArrowRight size={16} />
                  </button>
                </div>
              </div>
            ) : (
              <>
                <div className="card verdict-card">
                  <RadialScoreRing
                    score={result.readiness_score}
                    tier={getVerdictTier(result.verdict)}
                  />

                  <div className="verdict-details">
                    <div
                      className={`verdict-badge ${getVerdictTier(
                        result.verdict
                      )}`}
                    >
                      {result.verdict === "READY_FOR_RECONSTRUCTION" && (
                        <CheckCircle2 size={15} />
                      )}
                      {result.verdict === "USABLE_WITH_WARNINGS" && (
                        <AlertTriangle size={15} />
                      )}
                      {result.verdict === "RECAPTURE_RECOMMENDED" && (
                        <XCircle size={15} />
                      )}
                      <span>{result.verdict.replaceAll("_", " ")}</span>
                    </div>
                    <h2>{result.video_file}</h2>
                    <p>{result.summary}</p>
                  </div>
                </div>

                {/* 4-Pillar CV Health Breakdown */}
                <div className="card">
                  <div className="card-title">
                    <Gauge size={18} />
                    <span>4-Pillar Capture Health Breakdown</span>
                  </div>
                  <div className="pillars-grid">
                    {pillars.map((p) => {
                      const tier =
                        p.score >= 70 ? "good" : p.score >= 45 ? "warn" : "bad";
                      return (
                        <div key={p.key} className="pillar-card">
                          <div className="pillar-top">
                            <span>{p.label}</span>
                            <small>Wt {p.weight}</small>
                          </div>
                          <div className="pillar-val-row">
                            <strong className={`pillar-num ${tier}`}>
                              {p.score}%
                            </strong>
                          </div>
                          <div className="prob-track">
                            <div
                              className={`prob-fill ${
                                tier === "good"
                                  ? "clean"
                                  : tier === "bad"
                                  ? "alert"
                                  : "warn-fill"
                              }`}
                              style={{ width: `${p.score}%` }}
                            />
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>

                <div className="card">
                  <div className="card-title">
                    <AlertTriangle size={18} />
                    <span>Actionable Capture Recommendations</span>
                  </div>

                  {result.recommendations.length === 0 ? (
                    <div className="pass-banner">
                      <CheckCircle2 size={20} />
                      <span>
                        All quality gates passed. Frame sharpness, uniform
                        lighting, and 360° orbit overlap are strong enough for
                        3D reconstruction.
                      </span>
                    </div>
                  ) : (
                    <div className="rec-list">
                      {result.recommendations.map((rec) => (
                        <div
                          key={rec.issue_code}
                          className={`rec-item ${rec.severity}`}
                        >
                          <div className="rec-header">
                            <h4>{rec.title}</h4>
                            <div className="rec-tags">
                              <span className="tag">{rec.category}</span>
                              <span className="tag">
                                {rec.detected_by} (
                                {Math.round(rec.confidence * 100)}%)
                              </span>
                            </div>
                          </div>
                          <p className="rec-impact">{rec.impact}</p>
                          <ol className="rec-steps">
                            {rec.action_steps.map((step, idx) => (
                              <li key={idx}>{step}</li>
                            ))}
                          </ol>
                        </div>
                      ))}
                    </div>
                  )}
                </div>

                <div className="card">
                  <div className="card-title">
                    <Activity size={18} />
                    <span>Soft-Voting Ensemble Diagnosis Probabilities</span>
                  </div>

                  <div className="prob-list">
                    {Object.entries(result.defect_probabilities)
                      .sort((a, b) => b[1] - a[1])
                      .map(([cls, prob]) => {
                        const pct = Math.round(prob * 1000) / 10;
                        const fillClass =
                          cls === "none"
                            ? "clean"
                            : prob >= 0.25
                            ? "alert"
                            : "";
                        return (
                          <div key={cls} className="prob-row">
                            <span className="prob-name">
                              {formatDefectName(cls)}
                            </span>
                            <div className="prob-track">
                              <div
                                className={`prob-fill ${fillClass}`}
                                style={{ width: `${Math.min(pct, 100)}%` }}
                              />
                            </div>
                            <span className="prob-val">{pct.toFixed(1)}%</span>
                          </div>
                        );
                      })}
                  </div>
                </div>

                <div className="card">
                  <div className="card-title">
                    <Sliders size={18} />
                    <span>
                      Extracted Computer Vision &amp; Trajectory Telemetry
                    </span>
                  </div>

                  <div className="metrics-grid">
                    {METRIC_LABELS.map(({ key, label, unit, pct, cat }) => {
                      const rawVal = result.extracted_metrics?.[key];
                      const displayVal =
                        rawVal === undefined || rawVal === null
                          ? "—"
                          : pct
                          ? `${(rawVal * 100).toFixed(1)}${unit}`
                          : `${rawVal}${unit}`;
                      return (
                        <div key={key} className="metric-tile">
                          <div className="metric-top">
                            <span>{label}</span>
                            <small className="metric-cat">{cat}</small>
                          </div>
                          <strong>{displayVal}</strong>
                        </div>
                      );
                    })}
                  </div>
                </div>
              </>
            )}
          </section>
        </main>
      )}
    </div>
  );
}
