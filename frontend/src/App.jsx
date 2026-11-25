import { useEffect, useRef, useState } from "react";
import {
  Activity,
  AlertTriangle,
  Box,
  CheckCircle2,
  Database,
  Film,
  Loader2,
  Sliders,
  UploadCloud,
  XCircle,
} from "lucide-react";
import "./App.css";

const API_BASE = "http://localhost:8000/api";

const METRIC_LABELS = [
  { key: "duration_sec", label: "Duration (sec)", unit: "s" },
  { key: "sampled_frames", label: "Sampled Frames", unit: "" },
  { key: "laplacian_var_mean", label: "Sharpness (Laplacian)", unit: "" },
  { key: "blurry_frame_ratio", label: "Blurry Frame Ratio", unit: "%", pct: true },
  { key: "edge_density_mean", label: "Edge Density", unit: "%", pct: true },
  { key: "brightness_mean", label: "Mean Brightness", unit: "/255" },
  { key: "brightness_step_diff_mean", label: "Exposure Jump", unit: "" },
  { key: "dark_pixel_ratio", label: "Underexposed Pixels", unit: "%", pct: true },
  { key: "specular_glare_ratio", label: "Specular Glare Ratio", unit: "%", pct: true },
  { key: "orb_keypoints_mean", label: "Avg ORB Keypoints", unit: "" },
  { key: "optical_flow_mean", label: "Mean Optical Flow", unit: " px" },
  { key: "optical_flow_max", label: "Peak Optical Flow", unit: " px" },
  { key: "pair_match_ratio_mean", label: "Feature Match Ratio", unit: "%", pct: true },
  { key: "pair_inlier_ratio_mean", label: "RANSAC Inlier Ratio", unit: "%", pct: true },
  { key: "cumulative_pan_px", label: "Cumulative Orbit Pan", unit: " px" },
  { key: "loop_closure_inlier_ratio", label: "Loop Closure Inliers", unit: "%", pct: true },
];

export default function App() {
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

  const handleAnalyzeSample = async () => {
    if (!selectedSampleId) return;
    setLoading(true);
    setError("");
    try {
      const res = await fetch(`${API_BASE}/analyze-sample/${encodeURIComponent(selectedSampleId)}`, {
        method: "POST",
      });
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

        <div className="health-pill">
          <span className={`dot ${health ? "online" : "offline"}`} />
          {health ? (
            <span>
              API Online &bull; {health.defect_model} + {health.binary_model} ({health.num_features} CV features)
            </span>
          ) : (
            <span>API Offline (Start FastAPI on :8000)</span>
          )}
        </div>
      </header>

      <main className="main-grid">
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
                <UploadCloud size={32} color="#3b82f6" style={{ margin: "0 auto" }} />
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
                    <span>{(selectedFile.size / (1024 * 1024)).toFixed(2)} MB</span>
                  </div>
                </div>
              )}

              <button
                type="button"
                className="primary-btn"
                disabled={!selectedFile || loading}
                onClick={handleAnalyzeUpload}
              >
                {loading ? <Loader2 size={18} className="spin" /> : <Activity size={18} />}
                <span>{loading ? "Analyzing Frames & Geometry..." : "Run Pre-Flight Check"}</span>
              </button>
            </>
          ) : (
            <>
              <label
                htmlFor="sample-picker"
                style={{ display: "block", fontSize: "0.82rem", color: "var(--text-secondary)", marginBottom: "0.45rem" }}
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
                    [{s.readiness_label}] {s.category} — {s.defect_type} ({s.sample_id})
                  </option>
                ))}
              </select>

              <div className="sample-hint">
                Runs live CV feature extraction and ensemble inference directly on a curated clean or degraded sequence from your local dataset.
              </div>

              <button
                type="button"
                className="primary-btn"
                disabled={!selectedSampleId || loading}
                onClick={handleAnalyzeSample}
              >
                {loading ? <Loader2 size={18} className="spin" /> : <Activity size={18} />}
                <span>{loading ? "Running Live Inference..." : "Analyze Benchmark Sample"}</span>
              </button>
            </>
          )}

          {error && <div className="error-box">{error}</div>}
        </section>

        <section className="results-stack">
          {!result ? (
            <div className="card empty-state">
              <Box size={44} color="#3b82f6" style={{ margin: "0 auto" }} />
              <h3>Ready for Pre-Flight Inspection</h3>
              <p>
                Upload an object-centric orbit video or select a benchmark sample to inspect blur, exposure stability, specular glare, and 360° loop closure before running COLMAP or 3D Gaussian Splatting.
              </p>
            </div>
          ) : (
            <>
              <div className="card verdict-card">
                <div className={`score-ring ${getVerdictTier(result.verdict)}`}>
                  <span className="score-num">{result.readiness_score}</span>
                  <span className="score-label">Readiness</span>
                </div>

                <div className="verdict-details">
                  <div className={`verdict-badge ${getVerdictTier(result.verdict)}`}>
                    {result.verdict === "READY_FOR_RECONSTRUCTION" && <CheckCircle2 size={15} />}
                    {result.verdict === "USABLE_WITH_WARNINGS" && <AlertTriangle size={15} />}
                    {result.verdict === "RECAPTURE_RECOMMENDED" && <XCircle size={15} />}
                    <span>{result.verdict.replaceAll("_", " ")}</span>
                  </div>
                  <h2>{result.video_file}</h2>
                  <p>{result.summary}</p>
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
                      All quality gates passed. Frame sharpness, uniform lighting, and 360° orbit overlap are strong enough for 3D reconstruction.
                    </span>
                  </div>
                ) : (
                  <div className="rec-list">
                    {result.recommendations.map((rec) => (
                      <div key={rec.issue_code} className={`rec-item ${rec.severity}`}>
                        <div className="rec-header">
                          <h4>{rec.title}</h4>
                          <div className="rec-tags">
                            <span className="tag">{rec.category}</span>
                            <span className="tag">
                              {rec.detected_by} ({Math.round(rec.confidence * 100)}%)
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
                        cls === "none" ? "clean" : prob >= 0.25 ? "alert" : "";
                      return (
                        <div key={cls} className="prob-row">
                          <span className="prob-name">{formatDefectName(cls)}</span>
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
                  <span>Extracted Computer Vision &amp; Trajectory Telemetry</span>
                </div>

                <div className="metrics-grid">
                  {METRIC_LABELS.map(({ key, label, unit, pct }) => {
                    const rawVal = result.extracted_metrics?.[key];
                    const displayVal =
                      rawVal === undefined || rawVal === null
                        ? "—"
                        : pct
                        ? `${(rawVal * 100).toFixed(1)}${unit}`
                        : `${rawVal}${unit}`;
                    return (
                      <div key={key} className="metric-tile">
                        <span>{label}</span>
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
    </div>
  );
}
