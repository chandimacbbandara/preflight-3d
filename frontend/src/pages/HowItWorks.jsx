import { useState } from "react";
import {
  AlertOctagon,
  ArrowRight,
  Award,
  BarChart3,
  CheckCircle2,
  Cpu,
  Database,
  Eye,
  GitBranch,
  Layers,
  Maximize2,
  MonitorCheck,
  ShieldCheck,
  Sparkles,
  X,
  Zap,
} from "lucide-react";
import {
  BENCHMARK_RESULTS,
  FEATURE_CLASS_MEANS,
  FINAL_OUTPUT_GALLERY,
  HERO_STATS,
  HOLDOUT_CLASS_REPORT,
  PIPELINE_STAGES,
  PROBLEMS_LIST,
  SOLUTIONS_LIST,
  TAXONOMY_GROUPS,
  TOP_FEATURES,
} from "../data/projectData";

export default function HowItWorks({ onGoToAnalyzer }) {
  const [activeStageId, setActiveStageId] = useState("A");
  const [benchTask, setBenchTask] = useState("Defect_Diagnosis_7Class");
  const [lightboxItem, setLightboxItem] = useState(null);

  const activeStage =
    PIPELINE_STAGES.find((s) => s.id === activeStageId) || PIPELINE_STAGES[0];
  const benchRows = BENCHMARK_RESULTS[benchTask] || [];

  return (
    <div className="hiw-container">
      {/* Hero Banner */}
      <section className="hiw-hero card">
        <div className="hiw-hero-top">
          <div className="hiw-badge">
            <Sparkles size={14} />
            <span>A-to-Z Engineering Architecture &amp; Benchmark Report</span>
          </div>
          <h1>
            Pre-Flight Quality Gate for{" "}
            <span className="gradient-text">3D Reconstruction</span>
          </h1>
          <p className="hiw-hero-sub">
            PreFlight-3D evaluates smartphone and turntable object-capture videos
            in ~0.5 seconds on CPU — catching motion blur, underexposure,
            specular glare, exposure flicker, sparse overlap, and incomplete
            360° orbits before you commit heavy compute to COLMAP, NeRF, or 3D
            Gaussian Splatting.
          </p>

          <div className="hiw-hero-actions">
            <button
              type="button"
              className="primary-btn inline-btn"
              onClick={onGoToAnalyzer}
            >
              <Zap size={16} />
              <span>Try Live Analyzer</span>
              <ArrowRight size={16} />
            </button>
          </div>
        </div>

        <div className="hiw-kpi-strip">
          {HERO_STATS.map((stat) => (
            <div key={stat.label} className="kpi-tile">
              <span className="kpi-label">{stat.label}</span>
              <strong className="kpi-value">{stat.value}</strong>
              <small className="kpi-sub">{stat.sub}</small>
            </div>
          ))}
        </div>
      </section>

      {/* Problem vs Solution */}
      <section className="problem-solution-grid">
        <div className="card ps-card problem">
          <div className="ps-header">
            <div className="ps-icon bad">
              <AlertOctagon size={20} />
            </div>
            <div>
              <span className="ps-eyebrow">The Problem</span>
              <h2>Why 3D Reconstructions Fail Silently</h2>
            </div>
          </div>
          <div className="ps-list">
            {PROBLEMS_LIST.map((item) => (
              <div key={item.title} className="ps-item">
                <h4>{item.title}</h4>
                <p>{item.detail}</p>
              </div>
            ))}
          </div>
        </div>

        <div className="card ps-card solution">
          <div className="ps-header">
            <div className="ps-icon good">
              <ShieldCheck size={20} />
            </div>
            <div>
              <span className="ps-eyebrow">Our Solution</span>
              <h2>Hybrid CV + Ensemble + Physics Gate</h2>
            </div>
          </div>
          <div className="ps-list">
            {SOLUTIONS_LIST.map((item) => (
              <div key={item.title} className="ps-item">
                <h4>{item.title}</h4>
                <p>{item.detail}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Visual GIF Showcases */}
      <section className="card">
        <div className="section-head">
          <div>
            <div className="card-title">
              <Eye size={18} />
              <span>Visual Dataset &amp; Failure Simulation Showcase</span>
            </div>
            <p className="section-desc">
              Real 360° captures curated from Meta&apos;s uCO3D dataset alongside
              live PreFlight-3D inference scores across all 7 diagnostic classes.
            </p>
          </div>
        </div>

        <div className="gif-grid">
          <div className="gif-card">
            <img
              src="/selected_samples_showcase.gif"
              alt="Selected uCO3D Benchmark Captures Showcase"
            />
            <div className="gif-caption">
              <strong>Selected uCO3D Object Taxonomy (6 Clean Samples)</strong>
              <span>
                Covers Glossy (Vase), Metallic (Fire Extinguisher), Complex
                (Helicopter), Machinery (Motor), Matte (Handbag), and Rigid
                (Battery) captures — scoring 95–98/100 READY.
              </span>
            </div>
          </div>

          <div className="gif-card">
            <img
              src="/defect_modes_comparison.gif"
              alt="Clean vs 6 Synthetic Capture Defects Comparison"
            />
            <div className="gif-caption">
              <strong>
                Clean vs. 6 Synthetic Defects (Vase 47433-58842-22217)
              </strong>
              <span>
                Live hybrid inference separates clean capture (98/100 PASS) from
                Motion Blur, Low Light, Harsh Glare, Exposure Flicker, Low
                Overlap (23/100), and Incomplete Orbit (stopped at 35%).
              </span>
            </div>
          </div>
        </div>
      </section>

      {/* Interactive A to Z Pipeline */}
      <section className="card">
        <div className="section-head">
          <div>
            <div className="card-title">
              <GitBranch size={18} />
              <span>A-to-Z End-to-End Pipeline (Stages A → I)</span>
            </div>
            <p className="section-desc">
              Click any stage below to inspect the exact scripts, design
              decisions, and verified outputs built across the project.
            </p>
          </div>
        </div>

        <div className="stage-stepper">
          {PIPELINE_STAGES.map((stage) => (
            <button
              key={stage.id}
              type="button"
              className={`stage-pill ${
                activeStageId === stage.id ? "active" : ""
              }`}
              onClick={() => setActiveStageId(stage.id)}
            >
              <span className="stage-letter">{stage.id}</span>
              <span className="stage-mini-title">{stage.tag}</span>
            </button>
          ))}
        </div>

        <div className="stage-detail-card">
          <div className="stage-detail-head">
            <div className="stage-badge-lg">Stage {activeStage.id}</div>
            <div>
              <h3>{activeStage.title}</h3>
              <code className="script-chip">{activeStage.script}</code>
            </div>
          </div>
          <p className="stage-detail-summary">{activeStage.summary}</p>
          <div className="stage-highlights">
            {activeStage.highlights.map((hl) => (
              <span key={hl} className="hl-chip">
                <CheckCircle2 size={13} />
                <span>{hl}</span>
              </span>
            ))}
          </div>
        </div>

        <div className="timeline-all-grid">
          {PIPELINE_STAGES.map((stage) => (
            <div
              key={stage.id}
              className={`timeline-mini-card ${
                activeStageId === stage.id ? "selected" : ""
              }`}
              onClick={() => setActiveStageId(stage.id)}
            >
              <div className="tm-top">
                <span className="tm-id">Stage {stage.id}</span>
                <span className="tag">{stage.tag}</span>
              </div>
              <h4>{stage.title}</h4>
              <p>{stage.summary}</p>
            </div>
          ))}
        </div>
      </section>

      {/* 40-Category Taxonomy & Feature Means */}
      <section className="card">
        <div className="section-head">
          <div>
            <div className="card-title">
              <Database size={18} />
              <span>
                uCO3D 40-Category Curation &amp; Extracted Feature Signatures
              </span>
            </div>
            <p className="section-desc">
              40 clean object sequences (1.000 visibility ratio, 18,361 mean 3D
              points) expanded into 280 labeled clips across 7 classes.
            </p>
          </div>
        </div>

        <div className="taxonomy-grid">
          {TAXONOMY_GROUPS.map((g) => (
            <div key={g.group} className="tax-card">
              <h4>{g.group}</h4>
              <p className="tax-items">{g.items}</p>
              <span className="tax-challenge">Edge case: {g.challenge}</span>
            </div>
          ))}
        </div>

        <h4 className="subtable-title">
          Mean Extracted CV Features Across All 7 Defect Categories (280 Videos)
        </h4>
        <div className="table-scroll">
          <table className="data-table">
            <thead>
              <tr>
                <th>Defect Class</th>
                <th>Duration (s)</th>
                <th>Laplacian Var</th>
                <th>Blurry Ratio</th>
                <th>Brightness</th>
                <th>Step Diff</th>
                <th>Dark Ratio</th>
                <th>Glare Ratio</th>
                <th>ORB KPs</th>
                <th>Loop Closure</th>
              </tr>
            </thead>
            <tbody>
              {FEATURE_CLASS_MEANS.map((row) => (
                <tr key={row.defect}>
                  <td className="mono-strong">{row.defect}</td>
                  <td>{row.duration}</td>
                  <td>{row.laplacian}</td>
                  <td>{row.blurry}</td>
                  <td>{row.brightness}</td>
                  <td>{row.stepDiff}</td>
                  <td>{row.dark}</td>
                  <td>{row.glare}</td>
                  <td>{row.orb}</td>
                  <td>{row.loop}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      {/* Model Benchmark & Final Winning Models */}
      <section className="card">
        <div className="section-head-row">
          <div>
            <div className="card-title">
              <Award size={18} />
              <span>Model Benchmarks &amp; Final Production Models</span>
            </div>
            <p className="section-desc">
              Evaluated with 5-Fold StratifiedGroupKFold grouped by sequence_id
              (32 training objects / 8 unseen holdout objects = 56 test clips).
            </p>
          </div>

          <div className="tabs compact-tabs">
            <button
              type="button"
              className={`tab-btn ${
                benchTask === "Defect_Diagnosis_7Class" ? "active" : ""
              }`}
              onClick={() => setBenchTask("Defect_Diagnosis_7Class")}
            >
              7-Class Defect Diagnosis
            </button>
            <button
              type="button"
              className={`tab-btn ${
                benchTask === "Readiness_Binary" ? "active" : ""
              }`}
              onClick={() => setBenchTask("Readiness_Binary")}
            >
              Binary Readiness Gate
            </button>
          </div>
        </div>

        <div className="winner-cards-grid">
          <div className="winner-card">
            <div className="winner-head">
              <span className="winner-tag">Final 7-Class Diagnosis Model</span>
              <Cpu size={18} color="#3b82f6" />
            </div>
            <h3>SoftVoting_Ensemble</h3>
            <p className="winner-sub">
              Tuned RandomForest (350 trees) + XGBoost (lr=0.03, depth=4) +
              LightGBM (leaves=23, depth=3)
            </p>
            <div className="winner-metrics">
              <div>
                <span>5-Fold CV Acc</span>
                <strong>77.14%</strong>
              </div>
              <div>
                <span>CV Macro F1</span>
                <strong>0.7655</strong>
              </div>
              <div>
                <span>Holdout Acc</span>
                <strong>76.79%</strong>
              </div>
              <div>
                <span>Holdout F1</span>
                <strong>0.7721</strong>
              </div>
            </div>
          </div>

          <div className="winner-card">
            <div className="winner-head">
              <span className="winner-tag green">Final Binary Readiness Gate</span>
              <ShieldCheck size={18} color="#10b981" />
            </div>
            <h3>XGBoost_Tuned</h3>
            <p className="winner-sub">
              220 trees, max_depth=5, lr=0.08, subsample=0.75, reg_alpha=0.5,
              scale_pos_weight=6.0
            </p>
            <div className="winner-metrics">
              <div>
                <span>5-Fold CV Acc</span>
                <strong>86.79%</strong>
              </div>
              <div>
                <span>CV Std</span>
                <strong>±0.0143</strong>
              </div>
              <div>
                <span>CV Macro F1</span>
                <strong>0.7302</strong>
              </div>
              <div>
                <span>Holdout Acc</span>
                <strong>85.71%</strong>
              </div>
            </div>
          </div>
        </div>

        <div className="table-scroll">
          <table className="data-table">
            <thead>
              <tr>
                <th>Stage</th>
                <th>Model</th>
                <th>5-Fold CV Acc</th>
                <th>CV Std</th>
                <th>CV Macro F1</th>
                <th>Holdout Acc</th>
                <th>Holdout Macro F1</th>
              </tr>
            </thead>
            <tbody>
              {benchRows.map((row) => (
                <tr
                  key={`${row.stage}-${row.model}`}
                  className={row.winner ? "winner-row" : ""}
                >
                  <td>{row.stage}</td>
                  <td className="mono-strong">
                    {row.model}{" "}
                    {row.winner && <span className="win-pill">SELECTED</span>}
                  </td>
                  <td>{row.cvAcc}</td>
                  <td>{row.cvStd}</td>
                  <td>{row.cvF1}</td>
                  <td>{row.holdAcc}</td>
                  <td>{row.holdF1}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      {/* Holdout Per-Class F1 & Top 10 Feature Importances */}
      <section className="hiw-two-col">
        <div className="card">
          <div className="card-title">
            <BarChart3 size={18} />
            <span>Unseen Holdout Per-Class F1 (56 Test Videos)</span>
          </div>
          <p className="section-desc">
            Strong precision/recall on glare, orbit cutoff, low light, blur, and
            flicker. Deterministic Geometric Guardrails backstop{" "}
            <code>low_overlap_fast_move</code> where 2.5 FPS downsampling
            compresses frame-drop gaps.
          </p>
          <div className="prob-list" style={{ marginTop: "1rem" }}>
            {HOLDOUT_CLASS_REPORT.map((item) => (
              <div key={item.cls} className="prob-row hiw-bar-row">
                <span className="prob-name">{item.cls}</span>
                <div className="prob-track">
                  <div
                    className={`prob-fill ${
                      item.f1 >= 0.8 ? "clean" : item.f1 < 0.5 ? "alert" : ""
                    }`}
                    style={{ width: `${item.f1 * 100}%` }}
                  />
                </div>
                <span className="prob-val">{item.f1.toFixed(3)}</span>
              </div>
            ))}
          </div>
        </div>

        <div className="card">
          <div className="card-title">
            <Layers size={18} />
            <span>Top 10 Consensus Feature Importances</span>
          </div>
          <p className="section-desc">
            Normalized feature importances averaged across Tuned Random Forest,
            XGBoost, and LightGBM inside the Soft-Voting Ensemble.
          </p>
          <div className="prob-list" style={{ marginTop: "1rem" }}>
            {TOP_FEATURES.map((feat) => (
              <div key={feat.name} className="prob-row hiw-bar-row">
                <span className="prob-name mono-feat" title={feat.role}>
                  {feat.name}
                </span>
                <div className="prob-track">
                  <div
                    className="prob-fill"
                    style={{ width: `${(feat.importance / 0.16) * 100}%` }}
                  />
                </div>
                <span className="prob-val">
                  {(feat.importance * 100).toFixed(1)}%
                </span>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Completed Final Output Showcase */}
      <section className="card">
        <div className="section-head">
          <div>
            <div className="card-title">
              <MonitorCheck size={18} />
              <span>Completed Final Output — Live Dashboard Results</span>
            </div>
            <p className="section-desc">
              End-to-end production runs captured on real uploaded test &amp;
              smartphone videos, demonstrating both the Soft-Voting Ensemble and
              deterministic Geometric Guardrail in action. Click any screenshot
              to inspect full telemetry.
            </p>
          </div>
        </div>

        <div className="output-gallery-grid">
          {FINAL_OUTPUT_GALLERY.map((item) => (
            <div
              key={item.src}
              className="output-card"
              onClick={() => setLightboxItem(item)}
            >
              <div className="output-img-wrap">
                <img src={item.src} alt={item.file} />
                <span className="output-zoom-pill">
                  <Maximize2 size={13} />
                  <span>Inspect</span>
                </span>
              </div>
              <div className="output-body">
                <div className="output-meta-row">
                  <span className={`verdict-badge ${item.tier}`}>
                    Score {item.score}/100 &bull; {item.verdict}
                  </span>
                  <span className="tag">{item.trigger}</span>
                </div>
                <h4 className="mono-strong">{item.file}</h4>
                <p>{item.summary}</p>
              </div>
            </div>
          ))}
        </div>
      </section>

      {lightboxItem && (
        <div
          className="lightbox-backdrop"
          onClick={() => setLightboxItem(null)}
        >
          <div
            className="lightbox-modal"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="lightbox-head">
              <div>
                <span className={`verdict-badge ${lightboxItem.tier}`}>
                  Score {lightboxItem.score}/100 &bull; {lightboxItem.verdict}
                </span>
                <h3>{lightboxItem.file}</h3>
              </div>
              <button
                type="button"
                className="lightbox-close"
                onClick={() => setLightboxItem(null)}
                aria-label="Close preview"
              >
                <X size={18} />
              </button>
            </div>
            <img src={lightboxItem.src} alt={lightboxItem.file} />
            <p className="lightbox-desc">{lightboxItem.summary}</p>
          </div>
        </div>
      )}
    </div>
  );
}
