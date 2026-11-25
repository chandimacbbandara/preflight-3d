import os
import shutil
import tempfile
from functools import lru_cache
from typing import List
import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile

from backend.app.core.config import (
    ALLOWED_VIDEO_EXTENSIONS,
    MANIFEST_CSV_PATH,
    MODEL_BUNDLE_PATH,
)
from backend.app.schemas.analysis import (
    AnalysisResponse,
    HealthResponse,
    SampleVideoItem,
)
from backend.app.services.recommendation_engine import PreFlightAnalyzer

router = APIRouter()


@lru_cache(maxsize=1)
def get_analyzer() -> PreFlightAnalyzer:
    return PreFlightAnalyzer(bundle_path=MODEL_BUNDLE_PATH)


@router.get("/health", response_model=HealthResponse)
def health_check():
    analyzer = get_analyzer()
    return HealthResponse(
        status="ok",
        defect_model=analyzer.bundle.get("defect_model_name", "unknown"),
        binary_model=analyzer.bundle.get("binary_model_name", "unknown"),
        num_features=len(analyzer.feature_cols),
    )


@router.post("/analyze", response_model=AnalysisResponse)
def analyze_uploaded_video(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No video file provided.")

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format '{ext}'. Allowed formats: {sorted(ALLOWED_VIDEO_EXTENSIONS)}",
        )

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        analyzer = get_analyzer()
        result = analyzer.analyze_video(tmp_path)
        result["video_file"] = file.filename
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Video analysis failed: {str(exc)}")
    finally:
        file.file.close()
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


@router.get("/samples", response_model=List[SampleVideoItem])
def list_dataset_samples():
    """Return a curated list of clean & degraded dataset samples for instant UI testing."""
    if not os.path.exists(MANIFEST_CSV_PATH):
        return []

    df = pd.read_csv(MANIFEST_CSV_PATH)
    # Pick 2 representative samples per defect_type (14 total)
    sampled = df.groupby("defect_type", group_keys=False).head(2)
    return [
        SampleVideoItem(
            sample_id=row["sample_id"],
            category=row["category"],
            defect_type=row["defect_type"],
            readiness_label=row["readiness_label"],
        )
        for _, row in sampled.iterrows()
    ]


@router.post("/analyze-sample/{sample_id}", response_model=AnalysisResponse)
def analyze_dataset_sample(sample_id: str):
    """Run live inference on a sample from the local dataset manifest."""
    if not os.path.exists(MANIFEST_CSV_PATH):
        raise HTTPException(status_code=404, detail="Dataset manifest not found.")

    df = pd.read_csv(MANIFEST_CSV_PATH)
    match = df[df["sample_id"] == sample_id]
    if match.empty:
        raise HTTPException(status_code=404, detail=f"Sample '{sample_id}' not found.")

    video_path = match.iloc[0]["video_path"]
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Sample video file missing on disk.")

    analyzer = get_analyzer()
    return analyzer.analyze_video(video_path)