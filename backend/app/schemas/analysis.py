from typing import Dict, List, Any
from pydantic import BaseModel


class RecommendationItem(BaseModel):
    issue_code: str
    confidence: float
    detected_by: str
    title: str
    category: str
    severity: str
    impact: str
    action_steps: List[str]


class AnalysisResponse(BaseModel):
    video_file: str
    verdict: str
    readiness_score: int
    summary: str
    defect_probabilities: Dict[str, float]
    recommendations: List[RecommendationItem]
    extracted_metrics: Dict[str, Any]


class HealthResponse(BaseModel):
    status: str
    defect_model: str
    binary_model: str
    num_features: int


class SampleVideoItem(BaseModel):
    sample_id: str
    category: str
    defect_type: str
    readiness_label: str