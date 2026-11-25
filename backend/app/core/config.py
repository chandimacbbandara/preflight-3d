import os

PROJECT_NAME = "PreFlight-3D API"
API_V1_PREFIX = "/api"

MODEL_BUNDLE_PATH = os.path.abspath("models/preflight_model_bundle.joblib")
MANIFEST_CSV_PATH = os.path.abspath("data/processed/quality_measurements/dataset_manifest.csv")

ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}
CORS_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]