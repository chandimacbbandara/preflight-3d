from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.routes import router as api_router
from backend.app.core.config import API_V1_PREFIX, CORS_ORIGINS, PROJECT_NAME

app = FastAPI(
    title=PROJECT_NAME,
    description="Pre-flight Computer Vision & ML quality gate for 3D reconstruction video captures.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix=API_V1_PREFIX)


@app.get("/")
def root():
    return {
        "service": PROJECT_NAME,
        "docs": "/docs",
        "health": f"{API_V1_PREFIX}/health",
    }