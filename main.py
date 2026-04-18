"""
CDC Milestone Tracker — Backend Entry Point

Architecture (3-layer, no end-to-end training):
  Layer 1: MediaPipe API (Pose / Face Mesh / Hands) — direct use, no training
  Layer 2: Hand-written clinical feature extraction — geometric calculations
  Layer 3: Rule-based scoring — threshold rules aligned with clinical norms

Tasks:
  1. Gross Motor Assessment (walking, single-leg stand, stair climbing, ball throwing)
  2. Fine Motor Assessment (pincer grasp, block stacking, object placement)
  3. Joint Attention / Gaze (pointing, eye gaze Route A, head nod/shake, imitation DTW)
  4. Pain / Discomfort Monitoring (NFCS-based AU threshold scoring)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api.routes.assessment import router as assessment_router
from api.routes.realtime import router as realtime_router
from api.routes.report import router as report_router

STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(
    title="CDC Milestone Tracker",
    description="AI-assisted pediatric developmental screening via video analysis",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(assessment_router)
app.include_router(realtime_router)
app.include_router(report_router)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
async def health():
    return {"status": "ok", "version": "0.1.0"}


@app.get("/api/v1/tasks")
async def list_tasks():
    return {
        "tasks": [
            {
                "id": "gross_motor",
                "name": "Gross Motor Assessment",
                "actions": ["single_leg_stand", "walking", "ball_throwing", "stair_climbing"],
                "method": "MediaPipe Pose → clinical features → rule scoring",
            },
            {
                "id": "fine_motor",
                "name": "Fine Motor Assessment",
                "actions": ["pincer_grasp", "block_stacking", "object_in_container"],
                "method": "MediaPipe Hands → geometric features → rule scoring",
            },
            {
                "id": "joint_attention",
                "name": "Joint Attention / Gaze Tracking",
                "behaviors": ["pointing", "eye_gaze", "head_nod_shake", "imitation"],
                "method": "Route A: Face Mesh head pose + eye offset → rule engine",
            },
            {
                "id": "pain_monitor",
                "name": "Pain / Discomfort Monitoring",
                "method": "Face Mesh → FACS AU geometric features → NFCS threshold scoring",
            },
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
