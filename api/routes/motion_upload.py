"""
Motion Monitor – Video Upload Route.
Accepts a video file + focus target, processes frames with MediaPipe Pose + Hands,
returns per-frame landmark data for frontend replay with skeleton overlay.
"""

from __future__ import annotations

import shutil
import time
import uuid
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.responses import JSONResponse

from config import MAX_VIDEO_SIZE_MB, SUPPORTED_VIDEO_FORMATS, UPLOAD_DIR
from core.developmental_scorer import DevelopmentalScorer, assessment_to_dict
from core.mediapipe_engine import HandDetector, PoseDetector
from core.realtime_processor import RealtimeProcessor
from utils.geometry import joint_angle, landmark_to_array, trunk_lean_angle

router = APIRouter(prefix="/api/v1/motion", tags=["motion"])

# Body-part focus → which MediaPipe landmarks to include
FOCUS_LANDMARK_GROUPS: dict[str, list[int]] = {
    "full_body": list(range(33)),
    "arms":      [11, 12, 13, 14, 15, 16],          # shoulders → elbows → wrists
    "legs":      [23, 24, 25, 26, 27, 28, 29, 30, 31, 32],  # hips → knees → ankles → feet
    "hands":     [],                                  # use hand detector only
    "head":      [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],  # nose + eyes + ears
    "torso":     [11, 12, 23, 24],                    # shoulders + hips
}

# Skeleton connections per focus
FOCUS_CONNECTIONS: dict[str, list[tuple[int, int]]] = {
    "full_body": [
        (11,12),(11,13),(13,15),(12,14),(14,16),
        (11,23),(12,24),(23,24),
        (23,25),(25,27),(27,29),(27,31),
        (24,26),(26,28),(28,30),(28,32),
        (0,11),(0,12),
    ],
    "arms":  [(11,12),(11,13),(13,15),(12,14),(14,16)],
    "legs":  [(23,24),(23,25),(25,27),(27,29),(27,31),(24,26),(26,28),(28,30),(28,32)],
    "hands": [],
    "head":  [(0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),(9,10)],
    "torso": [(11,12),(11,23),(12,24),(23,24)],
}

SAMPLE_EVERY_N = 2   # process every 2nd frame to keep response size reasonable
MAX_FRAMES    = 600  # cap at 600 processed frames (~20s at 30fps sampled every 2)
SESSION_TTL_SECONDS = 60 * 60
SESSION_VIDEO_DIR = UPLOAD_DIR / "sessions"
SESSION_VIDEO_DIR.mkdir(parents=True, exist_ok=True)
_SESSION_FILES: dict[str, dict[str, float | Path]] = {}
UPLOAD_TASKS = {
    "gross_motor",
    "fine_motor",
    "joint_attention",
    "pain_monitor",
    "motion_monitor",
}


def _cleanup_expired_sessions() -> None:
    now = time.time()
    expired = [
        sid for sid, meta in _SESSION_FILES.items()
        if now - float(meta["created_at"]) > SESSION_TTL_SECONDS
    ]
    for sid in expired:
        entry = _SESSION_FILES.pop(sid, None)
        if entry:
            Path(entry["path"]).unlink(missing_ok=True)


def _register_session_video(path: Path) -> str:
    _cleanup_expired_sessions()
    session_id = uuid.uuid4().hex
    _SESSION_FILES[session_id] = {
        "path": path,
        "created_at": time.time(),
    }
    return session_id


def _delete_session_video(session_id: str) -> bool:
    entry = _SESSION_FILES.pop(session_id, None)
    if not entry:
        return False
    Path(entry["path"]).unlink(missing_ok=True)
    return True


def _save_upload(file: UploadFile) -> Path:
    suffix = Path(file.filename or "video.mp4").suffix.lower()
    if suffix not in SUPPORTED_VIDEO_FORMATS:
        raise HTTPException(400, f"Unsupported format: {suffix}")
    file_id = uuid.uuid4().hex
    dest = SESSION_VIDEO_DIR / f"motion_{file_id}{suffix}"
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)
    size_mb = dest.stat().st_size / (1024 * 1024)
    if size_mb > MAX_VIDEO_SIZE_MB:
        dest.unlink()
        raise HTTPException(413, f"Video too large ({size_mb:.1f} MB, max {MAX_VIDEO_SIZE_MB} MB)")
    return dest


def _extract_features(lm, focus: str) -> dict:
    """Compute joint angles relevant to the chosen focus."""
    features: dict = {}
    try:
        if focus in ("full_body", "arms"):
            ls = landmark_to_array(lm[11]); rs = landmark_to_array(lm[12])
            le = landmark_to_array(lm[13]); re = landmark_to_array(lm[14])
            lw = landmark_to_array(lm[15]); rw = landmark_to_array(lm[16])
            features["left_elbow_angle"]  = round(joint_angle(ls, le, lw), 1)
            features["right_elbow_angle"] = round(joint_angle(rs, re, rw), 1)
            features["left_shoulder_angle"]  = round(joint_angle(landmark_to_array(lm[23]), ls, le), 1)
            features["right_shoulder_angle"] = round(joint_angle(landmark_to_array(lm[24]), rs, re), 1)

        if focus in ("full_body", "legs"):
            lh = landmark_to_array(lm[23]); rh = landmark_to_array(lm[24])
            lk = landmark_to_array(lm[25]); rk = landmark_to_array(lm[26])
            la = landmark_to_array(lm[27]); ra = landmark_to_array(lm[28])
            ls = landmark_to_array(lm[11]); rs = landmark_to_array(lm[12])
            features["left_knee_angle"]  = round(joint_angle(lh, lk, la), 1)
            features["right_knee_angle"] = round(joint_angle(rh, rk, ra), 1)
            features["left_hip_angle"]   = round(joint_angle(ls, lh, lk), 1)
            features["right_hip_angle"]  = round(joint_angle(rs, rh, rk), 1)

        if focus in ("full_body", "torso"):
            features["trunk_lean"] = round(trunk_lean_angle(lm), 1)

    except (IndexError, AttributeError):
        pass
    return features


@router.post("/upload")
async def upload_motion_video(
    file: UploadFile = File(...),
    focus: str = Form("full_body"),
):
    """
    Upload a video and run motion tracking on it.

    - **file**: video file (mp4, avi, mov, mkv, webm)
    - **focus**: one of full_body | arms | legs | hands | head | torso
    """
    if focus not in FOCUS_LANDMARK_GROUPS:
        raise HTTPException(400, f"Invalid focus '{focus}'. Choose from: {list(FOCUS_LANDMARK_GROUPS)}")

    video_path = _save_upload(file)
    session_id = _register_session_video(video_path)

    pose_det  = PoseDetector()  if focus != "hands" else None
    hand_det  = HandDetector()  if focus in ("hands", "full_body", "arms") else None

    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise HTTPException(422, "Could not open video file")

        fps       = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_raw = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        frames_out: list[dict] = []
        raw_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            raw_idx += 1

            # Sample every N frames
            if raw_idx % SAMPLE_EVERY_N != 0:
                continue
            if len(frames_out) >= MAX_FRAMES:
                break

            timestamp = raw_idx / fps
            frame_data: dict = {
                "frame_idx": raw_idx,
                "timestamp":  round(timestamp, 3),
                "pose_landmarks": [],
                "hand_landmarks": [],
                "features": {},
                "connections": FOCUS_CONNECTIONS[focus],
                "focus_indices": FOCUS_LANDMARK_GROUPS[focus],
            }

            # Pose
            if pose_det:
                result = pose_det.process(frame)
                if result.pose_landmarks:
                    lm = result.pose_landmarks.landmark
                    all_lms = [[l.x, l.y, l.z, l.visibility] for l in lm]
                    # Only include landmarks relevant to focus
                    focus_idx = FOCUS_LANDMARK_GROUPS[focus]
                    if focus_idx:
                        frame_data["pose_landmarks"] = [
                            all_lms[i] if i < len(all_lms) else None for i in focus_idx
                        ]
                    else:
                        frame_data["pose_landmarks"] = all_lms
                    frame_data["features"] = _extract_features(lm, focus)

            # Hands
            if hand_det:
                hresult = hand_det.process(frame)
                if hresult.multi_hand_landmarks:
                    frame_data["hand_landmarks"] = [
                        [[l.x, l.y, l.z] for l in hl.landmark]
                        for hl in hresult.multi_hand_landmarks[:2]
                    ]

            frames_out.append(frame_data)

        cap.release()

    finally:
        if pose_det:
            pose_det.close()
        if hand_det:
            hand_det.close()

    return JSONResponse({
        "session_id": session_id,
        "video_url": f"/api/v1/motion/session/{session_id}/video",
        "video_fps":     fps,
        "sample_every":  SAMPLE_EVERY_N,
        "width":         width,
        "height":        height,
        "total_raw_frames": total_raw,
        "processed_frames": len(frames_out),
        "focus":         focus,
        "frames":        frames_out,
    })


@router.get("/session/{session_id}/video")
async def get_session_video(session_id: str):
    _cleanup_expired_sessions()
    entry = _SESSION_FILES.get(session_id)
    if not entry:
        raise HTTPException(404, "Session video not found or expired")
    path = Path(entry["path"])
    if not path.exists():
        _SESSION_FILES.pop(session_id, None)
        raise HTTPException(404, "Session video file missing")
    return FileResponse(path)


@router.post("/session/{session_id}/close")
async def close_session_video(session_id: str):
    deleted = _delete_session_video(session_id)
    return JSONResponse({"ok": True, "deleted": deleted})


@router.post("/analyze")
async def upload_analyze_video(
    file: UploadFile = File(...),
    task: str = Form("gross_motor"),
    age_months: int = Form(24),
    focus: str = Form("full_body"),
):
    """
    Upload a recorded video and analyze it for any assessment task.

    - **file**: video file (mp4, avi, mov, mkv, webm)
    - **task**: gross_motor | fine_motor | joint_attention | pain_monitor | motion_monitor
    - **age_months**: child age used for scoring context
    - **focus**: optional motion-monitor focus value, currently passthrough for UI compatibility
    """
    if task not in UPLOAD_TASKS:
        raise HTTPException(400, f"Invalid task '{task}'. Choose from: {sorted(UPLOAD_TASKS)}")

    video_path = _save_upload(file)
    session_id = _register_session_video(video_path)
    processor = RealtimeProcessor(task=task, age_months=age_months)
    scorer = DevelopmentalScorer(child_age_months=age_months)

    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise HTTPException(422, "Could not open video file")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_raw = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        frames_out: list[dict] = []
        raw_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            raw_idx += 1

            if raw_idx % SAMPLE_EVERY_N != 0:
                continue
            if len(frames_out) >= MAX_FRAMES:
                break

            result = processor.process_frame(frame)
            features = result.get("features", {})
            if task == "gross_motor":
                scorer.update_from_gross_motor(features)
            elif task == "fine_motor":
                scorer.update_from_fine_motor(features)
            elif task == "joint_attention":
                scorer.update_from_attention(features)
            elif task == "pain_monitor":
                scorer.update_from_pain(result)
            elif task == "motion_monitor":
                scorer.update_from_gross_motor(features)

            frames_out.append({
                "frame_idx": raw_idx,
                "timestamp": round(raw_idx / fps, 3),
                "result": result,
            })

        cap.release()
        assessment = scorer.compute_assessment()

    finally:
        processor.close()

    return JSONResponse({
        "session_id": session_id,
        "video_url": f"/api/v1/motion/session/{session_id}/video",
        "video_fps": fps,
        "sample_every": SAMPLE_EVERY_N,
        "width": width,
        "height": height,
        "total_raw_frames": total_raw,
        "processed_frames": len(frames_out),
        "task": task,
        "focus": focus,
        "frames": frames_out,
        "assessment": assessment_to_dict(assessment),
    })
