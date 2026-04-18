"""
Main assessment API routes.
Handles video upload + dispatches to the appropriate task analyzer.
"""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from config import MAX_VIDEO_SIZE_MB, SUPPORTED_VIDEO_FORMATS, UPLOAD_DIR
from core.rule_engine import norm_comparison_text
from models.schemas import (
    AssessmentResponse,
    AttentionBehavior,
    FineMotorAction,
    GrossMotorAction,
    ScoreLevel,
    TaskType,
)
from tasks.fine_motor import (
    analyze_block_stacking,
    analyze_object_in_container,
    analyze_pincer_grasp,
)
from tasks.gross_motor import (
    analyze_ball_throwing,
    analyze_single_leg_stand,
    analyze_stair_climbing,
    analyze_walking,
)
from tasks.joint_attention import (
    analyze_eye_gaze,
    analyze_head_nod_shake,
    analyze_imitation,
    analyze_pointing,
)
from tasks.pain_monitor import analyze_pain

router = APIRouter(prefix="/api/v1/assessment", tags=["assessment"])


def _save_upload(file: UploadFile) -> Path:
    suffix = Path(file.filename or "video.mp4").suffix.lower()
    if suffix not in SUPPORTED_VIDEO_FORMATS:
        raise HTTPException(400, f"Unsupported format: {suffix}. Supported: {SUPPORTED_VIDEO_FORMATS}")

    file_id = uuid.uuid4().hex
    dest = UPLOAD_DIR / f"{file_id}{suffix}"
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)

    size_mb = dest.stat().st_size / (1024 * 1024)
    if size_mb > MAX_VIDEO_SIZE_MB:
        dest.unlink()
        raise HTTPException(413, f"Video too large: {size_mb:.1f}MB (max {MAX_VIDEO_SIZE_MB}MB)")

    return dest


def _recommendation(score: ScoreLevel, task: str) -> str:
    if score == ScoreLevel.NORMAL:
        return f"{task}: within normal developmental range. Continue routine monitoring."
    if score == ScoreLevel.BORDERLINE:
        return (
            f"{task}: borderline result. Recommend follow-up assessment in 1-2 months. "
            "Discuss with pediatrician at next visit."
        )
    return (
        f"{task}: below expected range. Recommend prompt consultation with pediatrician "
        "or developmental specialist for comprehensive evaluation."
    )


# ─── Gross Motor ─────────────────────────────────────────────────

@router.post("/gross-motor", response_model=AssessmentResponse)
async def assess_gross_motor(
    video: UploadFile = File(...),
    child_age_months: int = Form(..., ge=0, le=60),
    action: GrossMotorAction = Form(...),
):
    path = _save_upload(video)
    try:
        analyzers = {
            GrossMotorAction.SINGLE_LEG_STAND: analyze_single_leg_stand,
            GrossMotorAction.WALKING: analyze_walking,
            GrossMotorAction.BALL_THROWING: analyze_ball_throwing,
            GrossMotorAction.STAIR_CLIMBING: analyze_stair_climbing,
        }
        result = analyzers[action](str(path), child_age_months)
    finally:
        path.unlink(missing_ok=True)

    return AssessmentResponse(
        task_type=TaskType.GROSS_MOTOR,
        child_age_months=child_age_months,
        score_level=result.score_level,
        result=result,
        norm_comparison=norm_comparison_text(child_age_months, "gross_motor"),
        recommendation=_recommendation(result.score_level, f"Gross motor ({action.value})"),
    )


# ─── Fine Motor ──────────────────────────────────────────────────

@router.post("/fine-motor", response_model=AssessmentResponse)
async def assess_fine_motor(
    video: UploadFile = File(...),
    child_age_months: int = Form(..., ge=0, le=60),
    action: FineMotorAction = Form(...),
):
    path = _save_upload(video)
    try:
        analyzers = {
            FineMotorAction.PINCER_GRASP: analyze_pincer_grasp,
            FineMotorAction.BLOCK_STACKING: analyze_block_stacking,
            FineMotorAction.OBJECT_IN_CONTAINER: analyze_object_in_container,
        }
        result = analyzers[action](str(path), child_age_months)
    finally:
        path.unlink(missing_ok=True)

    return AssessmentResponse(
        task_type=TaskType.FINE_MOTOR,
        child_age_months=child_age_months,
        score_level=result.score_level,
        result=result,
        norm_comparison=norm_comparison_text(child_age_months, "fine_motor"),
        recommendation=_recommendation(result.score_level, f"Fine motor ({action.value})"),
    )


# ─── Joint Attention ─────────────────────────────────────────────

@router.post("/joint-attention", response_model=AssessmentResponse)
async def assess_joint_attention(
    video: UploadFile = File(...),
    child_age_months: int = Form(..., ge=0, le=60),
    behavior: AttentionBehavior = Form(...),
):
    path = _save_upload(video)
    try:
        analyzers = {
            AttentionBehavior.POINTING: analyze_pointing,
            AttentionBehavior.EYE_GAZE: analyze_eye_gaze,
            AttentionBehavior.HEAD_NOD_SHAKE: analyze_head_nod_shake,
        }
        if behavior == AttentionBehavior.IMITATION:
            raise HTTPException(400, "Imitation requires two videos — use /joint-attention/imitation endpoint.")
        result = analyzers[behavior](str(path), child_age_months)
    finally:
        path.unlink(missing_ok=True)

    return AssessmentResponse(
        task_type=TaskType.JOINT_ATTENTION,
        child_age_months=child_age_months,
        score_level=result.score_level,
        result=result,
        norm_comparison=norm_comparison_text(child_age_months, "social_emotional"),
        recommendation=_recommendation(result.score_level, f"Joint attention ({behavior.value})"),
    )


@router.post("/joint-attention/imitation", response_model=AssessmentResponse)
async def assess_imitation(
    reference_video: UploadFile = File(..., description="Demonstrator / reference video"),
    child_video: UploadFile = File(..., description="Child's imitation video"),
    child_age_months: int = Form(..., ge=0, le=60),
):
    ref_path = _save_upload(reference_video)
    child_path = _save_upload(child_video)
    try:
        result = analyze_imitation(str(ref_path), str(child_path), child_age_months)
    finally:
        ref_path.unlink(missing_ok=True)
        child_path.unlink(missing_ok=True)

    return AssessmentResponse(
        task_type=TaskType.JOINT_ATTENTION,
        child_age_months=child_age_months,
        score_level=result.score_level,
        result=result,
        norm_comparison=norm_comparison_text(child_age_months, "social_emotional"),
        recommendation=_recommendation(result.score_level, "Imitation"),
    )


# ─── Pain Monitor ────────────────────────────────────────────────

@router.post("/pain-monitor", response_model=AssessmentResponse)
async def assess_pain(
    video: UploadFile = File(...),
    child_age_months: int = Form(..., ge=0, le=60),
):
    path = _save_upload(video)
    try:
        result = analyze_pain(str(path), child_age_months)
    finally:
        path.unlink(missing_ok=True)

    return AssessmentResponse(
        task_type=TaskType.PAIN_MONITOR,
        child_age_months=child_age_months,
        score_level=result.score_level,
        result=result,
        recommendation=_recommendation(result.score_level, "Pain monitoring (NFCS)"),
    )
