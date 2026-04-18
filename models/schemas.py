from __future__ import annotations
from enum import Enum
from pydantic import BaseModel, Field


class TaskType(str, Enum):
    GROSS_MOTOR = "gross_motor"
    FINE_MOTOR = "fine_motor"
    JOINT_ATTENTION = "joint_attention"
    PAIN_MONITOR = "pain_monitor"


class GrossMotorAction(str, Enum):
    SINGLE_LEG_STAND = "single_leg_stand"
    STAIR_CLIMBING = "stair_climbing"
    BALL_THROWING = "ball_throwing"
    WALKING = "walking"


class FineMotorAction(str, Enum):
    PINCER_GRASP = "pincer_grasp"
    BLOCK_STACKING = "block_stacking"
    OBJECT_IN_CONTAINER = "object_in_container"


class AttentionBehavior(str, Enum):
    POINTING = "pointing"
    EYE_GAZE = "eye_gaze"
    HEAD_NOD_SHAKE = "head_nod_shake"
    IMITATION = "imitation"


# ---------- Request ----------

class AssessmentRequest(BaseModel):
    child_age_months: int = Field(..., ge=0, le=60, description="Child age in months")
    task_type: TaskType
    action: str | None = Field(None, description="Specific action within the task")


# ---------- Shared result components ----------

class ScoreLevel(str, Enum):
    NORMAL = "normal"
    BORDERLINE = "borderline"
    AT_RISK = "at_risk"


class FeatureValue(BaseModel):
    name: str
    value: float
    unit: str = ""
    normal_range: str = ""


class FrameResult(BaseModel):
    frame_index: int
    timestamp_sec: float
    features: list[FeatureValue] = []


# ---------- Task-specific results ----------

class GrossMotorResult(BaseModel):
    action: str
    score_level: ScoreLevel
    features: list[FeatureValue]
    percentile: float | None = None
    summary: str = ""


class FineMotorResult(BaseModel):
    action: str
    score_level: ScoreLevel
    features: list[FeatureValue]
    summary: str = ""


class GazeMetrics(BaseModel):
    gaze_direction: str
    head_yaw_deg: float
    head_pitch_deg: float
    eye_relative_offset_x: float
    eye_relative_offset_y: float


class JointAttentionResult(BaseModel):
    behavior: str
    score_level: ScoreLevel
    features: list[FeatureValue]
    gaze_timeline: list[GazeMetrics] = []
    latency_sec: float | None = None
    summary: str = ""


class AUActivation(BaseModel):
    au_name: str
    au_code: str
    intensity: float = Field(..., ge=0.0, le=1.0)


class PainMonitorResult(BaseModel):
    nfcs_score: float
    score_level: ScoreLevel
    au_activations: list[AUActivation]
    features: list[FeatureValue]
    summary: str = ""


# ---------- Unified response ----------

class AssessmentResponse(BaseModel):
    task_type: TaskType
    child_age_months: int
    score_level: ScoreLevel
    result: GrossMotorResult | FineMotorResult | JointAttentionResult | PainMonitorResult
    norm_comparison: str = ""
    recommendation: str = ""
