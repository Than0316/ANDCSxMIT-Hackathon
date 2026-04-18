"""
Layer 3 — Rule-based scoring engine (no training).
Threshold rules + norm comparison for all tasks.
"""

from __future__ import annotations

from models.norms import (
    CDC_MILESTONES,
    GAIT_SYMMETRY_NORM,
    JOINT_ATTENTION_LATENCY_NORMS,
    NFCS_BORDERLINE_THRESHOLD,
    NFCS_PAIN_THRESHOLD,
    SINGLE_LEG_STAND_NORMS,
    STEP_FREQUENCY_NORMS,
    get_closest_age_norm,
)
from models.schemas import ScoreLevel


def score_single_leg_stand(duration_sec: float, age_months: int) -> ScoreLevel:
    borderline, normal = get_closest_age_norm(age_months, SINGLE_LEG_STAND_NORMS)
    if duration_sec >= normal:
        return ScoreLevel.NORMAL
    if duration_sec >= borderline:
        return ScoreLevel.BORDERLINE
    return ScoreLevel.AT_RISK


def score_gait_symmetry(symmetry_ratio: float) -> ScoreLevel:
    borderline, normal = GAIT_SYMMETRY_NORM
    if symmetry_ratio >= normal:
        return ScoreLevel.NORMAL
    if symmetry_ratio >= borderline:
        return ScoreLevel.BORDERLINE
    return ScoreLevel.AT_RISK


def score_step_frequency(freq_hz: float, age_months: int) -> ScoreLevel:
    low, high = get_closest_age_norm(age_months, STEP_FREQUENCY_NORMS)
    if low <= freq_hz <= high:
        return ScoreLevel.NORMAL
    if abs(freq_hz - low) < 0.3 or abs(freq_hz - high) < 0.3:
        return ScoreLevel.BORDERLINE
    return ScoreLevel.AT_RISK


def score_joint_attention_latency(latency_sec: float, age_months: int) -> ScoreLevel:
    borderline, normal = get_closest_age_norm(age_months, JOINT_ATTENTION_LATENCY_NORMS)
    if latency_sec <= normal:
        return ScoreLevel.NORMAL
    if latency_sec <= borderline:
        return ScoreLevel.BORDERLINE
    return ScoreLevel.AT_RISK


def score_nfcs(nfcs_total: float) -> ScoreLevel:
    if nfcs_total >= NFCS_PAIN_THRESHOLD:
        return ScoreLevel.AT_RISK
    if nfcs_total >= NFCS_BORDERLINE_THRESHOLD:
        return ScoreLevel.BORDERLINE
    return ScoreLevel.NORMAL


def score_pincer_grasp(thumb_index_dist: float, grasp_detected: bool) -> ScoreLevel:
    if grasp_detected and thumb_index_dist < 0.08:
        return ScoreLevel.NORMAL
    if grasp_detected:
        return ScoreLevel.BORDERLINE
    return ScoreLevel.AT_RISK


def score_block_stacking(stability_ratio: float, block_count: int) -> ScoreLevel:
    if block_count >= 3 and stability_ratio < 0.15:
        return ScoreLevel.NORMAL
    if block_count >= 2:
        return ScoreLevel.BORDERLINE
    return ScoreLevel.AT_RISK


def score_object_placement(success_count: int, attempt_count: int) -> ScoreLevel:
    if attempt_count == 0:
        return ScoreLevel.AT_RISK
    ratio = success_count / attempt_count
    if ratio >= 0.8:
        return ScoreLevel.NORMAL
    if ratio >= 0.5:
        return ScoreLevel.BORDERLINE
    return ScoreLevel.AT_RISK


def aggregate_score(scores: list[ScoreLevel]) -> ScoreLevel:
    if any(s == ScoreLevel.AT_RISK for s in scores):
        return ScoreLevel.AT_RISK
    if any(s == ScoreLevel.BORDERLINE for s in scores):
        return ScoreLevel.BORDERLINE
    return ScoreLevel.NORMAL


def norm_comparison_text(age_months: int, task_domain: str) -> str:
    closest_age = min(CDC_MILESTONES.keys(), key=lambda a: abs(a - age_months))
    milestones = CDC_MILESTONES.get(closest_age, {})
    items = milestones.get(task_domain, [])
    if not items:
        return f"No CDC milestones available for {task_domain} at {age_months} months."
    bullet_list = "; ".join(items)
    return f"CDC milestones for ~{closest_age} months ({task_domain}): {bullet_list}"
