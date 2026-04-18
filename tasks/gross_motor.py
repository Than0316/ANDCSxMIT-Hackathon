"""
Task 1: Gross Motor Assessment
MediaPipe Pose → hand-written clinical features → rule-based scoring
"""

from __future__ import annotations

import numpy as np

from core.mediapipe_engine import PoseDetector
from core.rule_engine import (
    norm_comparison_text,
    score_gait_symmetry,
    score_single_leg_stand,
    score_step_frequency,
    aggregate_score,
)
from models.schemas import FeatureValue, GrossMotorResult, ScoreLevel
from utils.geometry import (
    angular_velocity,
    euclidean_distance,
    joint_angle,
    landmark_to_array,
    midpoint,
    trunk_lean_angle,
)
from utils.video_processor import extract_frames


def _extract_pose_features_per_frame(pose_landmarks) -> dict:
    lm = pose_landmarks.landmark

    left_hip = landmark_to_array(lm[23])
    left_knee = landmark_to_array(lm[25])
    left_ankle = landmark_to_array(lm[27])
    right_hip = landmark_to_array(lm[24])
    right_knee = landmark_to_array(lm[26])
    right_ankle = landmark_to_array(lm[28])

    left_shoulder = landmark_to_array(lm[11])
    right_shoulder = landmark_to_array(lm[12])
    left_elbow = landmark_to_array(lm[13])
    right_elbow = landmark_to_array(lm[14])
    left_wrist = landmark_to_array(lm[15])
    right_wrist = landmark_to_array(lm[16])

    hip_mid = midpoint(left_hip, right_hip)
    shoulder_mid = midpoint(left_shoulder, right_shoulder)

    return {
        "left_knee_angle": joint_angle(left_hip, left_knee, left_ankle),
        "right_knee_angle": joint_angle(right_hip, right_knee, right_ankle),
        "left_hip_angle": joint_angle(left_shoulder, left_hip, left_knee),
        "right_hip_angle": joint_angle(right_shoulder, right_hip, right_knee),
        "left_ankle_y": left_ankle[1],
        "right_ankle_y": right_ankle[1],
        "hip_mid_y": hip_mid[1],
        "shoulder_mid_y": shoulder_mid[1],
        "trunk_lean": trunk_lean_angle(lm),
        "left_shoulder_angle": joint_angle(left_hip, left_shoulder, left_elbow),
        "right_shoulder_angle": joint_angle(right_hip, right_shoulder, right_elbow),
        "left_elbow_angle": joint_angle(left_shoulder, left_elbow, left_wrist),
        "right_elbow_angle": joint_angle(right_shoulder, right_elbow, right_wrist),
        "com_y": (hip_mid[1] + shoulder_mid[1]) / 2.0,
    }


def analyze_single_leg_stand(video_path: str, age_months: int) -> GrossMotorResult:
    detector = PoseDetector()
    frames = extract_frames(video_path)

    standing_frames = 0
    total_frames = 0
    com_positions = []
    support_verticality_values = []

    for _, ts, frame in frames:
        result = detector.process(frame)
        if not result.pose_landmarks:
            continue
        feats = _extract_pose_features_per_frame(result.pose_landmarks)
        total_frames += 1

        left_lifted = feats["left_ankle_y"] < feats["right_ankle_y"] - 0.05
        right_lifted = feats["right_ankle_y"] < feats["left_ankle_y"] - 0.05

        if left_lifted or right_lifted:
            standing_frames += 1
            if left_lifted:
                support_verticality_values.append(feats["right_knee_angle"])
            else:
                support_verticality_values.append(feats["left_knee_angle"])

        com_positions.append(feats["com_y"])

    detector.close()

    if total_frames == 0:
        return GrossMotorResult(
            action="single_leg_stand",
            score_level=ScoreLevel.AT_RISK,
            features=[],
            summary="No pose detected in video.",
        )

    fps = len(frames) / (frames[-1][1] - frames[0][1] + 1e-8) if len(frames) > 1 else 15
    duration = standing_frames / fps
    com_std = float(np.std(com_positions)) if com_positions else 999.0
    avg_verticality = float(np.mean(support_verticality_values)) if support_verticality_values else 0.0

    score = score_single_leg_stand(duration, age_months)

    features = [
        FeatureValue(name="standing_duration", value=round(duration, 2), unit="sec"),
        FeatureValue(name="com_stability", value=round(com_std, 4), unit="std_dev"),
        FeatureValue(name="support_leg_verticality", value=round(avg_verticality, 1), unit="deg"),
        FeatureValue(name="standing_frame_ratio", value=round(standing_frames / total_frames, 2)),
    ]

    return GrossMotorResult(
        action="single_leg_stand",
        score_level=score,
        features=features,
        summary=f"Single-leg stand duration: {duration:.1f}s, CoM stability: {com_std:.4f}",
    )


def analyze_walking(video_path: str, age_months: int) -> GrossMotorResult:
    detector = PoseDetector()
    frames = extract_frames(video_path)

    left_ankle_y_series = []
    right_ankle_y_series = []
    timestamps = []
    trunk_leans = []

    for _, ts, frame in frames:
        result = detector.process(frame)
        if not result.pose_landmarks:
            continue
        feats = _extract_pose_features_per_frame(result.pose_landmarks)
        left_ankle_y_series.append(feats["left_ankle_y"])
        right_ankle_y_series.append(feats["right_ankle_y"])
        timestamps.append(ts)
        trunk_leans.append(feats["trunk_lean"])

    detector.close()

    if len(timestamps) < 10:
        return GrossMotorResult(
            action="walking",
            score_level=ScoreLevel.AT_RISK,
            features=[],
            summary="Insufficient frames for gait analysis.",
        )

    left_arr = np.array(left_ankle_y_series)
    right_arr = np.array(right_ankle_y_series)
    ts_arr = np.array(timestamps)
    duration = ts_arr[-1] - ts_arr[0]

    left_steps = _count_peaks(left_arr)
    right_steps = _count_peaks(right_arr)
    total_steps = left_steps + right_steps
    step_freq = total_steps / duration if duration > 0 else 0

    symmetry = min(left_steps, right_steps) / max(left_steps, right_steps, 1)
    avg_trunk_lean = float(np.mean(trunk_leans))

    freq_score = score_step_frequency(step_freq, age_months)
    sym_score = score_gait_symmetry(symmetry)
    overall = aggregate_score([freq_score, sym_score])

    features = [
        FeatureValue(name="step_frequency", value=round(step_freq, 2), unit="Hz",
                     normal_range=f"see norms for {age_months}mo"),
        FeatureValue(name="left_right_symmetry", value=round(symmetry, 2),
                     normal_range=">=0.92 normal"),
        FeatureValue(name="trunk_forward_lean", value=round(avg_trunk_lean, 1), unit="deg"),
        FeatureValue(name="total_steps", value=float(total_steps)),
    ]

    return GrossMotorResult(
        action="walking",
        score_level=overall,
        features=features,
        summary=f"Step freq: {step_freq:.2f} Hz, symmetry: {symmetry:.2f}, trunk lean: {avg_trunk_lean:.1f}°",
    )


def analyze_ball_throwing(video_path: str, age_months: int) -> GrossMotorResult:
    detector = PoseDetector()
    frames = extract_frames(video_path)

    shoulder_angles = []
    elbow_angles = []
    wrist_velocities = []
    prev_wrist = None
    prev_ts = None

    for _, ts, frame in frames:
        result = detector.process(frame)
        if not result.pose_landmarks:
            continue
        feats = _extract_pose_features_per_frame(result.pose_landmarks)
        shoulder_angles.append(feats["right_shoulder_angle"])
        elbow_angles.append(feats["right_elbow_angle"])

        wrist = landmark_to_array(result.pose_landmarks.landmark[16])
        if prev_wrist is not None and prev_ts is not None:
            dt = ts - prev_ts
            if dt > 0:
                vel = euclidean_distance(wrist, prev_wrist) / dt
                wrist_velocities.append(vel)
        prev_wrist = wrist
        prev_ts = ts

    detector.close()

    if not shoulder_angles:
        return GrossMotorResult(
            action="ball_throwing",
            score_level=ScoreLevel.AT_RISK,
            features=[],
            summary="No pose detected.",
        )

    max_shoulder = float(np.max(shoulder_angles))
    max_elbow_range = float(np.max(elbow_angles) - np.min(elbow_angles))
    peak_wrist_vel = float(np.max(wrist_velocities)) if wrist_velocities else 0

    overhead = max_shoulder > 120
    score = ScoreLevel.NORMAL if overhead and max_elbow_range > 40 else (
        ScoreLevel.BORDERLINE if overhead else ScoreLevel.AT_RISK
    )

    features = [
        FeatureValue(name="max_shoulder_angle", value=round(max_shoulder, 1), unit="deg"),
        FeatureValue(name="elbow_range_of_motion", value=round(max_elbow_range, 1), unit="deg"),
        FeatureValue(name="peak_wrist_velocity", value=round(peak_wrist_vel, 3), unit="units/sec"),
        FeatureValue(name="overhead_motion_detected", value=1.0 if overhead else 0.0),
    ]

    return GrossMotorResult(
        action="ball_throwing",
        score_level=score,
        features=features,
        summary=f"Shoulder ROM: {max_shoulder:.1f}°, elbow ROM: {max_elbow_range:.1f}°, overhead: {overhead}",
    )


def analyze_stair_climbing(video_path: str, age_months: int) -> GrossMotorResult:
    detector = PoseDetector()
    frames = extract_frames(video_path)

    knee_angles_left = []
    knee_angles_right = []
    hip_heights = []
    trunk_leans = []

    for _, ts, frame in frames:
        result = detector.process(frame)
        if not result.pose_landmarks:
            continue
        feats = _extract_pose_features_per_frame(result.pose_landmarks)
        knee_angles_left.append(feats["left_knee_angle"])
        knee_angles_right.append(feats["right_knee_angle"])
        hip_heights.append(feats["hip_mid_y"])
        trunk_leans.append(feats["trunk_lean"])

    detector.close()

    if len(hip_heights) < 5:
        return GrossMotorResult(
            action="stair_climbing",
            score_level=ScoreLevel.AT_RISK,
            features=[],
            summary="Insufficient data for stair climbing analysis.",
        )

    hip_arr = np.array(hip_heights)
    height_change = float(hip_arr[0] - hip_arr[-1])
    left_sym = float(np.mean(knee_angles_left))
    right_sym = float(np.mean(knee_angles_right))
    knee_symmetry = 1.0 - abs(left_sym - right_sym) / (max(left_sym, right_sym) + 1e-8)
    avg_trunk = float(np.mean(trunk_leans))

    score = ScoreLevel.NORMAL if knee_symmetry > 0.85 and avg_trunk < 25 else (
        ScoreLevel.BORDERLINE if knee_symmetry > 0.7 else ScoreLevel.AT_RISK
    )

    features = [
        FeatureValue(name="height_progression", value=round(height_change, 3)),
        FeatureValue(name="knee_angle_symmetry", value=round(knee_symmetry, 2)),
        FeatureValue(name="avg_trunk_lean", value=round(avg_trunk, 1), unit="deg"),
        FeatureValue(name="left_knee_mean", value=round(left_sym, 1), unit="deg"),
        FeatureValue(name="right_knee_mean", value=round(right_sym, 1), unit="deg"),
    ]

    return GrossMotorResult(
        action="stair_climbing",
        score_level=score,
        features=features,
        summary=f"Knee symmetry: {knee_symmetry:.2f}, trunk lean: {avg_trunk:.1f}°",
    )


def _count_peaks(series: np.ndarray, prominence: float = 0.02) -> int:
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(-series, prominence=prominence)
    return len(peaks)
