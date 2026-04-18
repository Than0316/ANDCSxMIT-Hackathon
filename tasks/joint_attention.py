"""
Task 2: Joint Attention / Gaze Tracking
Route A: Face Mesh keypoints → hand-calculated head pose + eye offsets → rule engine

Behavioral assessments:
  - Pointing (hand keypoints + direction vector)
  - Eye gaze (gaze estimation via head pose + eye offset)
  - Head nod/shake (pitch/yaw tracking)
  - Imitation (motion matching via DTW/cosine)
"""

from __future__ import annotations

import numpy as np

from core.mediapipe_engine import FaceMeshDetector, HandDetector, PoseDetector
from core.rule_engine import score_joint_attention_latency, aggregate_score
from models.schemas import (
    AttentionBehavior,
    FeatureValue,
    GazeMetrics,
    JointAttentionResult,
    ScoreLevel,
)
from utils.dtw import dtw_similarity, pose_sequence_to_array
from utils.geometry import (
    cosine_similarity,
    direction_vector_2d,
    eye_gaze_offset,
    gaze_direction_label,
    head_pose_from_landmarks,
    landmark_to_array,
)
from utils.video_processor import extract_frames


# ─── Pointing Detection ─────────────────────────────────────────

def analyze_pointing(video_path: str, age_months: int) -> JointAttentionResult:
    """
    Detect pointing gesture using hand keypoints + direction vector.
    Index finger extended while other fingers curled → pointing.
    Direction = vector from wrist to index_tip.
    """
    hand_detector = HandDetector()
    frames = extract_frames(video_path)

    pointing_frames = 0
    total_frames = 0
    directions = []

    for _, ts, frame in frames:
        result = hand_detector.process(frame)
        if not result.multi_hand_landmarks:
            continue
        total_frames += 1

        hand = result.multi_hand_landmarks[0]
        lm = hand.landmark

        index_tip = landmark_to_array(lm[8])
        index_mcp = landmark_to_array(lm[5])
        middle_tip = landmark_to_array(lm[12])
        middle_mcp = landmark_to_array(lm[9])
        ring_tip = landmark_to_array(lm[16])
        ring_mcp = landmark_to_array(lm[13])
        pinky_tip = landmark_to_array(lm[20])
        pinky_mcp = landmark_to_array(lm[17])
        wrist = landmark_to_array(lm[0])

        index_extended = index_tip[1] < index_mcp[1] - 0.02
        middle_curled = middle_tip[1] > middle_mcp[1] - 0.01
        ring_curled = ring_tip[1] > ring_mcp[1] - 0.01
        pinky_curled = pinky_tip[1] > pinky_mcp[1] - 0.01

        if index_extended and middle_curled and ring_curled and pinky_curled:
            pointing_frames += 1
            dir_vec = direction_vector_2d(wrist, index_tip)
            directions.append(dir_vec)

    hand_detector.close()

    if total_frames == 0:
        return JointAttentionResult(
            behavior="pointing",
            score_level=ScoreLevel.AT_RISK,
            features=[],
            summary="No hand detected.",
        )

    pointing_ratio = pointing_frames / total_frames
    avg_direction = np.mean(directions, axis=0) if directions else np.array([0, 0])
    direction_consistency = float(np.min([
        cosine_similarity(d, avg_direction) for d in directions
    ])) if len(directions) > 1 else 0.0

    score = ScoreLevel.NORMAL if pointing_ratio > 0.2 else (
        ScoreLevel.BORDERLINE if pointing_ratio > 0.05 else ScoreLevel.AT_RISK
    )

    features = [
        FeatureValue(name="pointing_detected", value=1.0 if pointing_frames > 0 else 0.0),
        FeatureValue(name="pointing_frame_ratio", value=round(pointing_ratio, 3)),
        FeatureValue(name="direction_consistency", value=round(direction_consistency, 3)),
        FeatureValue(name="avg_direction_x", value=round(float(avg_direction[0]), 3)),
        FeatureValue(name="avg_direction_y", value=round(float(avg_direction[1]), 3)),
    ]

    return JointAttentionResult(
        behavior="pointing",
        score_level=score,
        features=features,
        summary=f"Pointing ratio: {pointing_ratio:.2f}, direction consistency: {direction_consistency:.2f}",
    )


# ─── Eye Gaze (Route A) ─────────────────────────────────────────

def analyze_eye_gaze(video_path: str, age_months: int) -> JointAttentionResult:
    """
    Route A: Face Mesh head pose + eye keypoint relative offset → coarse gaze direction.
    No external gaze model — pure geometry.
    """
    face_detector = FaceMeshDetector()
    frames = extract_frames(video_path)

    gaze_timeline = []
    gaze_directions = []
    first_target_frame = None
    target_reached_frame = None

    for frame_idx, ts, frame in frames:
        result = face_detector.process(frame)
        if not result.multi_face_landmarks:
            continue

        face = result.multi_face_landmarks[0]
        lm = face.landmark

        yaw, pitch, roll = head_pose_from_landmarks(lm)
        offset_x, offset_y = eye_gaze_offset(lm)
        direction = gaze_direction_label(offset_x, offset_y)

        gaze_timeline.append(GazeMetrics(
            gaze_direction=direction,
            head_yaw_deg=round(yaw, 1),
            head_pitch_deg=round(pitch, 1),
            eye_relative_offset_x=round(offset_x, 4),
            eye_relative_offset_y=round(offset_y, 4),
        ))
        gaze_directions.append(direction)

        if first_target_frame is None and direction != "center":
            first_target_frame = ts
        if first_target_frame and target_reached_frame is None and direction != "center":
            target_reached_frame = ts

    face_detector.close()

    if not gaze_timeline:
        return JointAttentionResult(
            behavior="eye_gaze",
            score_level=ScoreLevel.AT_RISK,
            features=[],
            summary="No face detected.",
        )

    unique_dirs = set(gaze_directions)
    center_ratio = gaze_directions.count("center") / len(gaze_directions)
    gaze_shifts = sum(1 for i in range(1, len(gaze_directions)) if gaze_directions[i] != gaze_directions[i - 1])

    latency = None
    if first_target_frame is not None and target_reached_frame is not None:
        latency = target_reached_frame - first_target_frame

    latency_score = score_joint_attention_latency(latency, age_months) if latency is not None else ScoreLevel.BORDERLINE
    shift_score = ScoreLevel.NORMAL if gaze_shifts >= 3 else (
        ScoreLevel.BORDERLINE if gaze_shifts >= 1 else ScoreLevel.AT_RISK
    )
    score = aggregate_score([latency_score, shift_score])

    features = [
        FeatureValue(name="gaze_shift_count", value=float(gaze_shifts)),
        FeatureValue(name="center_gaze_ratio", value=round(center_ratio, 3)),
        FeatureValue(name="unique_directions", value=float(len(unique_dirs))),
        FeatureValue(name="response_latency", value=round(latency, 2) if latency else -1.0, unit="sec"),
    ]

    return JointAttentionResult(
        behavior="eye_gaze",
        score_level=score,
        features=features,
        gaze_timeline=gaze_timeline,
        latency_sec=latency,
        summary=f"Gaze shifts: {gaze_shifts}, center ratio: {center_ratio:.2f}, latency: {latency}s",
    )


# ─── Head Nod / Shake ───────────────────────────────────────────

def analyze_head_nod_shake(video_path: str, age_months: int) -> JointAttentionResult:
    """
    Detect nod (pitch oscillation) and shake (yaw oscillation) from head pose.
    """
    face_detector = FaceMeshDetector()
    frames = extract_frames(video_path)

    yaw_series = []
    pitch_series = []
    timestamps = []

    for _, ts, frame in frames:
        result = face_detector.process(frame)
        if not result.multi_face_landmarks:
            continue
        face = result.multi_face_landmarks[0]
        yaw, pitch, roll = head_pose_from_landmarks(face.landmark)
        yaw_series.append(yaw)
        pitch_series.append(pitch)
        timestamps.append(ts)

    face_detector.close()

    if len(yaw_series) < 5:
        return JointAttentionResult(
            behavior="head_nod_shake",
            score_level=ScoreLevel.AT_RISK,
            features=[],
            summary="Insufficient face data.",
        )

    yaw_arr = np.array(yaw_series)
    pitch_arr = np.array(pitch_series)

    nod_count = _count_oscillations(pitch_arr, threshold=5.0)
    shake_count = _count_oscillations(yaw_arr, threshold=8.0)

    yaw_range = float(np.ptp(yaw_arr))
    pitch_range = float(np.ptp(pitch_arr))

    has_nod = nod_count >= 2
    has_shake = shake_count >= 2
    score = ScoreLevel.NORMAL if (has_nod or has_shake) else (
        ScoreLevel.BORDERLINE if (nod_count >= 1 or shake_count >= 1) else ScoreLevel.AT_RISK
    )

    features = [
        FeatureValue(name="nod_count", value=float(nod_count)),
        FeatureValue(name="shake_count", value=float(shake_count)),
        FeatureValue(name="pitch_range", value=round(pitch_range, 1), unit="deg"),
        FeatureValue(name="yaw_range", value=round(yaw_range, 1), unit="deg"),
    ]

    return JointAttentionResult(
        behavior="head_nod_shake",
        score_level=score,
        features=features,
        summary=f"Nods: {nod_count}, shakes: {shake_count}, pitch range: {pitch_range:.1f}°, yaw range: {yaw_range:.1f}°",
    )


# ─── Imitation (DTW / Cosine) ───────────────────────────────────

def analyze_imitation(
    video_path_reference: str,
    video_path_child: str,
    age_months: int,
) -> JointAttentionResult:
    """
    Compare child's movement to a reference demonstration using DTW and cosine similarity.
    Uses full-body pose landmarks for motion matching.
    """
    pose_detector = PoseDetector()

    ref_poses = _extract_pose_sequence(video_path_reference, pose_detector)
    child_poses = _extract_pose_sequence(video_path_child, pose_detector)
    pose_detector.close()

    if len(ref_poses) < 5 or len(child_poses) < 5:
        return JointAttentionResult(
            behavior="imitation",
            score_level=ScoreLevel.AT_RISK,
            features=[],
            summary="Insufficient pose data for imitation comparison.",
        )

    ref_arr = pose_sequence_to_array(ref_poses)
    child_arr = pose_sequence_to_array(child_poses)

    dtw_sim = dtw_similarity(ref_arr, child_arr)

    ref_mean = np.mean(ref_arr, axis=0)
    child_mean = np.mean(child_arr, axis=0)
    cos_sim = cosine_similarity(ref_mean, child_mean)

    combined = (dtw_sim + cos_sim) / 2.0

    score = ScoreLevel.NORMAL if combined > 0.7 else (
        ScoreLevel.BORDERLINE if combined > 0.4 else ScoreLevel.AT_RISK
    )

    features = [
        FeatureValue(name="dtw_similarity", value=round(dtw_sim, 3)),
        FeatureValue(name="cosine_similarity", value=round(cos_sim, 3)),
        FeatureValue(name="combined_score", value=round(combined, 3)),
        FeatureValue(name="ref_frames", value=float(len(ref_poses))),
        FeatureValue(name="child_frames", value=float(len(child_poses))),
    ]

    return JointAttentionResult(
        behavior="imitation",
        score_level=score,
        features=features,
        summary=f"Imitation match — DTW: {dtw_sim:.2f}, cosine: {cos_sim:.2f}, combined: {combined:.2f}",
    )


# ─── Helpers ─────────────────────────────────────────────────────

def _extract_pose_sequence(video_path: str, detector: PoseDetector) -> list[list[np.ndarray]]:
    frames = extract_frames(video_path)
    sequence = []
    for _, _, frame in frames:
        result = detector.process(frame)
        if result.pose_landmarks:
            lms = [landmark_to_array(lm) for lm in result.pose_landmarks.landmark]
            sequence.append(lms)
    return sequence


def _count_oscillations(series: np.ndarray, threshold: float) -> int:
    """Count direction reversals exceeding threshold amplitude."""
    if len(series) < 3:
        return 0
    diff = np.diff(series)
    sign_changes = 0
    accumulated = 0.0
    prev_sign = np.sign(diff[0]) if diff[0] != 0 else 0

    for d in diff:
        curr_sign = np.sign(d)
        if curr_sign != 0 and curr_sign != prev_sign:
            if abs(accumulated) > threshold:
                sign_changes += 1
            accumulated = 0.0
            prev_sign = curr_sign
        accumulated += d

    return sign_changes
