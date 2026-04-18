"""
Task: Fine Motor Assessment
MediaPipe Hands → geometric features → rule-based scoring

Required assessments:
  - Pincer grasp (thumb + index finger pinching)
  - Block stacking (2-4 blocks)
  - Object placement into container
"""

from __future__ import annotations

import numpy as np

from core.mediapipe_engine import HandDetector
from core.rule_engine import score_block_stacking, score_object_placement, score_pincer_grasp
from models.schemas import FeatureValue, FineMotorResult, ScoreLevel
from utils.geometry import euclidean_distance, landmark_to_array
from utils.video_processor import extract_frames


def _hand_size(hand_landmarks) -> float:
    wrist = landmark_to_array(hand_landmarks.landmark[0])
    middle_tip = landmark_to_array(hand_landmarks.landmark[12])
    return euclidean_distance(wrist, middle_tip)


def analyze_pincer_grasp(video_path: str, age_months: int) -> FineMotorResult:
    """
    Detect thumb-index pincer grasp.
    MediaPipe Hand landmarks: thumb_tip=4, index_tip=8
    Grasp detected when thumb_tip-index_tip distance < threshold * hand_size.
    """
    detector = HandDetector()
    frames = extract_frames(video_path)

    grasp_frames = 0
    total_frames = 0
    min_distances = []
    grasp_durations = []
    current_grasp_len = 0

    for _, ts, frame in frames:
        result = detector.process(frame)
        if not result.multi_hand_landmarks:
            if current_grasp_len > 0:
                grasp_durations.append(current_grasp_len)
                current_grasp_len = 0
            continue

        hand = result.multi_hand_landmarks[0]
        total_frames += 1

        thumb_tip = landmark_to_array(hand.landmark[4])
        index_tip = landmark_to_array(hand.landmark[8])
        dist = euclidean_distance(thumb_tip, index_tip)
        hand_sz = _hand_size(hand)
        normalized_dist = dist / (hand_sz + 1e-8)

        min_distances.append(normalized_dist)

        if normalized_dist < 0.15:
            grasp_frames += 1
            current_grasp_len += 1
        else:
            if current_grasp_len > 0:
                grasp_durations.append(current_grasp_len)
            current_grasp_len = 0

    if current_grasp_len > 0:
        grasp_durations.append(current_grasp_len)

    detector.close()

    if total_frames == 0:
        return FineMotorResult(
            action="pincer_grasp",
            score_level=ScoreLevel.AT_RISK,
            features=[],
            summary="No hand detected in video.",
        )

    grasp_detected = grasp_frames > 3
    avg_dist = float(np.mean(min_distances)) if min_distances else 1.0
    max_grasp_duration = max(grasp_durations) if grasp_durations else 0
    score = score_pincer_grasp(avg_dist, grasp_detected)

    features = [
        FeatureValue(name="grasp_detected", value=1.0 if grasp_detected else 0.0),
        FeatureValue(name="grasp_frame_ratio", value=round(grasp_frames / total_frames, 2)),
        FeatureValue(name="avg_thumb_index_distance", value=round(avg_dist, 4), unit="normalized"),
        FeatureValue(name="max_continuous_grasp_frames", value=float(max_grasp_duration)),
        FeatureValue(name="grasp_episodes", value=float(len(grasp_durations))),
    ]

    return FineMotorResult(
        action="pincer_grasp",
        score_level=score,
        features=features,
        summary=f"Pincer grasp {'detected' if grasp_detected else 'not detected'}, "
                f"avg distance: {avg_dist:.3f}, episodes: {len(grasp_durations)}",
    )


def analyze_block_stacking(video_path: str, age_months: int) -> FineMotorResult:
    """
    Detect block stacking by tracking hand vertical movement patterns.
    Stacking motion: hand moves down (place), up, down (place), up...
    Count the number of downward placement motions and check hand stability.
    """
    detector = HandDetector()
    frames = extract_frames(video_path)

    wrist_y_series = []
    hand_stability = []
    timestamps = []

    for _, ts, frame in frames:
        result = detector.process(frame)
        if not result.multi_hand_landmarks:
            continue
        hand = result.multi_hand_landmarks[0]
        wrist = landmark_to_array(hand.landmark[0])
        wrist_y_series.append(wrist[1])
        timestamps.append(ts)

        fingers_spread = _finger_spread(hand)
        hand_stability.append(fingers_spread)

    detector.close()

    if len(wrist_y_series) < 10:
        return FineMotorResult(
            action="block_stacking",
            score_level=ScoreLevel.AT_RISK,
            features=[],
            summary="Insufficient hand data for block stacking analysis.",
        )

    y_arr = np.array(wrist_y_series)
    placements = _count_placement_motions(y_arr)
    avg_stability = float(np.std(hand_stability))

    score = score_block_stacking(avg_stability, placements)

    features = [
        FeatureValue(name="detected_placements", value=float(placements)),
        FeatureValue(name="hand_stability_std", value=round(avg_stability, 4)),
        FeatureValue(name="vertical_range", value=round(float(np.ptp(y_arr)), 3)),
        FeatureValue(name="total_tracked_frames", value=float(len(wrist_y_series))),
    ]

    return FineMotorResult(
        action="block_stacking",
        score_level=score,
        features=features,
        summary=f"Detected {placements} block placements, hand stability std: {avg_stability:.4f}",
    )


def analyze_object_in_container(video_path: str, age_months: int) -> FineMotorResult:
    """
    Detect object placement into container.
    Pattern: hand moves toward a region, opens (fingers spread), then retreats.
    Successful placement = grasp → move down → release sequence.
    """
    detector = HandDetector()
    frames = extract_frames(video_path)

    states = []
    for _, ts, frame in frames:
        result = detector.process(frame)
        if not result.multi_hand_landmarks:
            states.append("no_hand")
            continue
        hand = result.multi_hand_landmarks[0]
        spread = _finger_spread(hand)
        wrist_y = landmark_to_array(hand.landmark[0])[1]

        if spread < 0.3:
            states.append("closed")
        else:
            states.append("open")

    detector.close()

    attempts, successes = _detect_place_release_sequences(states)

    if not attempts:
        return FineMotorResult(
            action="object_in_container",
            score_level=ScoreLevel.AT_RISK,
            features=[],
            summary="No placement attempts detected.",
        )

    score = score_object_placement(successes, attempts)

    features = [
        FeatureValue(name="attempts", value=float(attempts)),
        FeatureValue(name="successes", value=float(successes)),
        FeatureValue(name="success_rate", value=round(successes / attempts, 2) if attempts > 0 else 0.0),
    ]

    return FineMotorResult(
        action="object_in_container",
        score_level=score,
        features=features,
        summary=f"Object placement: {successes}/{attempts} successful",
    )


def _finger_spread(hand_landmarks) -> float:
    thumb_tip = landmark_to_array(hand_landmarks.landmark[4])
    pinky_tip = landmark_to_array(hand_landmarks.landmark[20])
    hand_sz = _hand_size(hand_landmarks)
    return euclidean_distance(thumb_tip, pinky_tip) / (hand_sz + 1e-8)


def _count_placement_motions(y_series: np.ndarray, min_drop: float = 0.03) -> int:
    """Count down-then-up patterns (block placement motions)."""
    from scipy.signal import find_peaks
    peaks_down, _ = find_peaks(y_series, prominence=min_drop)
    return len(peaks_down)


def _detect_place_release_sequences(states: list[str]) -> tuple[int, int]:
    """
    Detect grasp→place→release pattern: closed → closed → open
    """
    attempts = 0
    successes = 0
    i = 0
    while i < len(states) - 2:
        if states[i] == "closed":
            j = i + 1
            while j < len(states) and states[j] == "closed":
                j += 1
            if j < len(states) and states[j] == "open":
                attempts += 1
                if j - i >= 3:
                    successes += 1
                i = j + 1
                continue
        i += 1
    return attempts, successes
