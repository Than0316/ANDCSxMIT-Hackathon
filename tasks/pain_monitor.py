"""
Task 3: Pain / Discomfort State Monitoring
MediaPipe Face Mesh → geometric AU features (FACS) → threshold rules (no training)

NFCS (Neonatal Facial Coding System) digital implementation:
  AU4  — Brow lowering (眉弓收缩)
  AU6  — Cheek raise (脸颊上提)
  AU7  — Lid tightening (眼睑收紧)
  AU9  — Nose wrinkle (鼻唇沟加深)
  AU10 — Upper lip raise (上唇提升)
  AU20 — Lip stretch (唇拉伸)
  AU25 — Lips part (唇张开)
  AU43 — Eye closure (闭眼)
"""

from __future__ import annotations

import numpy as np

from core.mediapipe_engine import FaceMeshDetector
from core.rule_engine import score_nfcs
from models.schemas import AUActivation, FeatureValue, PainMonitorResult, ScoreLevel
from utils.geometry import euclidean_distance, landmark_to_array, midpoint
from utils.video_processor import extract_frames


# Calibration: run first N frames as neutral baseline
_CALIBRATION_FRAMES = 15


def analyze_pain(video_path: str, age_months: int) -> PainMonitorResult:
    detector = FaceMeshDetector()
    frames = extract_frames(video_path)

    all_au_values: list[dict[str, float]] = []

    for _, ts, frame in frames:
        result = detector.process(frame)
        if not result.multi_face_landmarks:
            continue
        face = result.multi_face_landmarks[0]
        aus = _compute_au_intensities(face.landmark)
        all_au_values.append(aus)

    detector.close()

    if not all_au_values:
        return PainMonitorResult(
            nfcs_score=0.0,
            score_level=ScoreLevel.NORMAL,
            au_activations=[],
            features=[],
            summary="No face detected in video.",
        )

    # Use first frames as neutral baseline for calibration
    cal_end = min(_CALIBRATION_FRAMES, len(all_au_values) // 3, len(all_au_values))
    baseline = {}
    for key in all_au_values[0]:
        baseline[key] = float(np.mean([f[key] for f in all_au_values[:max(1, cal_end)]]))

    # Compute activation relative to baseline for test frames
    test_frames = all_au_values[cal_end:] if cal_end < len(all_au_values) else all_au_values
    if not test_frames:
        test_frames = all_au_values

    avg_activations = {}
    for key in test_frames[0]:
        raw_vals = [f[key] for f in test_frames]
        baseline_val = baseline.get(key, 0)
        activated = [max(0, v - baseline_val) for v in raw_vals]
        avg_activations[key] = float(np.mean(activated))

    # Normalize to 0-1 range
    max_act = max(avg_activations.values()) if avg_activations else 1.0
    if max_act < 1e-6:
        max_act = 1.0

    au_list = []
    for au_code, au_name in _AU_NAMES.items():
        raw = avg_activations.get(au_code, 0.0)
        intensity = min(1.0, raw / max_act)
        au_list.append(AUActivation(
            au_name=au_name,
            au_code=au_code,
            intensity=round(intensity, 3),
        ))

    # NFCS score: sum of binary activations (threshold 0.3)
    nfcs_score = sum(1.0 for au in au_list if au.intensity > 0.3)
    score_level = score_nfcs(nfcs_score)

    features = [
        FeatureValue(name="nfcs_total_score", value=nfcs_score,
                     normal_range="<1.5 normal, 1.5-3 borderline, >=3 pain"),
        FeatureValue(name="active_au_count", value=float(sum(1 for au in au_list if au.intensity > 0.3))),
        FeatureValue(name="max_au_intensity", value=round(max(au.intensity for au in au_list), 3)),
        FeatureValue(name="calibration_frames_used", value=float(cal_end)),
        FeatureValue(name="test_frames_analyzed", value=float(len(test_frames))),
    ]

    pain_aus = [au for au in au_list if au.intensity > 0.3]
    pain_desc = ", ".join(f"{au.au_code}({au.au_name})={au.intensity:.2f}" for au in pain_aus)

    return PainMonitorResult(
        nfcs_score=nfcs_score,
        score_level=score_level,
        au_activations=au_list,
        features=features,
        summary=f"NFCS score: {nfcs_score:.0f}/8. Active AUs: {pain_desc or 'none'}",
    )


_AU_NAMES = {
    "AU4": "Brow Lowering",
    "AU6": "Cheek Raise",
    "AU7": "Lid Tightening",
    "AU9": "Nose Wrinkle",
    "AU10": "Upper Lip Raise",
    "AU20": "Lip Stretch",
    "AU25": "Lips Part",
    "AU43": "Eye Closure",
}


def _compute_au_intensities(landmarks) -> dict[str, float]:
    """Compute geometric AU proxy values from Face Mesh 468 landmarks."""
    lm = landmarks

    # Key landmark positions
    left_brow_inner = landmark_to_array(lm[107])
    right_brow_inner = landmark_to_array(lm[336])
    left_brow_outer = landmark_to_array(lm[70])
    right_brow_outer = landmark_to_array(lm[300])

    left_eye_upper = landmark_to_array(lm[159])
    left_eye_lower = landmark_to_array(lm[145])
    right_eye_upper = landmark_to_array(lm[386])
    right_eye_lower = landmark_to_array(lm[374])

    left_eye_inner = landmark_to_array(lm[133])
    right_eye_inner = landmark_to_array(lm[362])

    nose_tip = landmark_to_array(lm[1])
    nose_wing_left = landmark_to_array(lm[129])
    nose_wing_right = landmark_to_array(lm[358])

    upper_lip_center = landmark_to_array(lm[13])
    lower_lip_center = landmark_to_array(lm[14])
    mouth_left = landmark_to_array(lm[61])
    mouth_right = landmark_to_array(lm[291])
    upper_lip_top = landmark_to_array(lm[0])

    # AU4 — Brow lowering: distance between inner brow points decreases
    brow_inner_dist = euclidean_distance(left_brow_inner, right_brow_inner)
    # Lower value = more activation
    au4 = max(0, 0.08 - brow_inner_dist) * 20

    # AU6 — Cheek raise: lower eyelid pushed up (eye aperture decreases from below)
    left_eye_aperture = euclidean_distance(left_eye_upper, left_eye_lower)
    right_eye_aperture = euclidean_distance(right_eye_upper, right_eye_lower)
    avg_eye_aperture = (left_eye_aperture + right_eye_aperture) / 2.0
    au6 = max(0, 0.04 - avg_eye_aperture) * 30

    # AU7 — Lid tightening: similar to AU6 but tracks upper-lower distance more tightly
    left_inner_corner_to_upper = euclidean_distance(left_eye_inner, left_eye_upper)
    right_inner_corner_to_upper = euclidean_distance(right_eye_inner, right_eye_upper)
    au7 = max(0, 0.035 - (left_inner_corner_to_upper + right_inner_corner_to_upper) / 2) * 35

    # AU9 — Nose wrinkle: nose wing to mouth corner distance decreases
    left_nasolabial = euclidean_distance(nose_wing_left, mouth_left)
    right_nasolabial = euclidean_distance(nose_wing_right, mouth_right)
    avg_nasolabial = (left_nasolabial + right_nasolabial) / 2.0
    au9 = max(0, 0.1 - avg_nasolabial) * 15

    # AU10 — Upper lip raise: distance from nose tip to upper lip center decreases
    nose_to_lip = euclidean_distance(nose_tip, upper_lip_center)
    au10 = max(0, 0.06 - nose_to_lip) * 20

    # AU20 — Lip stretch: mouth width increases
    mouth_width = euclidean_distance(mouth_left, mouth_right)
    au20 = max(0, mouth_width - 0.08) * 12

    # AU25 — Lips part: distance between upper and lower lip centers increases
    lip_separation = euclidean_distance(upper_lip_center, lower_lip_center)
    au25 = max(0, lip_separation - 0.01) * 25

    # AU43 — Eye closure: eye aperture approaches zero
    au43 = max(0, 0.015 - avg_eye_aperture) * 80

    return {
        "AU4": au4,
        "AU6": au6,
        "AU7": au7,
        "AU9": au9,
        "AU10": au10,
        "AU20": au20,
        "AU25": au25,
        "AU43": au43,
    }
