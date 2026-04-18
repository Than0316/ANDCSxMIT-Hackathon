"""Geometric calculation utilities for landmark-based feature extraction."""

from __future__ import annotations

import math
import numpy as np


def landmark_to_array(landmark) -> np.ndarray:
    return np.array([landmark.x, landmark.y, landmark.z])


def angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> float:
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    return float(np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0))))


def joint_angle(p1: np.ndarray, p_joint: np.ndarray, p2: np.ndarray) -> float:
    v1 = p1 - p_joint
    v2 = p2 - p_joint
    return angle_between_vectors(v1, v2)


def euclidean_distance(p1: np.ndarray, p2: np.ndarray) -> float:
    return float(np.linalg.norm(p1 - p2))


def midpoint(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    return (p1 + p2) / 2.0


def center_of_mass(points: list[np.ndarray]) -> np.ndarray:
    return np.mean(points, axis=0)


def angular_velocity(angle_prev: float, angle_curr: float, dt: float) -> float:
    if dt <= 0:
        return 0.0
    return abs(angle_curr - angle_prev) / dt


def direction_vector_2d(p_from: np.ndarray, p_to: np.ndarray) -> np.ndarray:
    v = p_to[:2] - p_from[:2]
    norm = np.linalg.norm(v)
    if norm < 1e-8:
        return np.array([0.0, 0.0])
    return v / norm


def head_pose_from_landmarks(face_landmarks) -> tuple[float, float, float]:
    """
    Estimate head pose (yaw, pitch, roll) from Face Mesh landmarks.
    Uses nose tip, chin, left/right eye corners, left/right mouth corners.
    Key indices: nose_tip=1, chin=152, left_eye_outer=33, right_eye_outer=263,
                 left_mouth=61, right_mouth=291, forehead=10
    """
    nose = landmark_to_array(face_landmarks[1])
    chin = landmark_to_array(face_landmarks[152])
    left_eye = landmark_to_array(face_landmarks[33])
    right_eye = landmark_to_array(face_landmarks[263])
    left_mouth = landmark_to_array(face_landmarks[61])
    right_mouth = landmark_to_array(face_landmarks[291])
    forehead = landmark_to_array(face_landmarks[10])

    eye_center = midpoint(left_eye, right_eye)
    mouth_center = midpoint(left_mouth, right_mouth)

    horizontal = right_eye - left_eye
    yaw = float(np.degrees(np.arctan2(horizontal[2], horizontal[0])))

    vertical = forehead - chin
    pitch = float(np.degrees(np.arctan2(vertical[2], vertical[1])))

    roll = float(np.degrees(np.arctan2(horizontal[1], horizontal[0])))

    return yaw, pitch, roll


def eye_gaze_offset(face_landmarks) -> tuple[float, float]:
    """
    Route A: approximate gaze direction from iris position relative to eye corners.
    Returns (offset_x, offset_y) where positive x = looking right, positive y = looking down.
    Left eye iris indices: 468-472, Right eye iris indices: 473-477 (if iris model enabled)
    Fallback: use pupil approximation from eye landmarks.
    """
    left_inner = landmark_to_array(face_landmarks[133])
    left_outer = landmark_to_array(face_landmarks[33])
    left_upper = landmark_to_array(face_landmarks[159])
    left_lower = landmark_to_array(face_landmarks[145])

    right_inner = landmark_to_array(face_landmarks[362])
    right_outer = landmark_to_array(face_landmarks[263])
    right_upper = landmark_to_array(face_landmarks[386])
    right_lower = landmark_to_array(face_landmarks[374])

    left_center = midpoint(midpoint(left_inner, left_outer), midpoint(left_upper, left_lower))
    right_center = midpoint(midpoint(right_inner, right_outer), midpoint(right_upper, right_lower))

    left_eye_mid = midpoint(left_inner, left_outer)
    right_eye_mid = midpoint(right_inner, right_outer)

    left_width = euclidean_distance(left_inner, left_outer)
    right_width = euclidean_distance(right_inner, right_outer)

    left_offset_x = (left_center[0] - left_eye_mid[0]) / (left_width + 1e-8)
    right_offset_x = (right_center[0] - right_eye_mid[0]) / (right_width + 1e-8)

    left_height = euclidean_distance(left_upper, left_lower)
    right_height = euclidean_distance(right_upper, right_lower)
    left_eye_vmid = midpoint(left_upper, left_lower)
    right_eye_vmid = midpoint(right_upper, right_lower)
    left_offset_y = (left_center[1] - left_eye_vmid[1]) / (left_height + 1e-8)
    right_offset_y = (right_center[1] - right_eye_vmid[1]) / (right_height + 1e-8)

    avg_offset_x = (left_offset_x + right_offset_x) / 2.0
    avg_offset_y = (left_offset_y + right_offset_y) / 2.0

    return float(avg_offset_x), float(avg_offset_y)


def gaze_direction_label(offset_x: float, offset_y: float, threshold: float = 0.15) -> str:
    directions = []
    if offset_y < -threshold:
        directions.append("up")
    elif offset_y > threshold:
        directions.append("down")
    if offset_x < -threshold:
        directions.append("left")
    elif offset_x > threshold:
        directions.append("right")
    return "_".join(directions) if directions else "center"


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    dot = np.dot(v1, v2)
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norm < 1e-8:
        return 0.0
    return float(dot / norm)


def trunk_lean_angle(landmarks) -> float:
    """Calculate trunk forward lean from pose landmarks. Indices: shoulder_mid, hip_mid."""
    left_shoulder = landmark_to_array(landmarks[11])
    right_shoulder = landmark_to_array(landmarks[12])
    left_hip = landmark_to_array(landmarks[23])
    right_hip = landmark_to_array(landmarks[24])

    shoulder_mid = midpoint(left_shoulder, right_shoulder)
    hip_mid = midpoint(left_hip, right_hip)

    trunk_vec = shoulder_mid - hip_mid
    vertical = np.array([0, -1, 0])
    return angle_between_vectors(trunk_vec, vertical)
