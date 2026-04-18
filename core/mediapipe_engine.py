"""
Layer 1 — MediaPipe API wrappers.
Direct API calls, no training. Extracts raw landmarks from video frames.
"""

from __future__ import annotations

import mediapipe as mp
import numpy as np

from config import (
    MEDIAPIPE_FACE_CONFIDENCE,
    MEDIAPIPE_HAND_CONFIDENCE,
    MEDIAPIPE_POSE_CONFIDENCE,
)
from utils.video_processor import frame_to_rgb

mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands


class PoseDetector:
    def __init__(self):
        self._pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=MEDIAPIPE_POSE_CONFIDENCE,
            min_tracking_confidence=MEDIAPIPE_POSE_CONFIDENCE,
        )

    def process(self, bgr_frame: np.ndarray):
        rgb = frame_to_rgb(bgr_frame)
        return self._pose.process(rgb)

    def close(self):
        self._pose.close()


class FaceMeshDetector:
    def __init__(self):
        self._mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=2,
            refine_landmarks=True,
            min_detection_confidence=MEDIAPIPE_FACE_CONFIDENCE,
            min_tracking_confidence=MEDIAPIPE_FACE_CONFIDENCE,
        )

    def process(self, bgr_frame: np.ndarray):
        rgb = frame_to_rgb(bgr_frame)
        return self._mesh.process(rgb)

    def close(self):
        self._mesh.close()


class HandDetector:
    def __init__(self):
        self._hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=MEDIAPIPE_HAND_CONFIDENCE,
            min_tracking_confidence=MEDIAPIPE_HAND_CONFIDENCE,
        )

    def process(self, bgr_frame: np.ndarray):
        rgb = frame_to_rgb(bgr_frame)
        return self._hands.process(rgb)

    def close(self):
        self._hands.close()
