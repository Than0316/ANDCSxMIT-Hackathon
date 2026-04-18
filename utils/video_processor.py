"""Video I/O: frame extraction at target FPS."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from config import VIDEO_FPS_SAMPLE


def extract_frames(video_path: str | Path, target_fps: int = VIDEO_FPS_SAMPLE) -> list[tuple[int, float, np.ndarray]]:
    """
    Extract frames from video at target FPS.
    Returns list of (frame_index, timestamp_sec, bgr_frame).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_interval = max(1, int(round(src_fps / target_fps)))

    frames = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % frame_interval == 0:
            timestamp = idx / src_fps
            frames.append((idx, timestamp, frame))
        idx += 1

    cap.release()
    return frames


def frame_to_rgb(bgr_frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
