"""Dynamic Time Warping for motion imitation matching."""

from __future__ import annotations

import numpy as np


def dtw_distance(seq1: np.ndarray, seq2: np.ndarray) -> float:
    n, m = len(seq1), len(seq2)
    cost = np.full((n + 1, m + 1), np.inf)
    cost[0, 0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d = np.linalg.norm(seq1[i - 1] - seq2[j - 1])
            cost[i, j] = d + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])

    return float(cost[n, m]) / max(n, m)


def dtw_similarity(seq1: np.ndarray, seq2: np.ndarray) -> float:
    dist = dtw_distance(seq1, seq2)
    return float(1.0 / (1.0 + dist))


def pose_sequence_to_array(pose_sequence: list[list[np.ndarray]]) -> np.ndarray:
    """Convert a list of frame landmarks to a 2D array (frames x flattened_coords)."""
    result = []
    for frame_landmarks in pose_sequence:
        flat = np.concatenate([lm.flatten() for lm in frame_landmarks])
        result.append(flat)
    return np.array(result)
