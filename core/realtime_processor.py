"""
Real-time frame-by-frame processing pipeline.
Maintains per-connection state for temporal analysis (gait cycles, gaze history, etc.)
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from core.mediapipe_engine import FaceMeshDetector, HandDetector, PoseDetector
from utils.geometry import (
    angular_velocity,
    cosine_similarity,
    direction_vector_2d,
    euclidean_distance,
    eye_gaze_offset,
    gaze_direction_label,
    head_pose_from_landmarks,
    joint_angle,
    landmark_to_array,
    midpoint,
    trunk_lean_angle,
)


@dataclass
class TemporalState:
    """Rolling state for temporal feature computation."""
    timestamps: deque = field(default_factory=lambda: deque(maxlen=300))
    # gross motor
    left_ankle_y: deque = field(default_factory=lambda: deque(maxlen=300))
    right_ankle_y: deque = field(default_factory=lambda: deque(maxlen=300))
    left_knee_y: deque = field(default_factory=lambda: deque(maxlen=300))
    right_knee_y: deque = field(default_factory=lambda: deque(maxlen=300))
    com_y: deque = field(default_factory=lambda: deque(maxlen=300))
    single_leg_start: float | None = None
    single_leg_duration: float = 0.0
    max_single_leg_duration: float = 0.0
    step_count: int = 0
    last_step_side: str = ""
    trunk_leans: deque = field(default_factory=lambda: deque(maxlen=60))
    # ground calibration
    ground_y: float | None = None
    ground_calibration_frames: list = field(default_factory=list)
    ground_calibrated: bool = False
    # body scale
    body_height: float = 0.3
    # arm tracking
    left_wrist_vel: deque = field(default_factory=lambda: deque(maxlen=60))
    right_wrist_vel: deque = field(default_factory=lambda: deque(maxlen=60))
    prev_left_wrist: np.ndarray | None = None
    prev_right_wrist: np.ndarray | None = None
    prev_ts: float | None = None
    # squat / jump
    is_squatting: bool = False
    squat_count: int = 0
    is_jumping: bool = False
    jump_count: int = 0
    # fine motor
    thumb_index_dists: deque = field(default_factory=lambda: deque(maxlen=120))
    grasp_active: bool = False
    grasp_count: int = 0
    wrist_y_series: deque = field(default_factory=lambda: deque(maxlen=120))
    placement_count: int = 0
    hand_open_closed: deque = field(default_factory=lambda: deque(maxlen=120))
    release_count: int = 0
    # gaze / attention
    yaw_series: deque = field(default_factory=lambda: deque(maxlen=120))
    pitch_series: deque = field(default_factory=lambda: deque(maxlen=120))
    gaze_dirs: deque = field(default_factory=lambda: deque(maxlen=120))
    nod_count: int = 0
    shake_count: int = 0
    last_yaw_dir: int = 0
    last_pitch_dir: int = 0
    yaw_accum: float = 0.0
    pitch_accum: float = 0.0
    pointing_frames: int = 0
    total_hand_frames: int = 0
    # pain AU
    au_baseline: dict = field(default_factory=dict)
    au_history: deque = field(default_factory=lambda: deque(maxlen=60))
    calibration_done: bool = False
    calibration_frames: list = field(default_factory=list)


class RealtimeProcessor:
    """One instance per WebSocket connection. Call process_frame() per frame."""

    def __init__(self, task: str, age_months: int = 24):
        self.task = task
        self.age_months = age_months
        self.state = TemporalState()
        self.frame_count = 0

        self.pose_detector = None
        self.face_detector = None
        self.hand_detector = None

        if task in ("gross_motor", "walking", "imitation", "motion_monitor"):
            self.pose_detector = PoseDetector()
        if task in ("joint_attention", "eye_gaze", "head_nod_shake", "pain_monitor"):
            self.face_detector = FaceMeshDetector()
        if task in ("fine_motor", "pointing", "motion_monitor"):
            self.hand_detector = HandDetector()
        if task == "pointing":
            self.hand_detector = HandDetector()
        if task == "joint_attention":
            self.hand_detector = HandDetector()
            self.face_detector = FaceMeshDetector()

    def process_frame(self, bgr_frame: np.ndarray) -> dict:
        ts = time.time()
        self.state.timestamps.append(ts)
        self.frame_count += 1

        if self.task == "gross_motor":
            return self._process_gross_motor(bgr_frame, ts)
        elif self.task == "fine_motor":
            return self._process_fine_motor(bgr_frame, ts)
        elif self.task in ("joint_attention", "eye_gaze", "head_nod_shake", "pointing"):
            return self._process_joint_attention(bgr_frame, ts)
        elif self.task == "pain_monitor":
            return self._process_pain(bgr_frame, ts)
        elif self.task == "motion_monitor":
            return self._process_motion_monitor(bgr_frame, ts)
        return {"error": "unknown task"}

    # ─── Gross Motor ────────────────────────────────────────────

    def _process_gross_motor(self, frame, ts) -> dict:
        result = self.pose_detector.process(frame)
        if not result.pose_landmarks:
            return {"detected": False, "landmarks": [], "features": {}}

        lm = result.pose_landmarks.landmark
        landmarks_list = [[l.x, l.y, l.z, l.visibility] for l in lm]

        # Extract key points
        left_shoulder = landmark_to_array(lm[11])
        right_shoulder = landmark_to_array(lm[12])
        left_elbow = landmark_to_array(lm[13])
        right_elbow = landmark_to_array(lm[14])
        left_wrist = landmark_to_array(lm[15])
        right_wrist = landmark_to_array(lm[16])
        left_hip = landmark_to_array(lm[23])
        right_hip = landmark_to_array(lm[24])
        left_knee = landmark_to_array(lm[25])
        right_knee = landmark_to_array(lm[26])
        left_ankle = landmark_to_array(lm[27])
        right_ankle = landmark_to_array(lm[28])

        hip_mid = midpoint(left_hip, right_hip)
        shoulder_mid = midpoint(left_shoulder, right_shoulder)

        # Body scale: distance from shoulder to ankle (used for adaptive thresholds)
        body_h = euclidean_distance(shoulder_mid, midpoint(left_ankle, right_ankle))
        if body_h > 0.05:
            self.state.body_height = body_h
        lift_threshold = self.state.body_height * 0.04

        # Ground calibration: use first 15 frames
        if not self.state.ground_calibrated:
            floor_y = max(left_ankle[1], right_ankle[1])
            self.state.ground_calibration_frames.append(floor_y)
            if len(self.state.ground_calibration_frames) >= 15:
                self.state.ground_y = float(np.median(self.state.ground_calibration_frames))
                self.state.ground_calibrated = True
            return {
                "detected": True,
                "landmarks": landmarks_list,
                "calibrating": True,
                "calibration_progress": len(self.state.ground_calibration_frames) / 15.0,
                "features": {},
                "score": "normal",
            }

        # ── Joint angles ──
        left_knee_angle = joint_angle(left_hip, left_knee, left_ankle)
        right_knee_angle = joint_angle(right_hip, right_knee, right_ankle)
        left_hip_angle = joint_angle(left_shoulder, left_hip, left_knee)
        right_hip_angle = joint_angle(right_shoulder, right_hip, right_knee)
        left_elbow_angle = joint_angle(left_shoulder, left_elbow, left_wrist)
        right_elbow_angle = joint_angle(right_shoulder, right_elbow, right_wrist)
        trunk_lean = trunk_lean_angle(lm)

        # ── Track history ──
        self.state.left_ankle_y.append(left_ankle[1])
        self.state.right_ankle_y.append(right_ankle[1])
        self.state.left_knee_y.append(left_knee[1])
        self.state.right_knee_y.append(right_knee[1])
        com_y = (hip_mid[1] + shoulder_mid[1]) / 2.0
        self.state.com_y.append(com_y)
        self.state.trunk_leans.append(trunk_lean)

        # ── Foot lift detection (relative to calibrated ground) ──
        left_lift_from_ground = self.state.ground_y - left_ankle[1]
        right_lift_from_ground = self.state.ground_y - right_ankle[1]
        left_lifted = left_lift_from_ground > lift_threshold
        right_lifted = right_lift_from_ground > lift_threshold

        # Also check relative difference between two feet
        ankle_diff = abs(left_ankle[1] - right_ankle[1])
        left_higher = left_ankle[1] < right_ankle[1] - lift_threshold * 0.5
        right_higher = right_ankle[1] < left_ankle[1] - lift_threshold * 0.5

        standing_on_one = (left_lifted and not right_lifted) or (right_lifted and not left_lifted) or left_higher or right_higher
        support_side = ""
        if standing_on_one:
            support_side = "right" if (left_lifted or left_higher) else "left"

        # Single-leg duration tracking
        if standing_on_one:
            if self.state.single_leg_start is None:
                self.state.single_leg_start = ts
            self.state.single_leg_duration = ts - self.state.single_leg_start
            self.state.max_single_leg_duration = max(
                self.state.max_single_leg_duration, self.state.single_leg_duration
            )
        else:
            self.state.single_leg_start = None
            self.state.single_leg_duration = 0.0

        # ── Squat detection (both knees bent below ~120°) ──
        avg_knee_angle = (left_knee_angle + right_knee_angle) / 2
        is_squatting = avg_knee_angle < 130
        if is_squatting and not self.state.is_squatting:
            self.state.squat_count += 1
        self.state.is_squatting = is_squatting

        # ── Jump detection (both feet above ground) ──
        both_lifted = left_lift_from_ground > lift_threshold and right_lift_from_ground > lift_threshold
        if both_lifted and not self.state.is_jumping:
            self.state.jump_count += 1
        self.state.is_jumping = both_lifted

        # ── Arm velocity tracking ──
        dt = ts - self.state.prev_ts if self.state.prev_ts else 0.1
        if dt > 0 and self.state.prev_left_wrist is not None:
            lv = euclidean_distance(left_wrist, self.state.prev_left_wrist) / dt
            rv = euclidean_distance(right_wrist, self.state.prev_right_wrist) / dt
            self.state.left_wrist_vel.append(lv)
            self.state.right_wrist_vel.append(rv)
        self.state.prev_left_wrist = left_wrist.copy()
        self.state.prev_right_wrist = right_wrist.copy()
        self.state.prev_ts = ts

        # Arm raise detection
        left_arm_raised = left_wrist[1] < left_shoulder[1]
        right_arm_raised = right_wrist[1] < right_shoulder[1]
        arms_state = ""
        if left_arm_raised and right_arm_raised:
            arms_state = "both_raised"
        elif left_arm_raised:
            arms_state = "left_raised"
        elif right_arm_raised:
            arms_state = "right_raised"

        # ── Step counting (improved) ──
        if len(self.state.left_ankle_y) > 3:
            la = list(self.state.left_ankle_y)
            ra = list(self.state.right_ankle_y)
            # Detect when a foot goes up then comes back down
            left_going_up = la[-1] < la[-2] - lift_threshold * 0.3
            right_going_up = ra[-1] < ra[-2] - lift_threshold * 0.3
            if left_going_up and self.state.last_step_side != "left":
                self.state.step_count += 1
                self.state.last_step_side = "left"
            elif right_going_up and self.state.last_step_side != "right":
                self.state.step_count += 1
                self.state.last_step_side = "right"

        # ── Aggregate metrics ──
        com_std = float(np.std(list(self.state.com_y)[-30:])) if len(self.state.com_y) > 5 else 0.0
        duration = ts - self.state.timestamps[0] if len(self.state.timestamps) > 1 else 1.0
        step_freq = self.state.step_count / max(duration, 0.1)
        avg_trunk = float(np.mean(list(self.state.trunk_leans)[-30:])) if self.state.trunk_leans else 0.0

        # Current movement description
        movement = "standing"
        if both_lifted:
            movement = "jumping"
        elif standing_on_one:
            movement = f"one_leg ({support_side} support)"
        elif is_squatting:
            movement = "squatting"
        elif self.state.step_count > 0 and step_freq > 0.3:
            movement = "walking"

        left_wrist_speed = float(np.mean(list(self.state.left_wrist_vel)[-5:])) if self.state.left_wrist_vel else 0.0
        right_wrist_speed = float(np.mean(list(self.state.right_wrist_vel)[-5:])) if self.state.right_wrist_vel else 0.0

        return {
            "detected": True,
            "landmarks": landmarks_list,
            "features": {
                "movement": movement,
                "left_knee_angle": round(left_knee_angle, 1),
                "right_knee_angle": round(right_knee_angle, 1),
                "left_hip_angle": round(left_hip_angle, 1),
                "right_hip_angle": round(right_hip_angle, 1),
                "left_elbow_angle": round(left_elbow_angle, 1),
                "right_elbow_angle": round(right_elbow_angle, 1),
                "trunk_lean": round(trunk_lean, 1),
                "avg_trunk_lean": round(avg_trunk, 1),
                "single_leg_standing": standing_on_one,
                "support_side": support_side,
                "single_leg_duration": round(self.state.single_leg_duration, 1),
                "max_single_leg_duration": round(self.state.max_single_leg_duration, 1),
                "left_foot_lift": round(left_lift_from_ground, 3),
                "right_foot_lift": round(right_lift_from_ground, 3),
                "is_squatting": is_squatting,
                "squat_count": self.state.squat_count,
                "is_jumping": both_lifted,
                "jump_count": self.state.jump_count,
                "arms_state": arms_state,
                "left_wrist_speed": round(left_wrist_speed, 3),
                "right_wrist_speed": round(right_wrist_speed, 3),
                "step_count": self.state.step_count,
                "step_frequency_hz": round(step_freq, 2),
                "com_stability": round(com_std, 4),
                "ground_y": round(self.state.ground_y, 3),
            },
            "score": self._score_gross_motor(movement),
        }

    def _score_gross_motor(self, movement):
        if movement in ("jumping", "walking") or "one_leg" in movement:
            return "normal"
        return "borderline"

    # ─── Fine Motor ─────────────────────────────────────────────

    def _process_fine_motor(self, frame, ts) -> dict:
        result = self.hand_detector.process(frame)
        if not result.multi_hand_landmarks:
            return {"detected": False, "landmarks": []}

        hand = result.multi_hand_landmarks[0]
        lm = hand.landmark
        landmarks_list = [[l.x, l.y, l.z] for l in lm]

        thumb_tip = landmark_to_array(lm[4])
        index_tip = landmark_to_array(lm[8])
        wrist = landmark_to_array(lm[0])
        middle_tip = landmark_to_array(lm[12])
        pinky_tip = landmark_to_array(lm[20])

        hand_size = euclidean_distance(wrist, middle_tip)
        thumb_index_dist = euclidean_distance(thumb_tip, index_tip)
        normalized_dist = thumb_index_dist / (hand_size + 1e-8)

        self.state.thumb_index_dists.append(normalized_dist)
        self.state.wrist_y_series.append(wrist[1])

        # Pincer grasp detection
        is_pinching = normalized_dist < 0.15
        if is_pinching and not self.state.grasp_active:
            self.state.grasp_count += 1
        self.state.grasp_active = is_pinching

        # Finger spread (open/closed for container placement)
        spread = euclidean_distance(thumb_tip, pinky_tip) / (hand_size + 1e-8)
        is_open = spread > 0.6
        self.state.hand_open_closed.append("open" if is_open else "closed")

        # Detect release (closed → open)
        if len(self.state.hand_open_closed) >= 3:
            seq = list(self.state.hand_open_closed)
            if seq[-1] == "open" and seq[-2] == "closed" and seq[-3] == "closed":
                self.state.release_count += 1

        # Block stacking: detect downward placement
        if len(self.state.wrist_y_series) >= 5:
            wy = list(self.state.wrist_y_series)
            if wy[-1] > wy[-3] > wy[-5] and wy[-1] - wy[-5] > 0.03:
                self.state.placement_count += 1

        return {
            "detected": True,
            "landmarks": landmarks_list,
            "features": {
                "thumb_index_distance": round(normalized_dist, 3),
                "is_pinching": is_pinching,
                "grasp_count": self.state.grasp_count,
                "finger_spread": round(spread, 3),
                "hand_open": is_open,
                "release_count": self.state.release_count,
                "placement_count": self.state.placement_count,
            },
            "score": "normal" if self.state.grasp_count > 0 else "borderline",
        }

    # ─── Joint Attention ────────────────────────────────────────

    def _process_joint_attention(self, frame, ts) -> dict:
        data = {"detected": False, "face_landmarks": [], "hand_landmarks": [], "features": {}}

        # Face processing
        if self.face_detector:
            face_result = self.face_detector.process(frame)
            if face_result.multi_face_landmarks:
                face = face_result.multi_face_landmarks[0]
                face_lm = face.landmark
                data["face_landmarks"] = [[l.x, l.y, l.z] for l in face_lm]
                data["detected"] = True

                yaw, pitch, roll = head_pose_from_landmarks(face_lm)
                offset_x, offset_y = eye_gaze_offset(face_lm)
                gaze_dir = gaze_direction_label(offset_x, offset_y)

                self.state.yaw_series.append(yaw)
                self.state.pitch_series.append(pitch)
                self.state.gaze_dirs.append(gaze_dir)

                # Nod/shake counting
                if len(self.state.pitch_series) >= 3:
                    dp = self.state.pitch_series[-1] - self.state.pitch_series[-2]
                    curr_dir = 1 if dp > 0 else (-1 if dp < 0 else 0)
                    if curr_dir != 0 and curr_dir != self.state.last_pitch_dir:
                        self.state.pitch_accum = abs(dp)
                        if self.state.pitch_accum > 5:
                            self.state.nod_count += 1
                    else:
                        self.state.pitch_accum += abs(dp)
                    self.state.last_pitch_dir = curr_dir

                if len(self.state.yaw_series) >= 3:
                    dy = self.state.yaw_series[-1] - self.state.yaw_series[-2]
                    curr_dir = 1 if dy > 0 else (-1 if dy < 0 else 0)
                    if curr_dir != 0 and curr_dir != self.state.last_yaw_dir:
                        self.state.yaw_accum = abs(dy)
                        if self.state.yaw_accum > 8:
                            self.state.shake_count += 1
                    else:
                        self.state.yaw_accum += abs(dy)
                    self.state.last_yaw_dir = curr_dir

                # Gaze shift count
                gaze_shifts = 0
                gd = list(self.state.gaze_dirs)
                for i in range(1, len(gd)):
                    if gd[i] != gd[i - 1]:
                        gaze_shifts += 1

                data["features"].update({
                    "head_yaw": round(yaw, 1),
                    "head_pitch": round(pitch, 1),
                    "head_roll": round(roll, 1),
                    "gaze_direction": gaze_dir,
                    "eye_offset_x": round(offset_x, 3),
                    "eye_offset_y": round(offset_y, 3),
                    "nod_count": self.state.nod_count,
                    "shake_count": self.state.shake_count,
                    "gaze_shift_count": gaze_shifts,
                })

        # Hand processing (pointing detection)
        if self.hand_detector:
            hand_result = self.hand_detector.process(frame)
            if hand_result.multi_hand_landmarks:
                hand = hand_result.multi_hand_landmarks[0]
                hlm = hand.landmark
                data["hand_landmarks"] = [[l.x, l.y, l.z] for l in hlm]
                data["detected"] = True
                self.state.total_hand_frames += 1

                index_tip = landmark_to_array(hlm[8])
                index_mcp = landmark_to_array(hlm[5])
                middle_tip = landmark_to_array(hlm[12])
                middle_mcp = landmark_to_array(hlm[9])
                ring_tip = landmark_to_array(hlm[16])
                ring_mcp = landmark_to_array(hlm[13])
                pinky_tip = landmark_to_array(hlm[20])
                pinky_mcp = landmark_to_array(hlm[17])
                wrist = landmark_to_array(hlm[0])

                index_ext = index_tip[1] < index_mcp[1] - 0.02
                others_curled = (
                    middle_tip[1] > middle_mcp[1] - 0.01
                    and ring_tip[1] > ring_mcp[1] - 0.01
                    and pinky_tip[1] > pinky_mcp[1] - 0.01
                )
                is_pointing = index_ext and others_curled

                if is_pointing:
                    self.state.pointing_frames += 1
                    dir_vec = direction_vector_2d(wrist, index_tip)
                    data["features"]["pointing_direction_x"] = round(float(dir_vec[0]), 3)
                    data["features"]["pointing_direction_y"] = round(float(dir_vec[1]), 3)

                data["features"]["is_pointing"] = is_pointing
                ratio = self.state.pointing_frames / max(self.state.total_hand_frames, 1)
                data["features"]["pointing_ratio"] = round(ratio, 3)

        data["score"] = self._score_attention()
        return data

    def _score_attention(self):
        gd = list(self.state.gaze_dirs)
        shifts = sum(1 for i in range(1, len(gd)) if gd[i] != gd[i - 1])
        if shifts >= 3 or self.state.nod_count >= 2 or self.state.pointing_frames > 5:
            return "normal"
        if shifts >= 1 or self.state.nod_count >= 1:
            return "borderline"
        return "at_risk"

    # ─── Pain Monitor ───────────────────────────────────────────

    def _process_pain(self, frame, ts) -> dict:
        result = self.face_detector.process(frame)
        if not result.multi_face_landmarks:
            return {"detected": False, "landmarks": [], "au_activations": {}}

        face = result.multi_face_landmarks[0]
        lm = face.landmark
        landmarks_list = [[l.x, l.y, l.z] for l in lm]

        aus = self._compute_aus(lm)

        # Calibration: first 30 frames as baseline
        if not self.state.calibration_done:
            self.state.calibration_frames.append(aus)
            if len(self.state.calibration_frames) >= 30:
                baseline = {}
                for key in aus:
                    baseline[key] = float(np.mean([f[key] for f in self.state.calibration_frames]))
                self.state.au_baseline = baseline
                self.state.calibration_done = True
            return {
                "detected": True,
                "landmarks": landmarks_list,
                "calibrating": True,
                "calibration_progress": len(self.state.calibration_frames) / 30.0,
                "au_activations": {},
                "nfcs_score": 0,
                "score": "normal",
            }

        # Compute activation relative to baseline
        activations = {}
        for key, val in aus.items():
            baseline_val = self.state.au_baseline.get(key, 0)
            activations[key] = max(0, val - baseline_val)

        # Normalize
        max_act = max(activations.values()) if activations else 1.0
        if max_act < 1e-6:
            max_act = 1.0
        normalized = {k: min(1.0, v / max_act) for k, v in activations.items()}

        self.state.au_history.append(normalized)

        # Smoothed AU (average over last 10 frames)
        smoothed = {}
        hist = list(self.state.au_history)
        for key in normalized:
            smoothed[key] = round(float(np.mean([h[key] for h in hist[-10:]])), 3)

        nfcs_score = sum(1.0 for v in smoothed.values() if v > 0.3)
        score = "at_risk" if nfcs_score >= 3 else ("borderline" if nfcs_score >= 1.5 else "normal")

        return {
            "detected": True,
            "landmarks": landmarks_list,
            "calibrating": False,
            "au_activations": smoothed,
            "nfcs_score": round(nfcs_score, 1),
            "score": score,
        }

    def _compute_aus(self, lm) -> dict:
        left_brow_inner = landmark_to_array(lm[107])
        right_brow_inner = landmark_to_array(lm[336])

        left_eye_upper = landmark_to_array(lm[159])
        left_eye_lower = landmark_to_array(lm[145])
        right_eye_upper = landmark_to_array(lm[386])
        right_eye_lower = landmark_to_array(lm[374])

        left_eye_inner = landmark_to_array(lm[133])
        right_eye_inner = landmark_to_array(lm[362])

        nose_tip = landmark_to_array(lm[1])
        nose_wing_left = landmark_to_array(lm[129])
        nose_wing_right = landmark_to_array(lm[358])

        upper_lip = landmark_to_array(lm[13])
        lower_lip = landmark_to_array(lm[14])
        mouth_left = landmark_to_array(lm[61])
        mouth_right = landmark_to_array(lm[291])

        brow_dist = euclidean_distance(left_brow_inner, right_brow_inner)
        au4 = max(0, 0.08 - brow_dist) * 20

        left_eye_h = euclidean_distance(left_eye_upper, left_eye_lower)
        right_eye_h = euclidean_distance(right_eye_upper, right_eye_lower)
        avg_eye = (left_eye_h + right_eye_h) / 2.0
        au6 = max(0, 0.04 - avg_eye) * 30
        au7 = max(0, 0.035 - (
            euclidean_distance(left_eye_inner, left_eye_upper)
            + euclidean_distance(right_eye_inner, right_eye_upper)
        ) / 2) * 35

        nasolabial = (
            euclidean_distance(nose_wing_left, mouth_left)
            + euclidean_distance(nose_wing_right, mouth_right)
        ) / 2
        au9 = max(0, 0.1 - nasolabial) * 15
        au10 = max(0, 0.06 - euclidean_distance(nose_tip, upper_lip)) * 20
        au20 = max(0, euclidean_distance(mouth_left, mouth_right) - 0.08) * 12
        au25 = max(0, euclidean_distance(upper_lip, lower_lip) - 0.01) * 25
        au43 = max(0, 0.015 - avg_eye) * 80

        return {"AU4": au4, "AU6": au6, "AU7": au7, "AU9": au9,
                "AU10": au10, "AU20": au20, "AU25": au25, "AU43": au43}

    # ─── Motion Monitor ─────────────────────────────────────────

    def _process_motion_monitor(self, frame, ts) -> dict:
        """Full-body motion monitor: pose + hands + face landmarks with joint angles."""
        data: dict = {
            "detected": False,
            "pose_landmarks": [],
            "hand_landmarks": [],
            "face_detected": False,
            "features": {},
        }

        # ── Pose (body skeleton) ──
        if self.pose_detector:
            pose_result = self.pose_detector.process(frame)
            if pose_result.pose_landmarks:
                lm = pose_result.pose_landmarks.landmark
                data["pose_landmarks"] = [[l.x, l.y, l.z, l.visibility] for l in lm]
                data["detected"] = True

                # Key points
                ls = landmark_to_array(lm[11])
                rs = landmark_to_array(lm[12])
                le = landmark_to_array(lm[13])
                re = landmark_to_array(lm[14])
                lw = landmark_to_array(lm[15])
                rw = landmark_to_array(lm[16])
                lh = landmark_to_array(lm[23])
                rh = landmark_to_array(lm[24])
                lk = landmark_to_array(lm[25])
                rk = landmark_to_array(lm[26])
                la = landmark_to_array(lm[27])
                ra = landmark_to_array(lm[28])
                nose = landmark_to_array(lm[0])

                # Joint angles
                l_elbow_ang = joint_angle(ls, le, lw)
                r_elbow_ang = joint_angle(rs, re, rw)
                l_knee_ang = joint_angle(lh, lk, la)
                r_knee_ang = joint_angle(rh, rk, ra)
                l_shoulder_ang = joint_angle(lh, ls, le)
                r_shoulder_ang = joint_angle(rh, rs, re)
                l_hip_ang = joint_angle(ls, lh, lk)
                r_hip_ang = joint_angle(rs, rh, rk)
                trunk = trunk_lean_angle(lm)

                # Visibility of key body parts (averaged L/R)
                vis_arms = round((lm[15].visibility + lm[16].visibility) / 2, 2)
                vis_legs = round((lm[27].visibility + lm[28].visibility) / 2, 2)
                vis_torso = round((lm[23].visibility + lm[24].visibility) / 2, 2)

                # Wrist velocity (motion intensity)
                motion_intensity = 0.0
                if self.state.prev_left_wrist is not None and self.state.prev_ts is not None:
                    dt = max(ts - self.state.prev_ts, 1e-6)
                    lv = euclidean_distance(lw, self.state.prev_left_wrist) / dt
                    rv = euclidean_distance(rw, self.state.prev_right_wrist) / dt
                    motion_intensity = round((lv + rv) / 2, 3)

                self.state.prev_left_wrist = lw
                self.state.prev_right_wrist = rw
                self.state.prev_ts = ts

                # Body height (nose to ankle midpoint)
                ankle_mid = midpoint(la, ra)
                body_h = euclidean_distance(nose, ankle_mid)

                data["features"].update({
                    "left_elbow_angle": round(l_elbow_ang, 1),
                    "right_elbow_angle": round(r_elbow_ang, 1),
                    "left_knee_angle": round(l_knee_ang, 1),
                    "right_knee_angle": round(r_knee_ang, 1),
                    "left_shoulder_angle": round(l_shoulder_ang, 1),
                    "right_shoulder_angle": round(r_shoulder_ang, 1),
                    "left_hip_angle": round(l_hip_ang, 1),
                    "right_hip_angle": round(r_hip_ang, 1),
                    "trunk_lean": round(trunk, 1),
                    "motion_intensity": motion_intensity,
                    "visibility_arms": vis_arms,
                    "visibility_legs": vis_legs,
                    "visibility_torso": vis_torso,
                    "body_height_norm": round(body_h, 3),
                })

        # ── Hands ──
        if self.hand_detector:
            hand_result = self.hand_detector.process(frame)
            if hand_result.multi_hand_landmarks:
                hands = []
                for hl in hand_result.multi_hand_landmarks[:2]:
                    hands.append([[l.x, l.y, l.z] for l in hl.landmark])
                data["hand_landmarks"] = hands
                data["detected"] = True
                data["features"]["hands_detected"] = len(hands)
            else:
                data["features"]["hands_detected"] = 0

        return data

    def close(self):
        if self.pose_detector:
            self.pose_detector.close()
        if self.face_detector:
            self.face_detector.close()
        if self.hand_detector:
            self.hand_detector.close()
