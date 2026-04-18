from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

MAX_VIDEO_SIZE_MB = 200
SUPPORTED_VIDEO_FORMATS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

MEDIAPIPE_POSE_CONFIDENCE = 0.5
MEDIAPIPE_FACE_CONFIDENCE = 0.5
MEDIAPIPE_HAND_CONFIDENCE = 0.5

VIDEO_FPS_SAMPLE = 15
