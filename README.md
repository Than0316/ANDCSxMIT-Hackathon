# CDC Milestone Tracker (ANDCS x MIT Hackathon)

AI-assisted pediatric developmental screening backend using video analysis.

This project uses a 3-layer architecture:
1. MediaPipe landmark extraction (Pose / Face Mesh / Hands)
2. Hand-crafted clinical feature extraction
3. Rule-based developmental scoring (no end-to-end model training)

## Features

- Gross motor assessment: `single_leg_stand`, `walking`, `ball_throwing`, `stair_climbing`
- Fine motor assessment: `pincer_grasp`, `block_stacking`, `object_in_container`
- Joint attention: `pointing`, `eye_gaze`, `head_nod_shake`, `imitation` (dual-video)
- Pain/discomfort monitoring: NFCS-style AU threshold scoring
- Real-time WebSocket frame processing
- Clinical-style report generation endpoint

## Tech Stack

- Python 3.10+
- FastAPI + Uvicorn
- MediaPipe
- OpenCV (headless)
- NumPy / SciPy
- Pydantic v2

## Project Structure

```text
api/routes/        # REST + WebSocket endpoints
core/              # scorer, realtime processor, rule engine
models/            # schemas and CDC milestone norms
tasks/             # domain-specific analyzers
utils/             # geometry, DTW, video helpers
static/            # simple frontend page
main.py            # FastAPI app entry point
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Open:
- App UI: `http://localhost:8000/`
- Swagger docs: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`

## REST API

Base path: `http://localhost:8000/api/v1`

### List available tasks

```bash
curl http://localhost:8000/api/v1/tasks
```

### Gross motor assessment

```bash
curl -X POST http://localhost:8000/api/v1/assessment/gross-motor \
	-F "video=@/path/to/video.mp4" \
	-F "child_age_months=24" \
	-F "action=walking"
```

Valid `action` values:
- `single_leg_stand`
- `stair_climbing`
- `ball_throwing`
- `walking`

### Fine motor assessment

```bash
curl -X POST http://localhost:8000/api/v1/assessment/fine-motor \
	-F "video=@/path/to/video.mp4" \
	-F "child_age_months=24" \
	-F "action=pincer_grasp"
```

Valid `action` values:
- `pincer_grasp`
- `block_stacking`
- `object_in_container`

### Joint attention (single video)

```bash
curl -X POST http://localhost:8000/api/v1/assessment/joint-attention \
	-F "video=@/path/to/video.mp4" \
	-F "child_age_months=24" \
	-F "behavior=eye_gaze"
```

Valid `behavior` values:
- `pointing`
- `eye_gaze`
- `head_nod_shake`

### Joint attention imitation (two videos)

```bash
curl -X POST http://localhost:8000/api/v1/assessment/joint-attention/imitation \
	-F "reference_video=@/path/to/reference.mp4" \
	-F "child_video=@/path/to/child.mp4" \
	-F "child_age_months=24"
```

### Pain monitor

```bash
curl -X POST http://localhost:8000/api/v1/assessment/pain-monitor \
	-F "video=@/path/to/video.mp4" \
	-F "child_age_months=24"
```

### Generate report

```bash
curl -X POST http://localhost:8000/api/v1/report/generate \
	-H "Content-Type: application/json" \
	-d '{
		"child_id": "C-001",
		"child_age_months": 24,
		"assessments": []
	}'
```

## WebSocket (Real-time)

Endpoint:

```text
ws://localhost:8000/ws/realtime/{task}
```

Example `task` values: `gross_motor`, `fine_motor`, `joint_attention`, `pain_monitor`.

Message flow:
1. Client connects and sends init JSON, for example: `{"age_months": 24}`
2. Client streams frame messages: `{"type": "frame", "data": "<base64-jpeg>"}`
3. Client requests scoring: `{"type": "assess"}`
4. Client stops stream: `{"type": "stop"}`

## Constraints and Notes

- Supported video formats: `.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`
- Max upload size: `200 MB`
- Default sampling FPS: `15`
- This tool is for screening support, not clinical diagnosis

## License

See `LICENSE`.