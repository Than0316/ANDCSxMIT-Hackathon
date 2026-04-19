# Lumi

This is a hackathon project done by Tianyi HAN, Xiaojing YAN, Leo Shan, Ling LING, I Cheong HONG and Xuan DENG.

---

## 🧠 Problem

In France, 1 in 6 children is affected by neurodevelopmental disorders (TND strategy 2023–2027). ADHD diagnosis is delayed by ~33 months on average, and autism is often diagnosed after age 4 despite detectable signs as early as 18 months. This delay contributes to an estimated €28B annual societal cost (PubMed 2024). Between pediatric visits, long observation gaps force parents to rely on fragmented and subjective observations, increasing uncertainty and delaying intervention. Existing screening tools remain largely underused outside clinical settings.

---

## 🚀 Solution

We build a home-based AI system centered on a developmental diary, aligned with natural parental behavior: documenting and cherishing a child’s growth. Rather than positioning it as a medical tool, we provide a simple and warm recording experience that encourages parents to capture daily moments—movements, interactions, and emotions of their child. In the background, multimodal AI analyzes videos, cries, facial expressions, and gestures, translating them into standardized developmental indicators (Denver II, ASQ-3, Bayley). This creates a passive, continuous early-warning layer for neurodevelopmental risks without increasing parental burden.

---

## 🧠 System Architecture

AI-assisted pediatric developmental screening system using multimodal video analysis.

This project uses a 3-layer architecture:
1. MediaPipe landmark extraction (Pose / Face Mesh / Hands)
2. Hand-crafted clinical feature extraction
3. Rule-based developmental scoring (no end-to-end model training)

### Features

- Gross motor assessment: `single_leg_stand`, `walking`, `ball_throwing`, `stair_climbing`
- Fine motor assessment: `pincer_grasp`, `block_stacking`, `object_in_container`
- Joint attention: `pointing`, `eye_gaze`, `head_nod_shake`, `imitation` (dual-video)
- Pain/discomfort monitoring: NFCS-style AU threshold scoring
- Real-time WebSocket frame processing
- Clinical-style report generation endpoint

### Tech Stack

- Python 3.10+
- FastAPI + Uvicorn
- MediaPipe
- OpenCV (headless)
- NumPy / SciPy
- Pydantic v2

### Project Structure

```text
api/routes/        # REST + WebSocket endpoints
core/              # scorer, realtime processor, rule engine
models/            # schemas and CDC milestone norms
tasks/             # domain-specific analyzers
utils/             # geometry, DTW, video helpers
static/            # simple frontend page
main.py            # FastAPI app entry point
```

### Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Open:
- App UI: `http://localhost:8000/`
- Swagger docs: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`

### REST API

Base path: `http://localhost:8000/api/v1`

#### List available tasks

```bash
curl http://localhost:8000/api/v1/tasks
```

#### Gross motor assessment

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

#### Fine motor assessment

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

#### Joint attention (single video)

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

#### Joint attention imitation (two videos)

```bash
curl -X POST http://localhost:8000/api/v1/assessment/joint-attention/imitation \
	-F "reference_video=@/path/to/reference.mp4" \
	-F "child_video=@/path/to/child.mp4" \
	-F "child_age_months=24"
```

#### Pain monitor

```bash
curl -X POST http://localhost:8000/api/v1/assessment/pain-monitor \
	-F "video=@/path/to/video.mp4" \
	-F "child_age_months=24"
```

#### Generate report

```bash
curl -X POST http://localhost:8000/api/v1/report/generate \
	-H "Content-Type: application/json" \
	-d '{
		"child_id": "C-001",
		"child_age_months": 24,
		"assessments": []
	}'
```

### WebSocket (Real-time)

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

### Constraints and Notes

- Supported video formats: `.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`
- Max upload size: `200 MB`
- Default sampling FPS: `15`
- This tool is for screening support, not clinical diagnosis

### License

See `LICENSE`.
