"""
WebSocket endpoint for real-time video stream processing.
Client sends base64-encoded JPEG frames → server returns analysis results.
Supports 'assess' message to trigger developmental assessment scoring.
"""

from __future__ import annotations

import base64
import json

import cv2
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from core.developmental_scorer import DevelopmentalScorer, assessment_to_dict
from core.realtime_processor import RealtimeProcessor

router = APIRouter()


@router.websocket("/ws/realtime/{task}")
async def realtime_stream(websocket: WebSocket, task: str):
    await websocket.accept()

    processor = None
    scorer = None
    try:
        init_msg = await websocket.receive_text()
        config = json.loads(init_msg)
        age_months = config.get("age_months", 24)

        processor = RealtimeProcessor(task=task, age_months=age_months)
        scorer = DevelopmentalScorer(child_age_months=age_months)
        await websocket.send_json({"type": "ready", "task": task})

        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)

            if msg.get("type") == "frame":
                img_b64 = msg["data"]
                img_bytes = base64.b64decode(img_b64)
                nparr = np.frombuffer(img_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if frame is None:
                    await websocket.send_json({"type": "error", "message": "invalid frame"})
                    continue

                result = processor.process_frame(frame)
                result["type"] = "result"
                result["frame_count"] = processor.frame_count

                # Feed observations to scorer
                features = result.get("features", {})
                if task == "gross_motor":
                    scorer.update_from_gross_motor(features)
                elif task == "fine_motor":
                    scorer.update_from_fine_motor(features)
                elif task in ("joint_attention", "eye_gaze", "head_nod_shake", "pointing"):
                    scorer.update_from_attention(features)
                elif task == "pain_monitor":
                    scorer.update_from_pain(result)

                await websocket.send_json(result)

            elif msg.get("type") == "assess":
                assessment = scorer.compute_assessment()
                await websocket.send_json({
                    "type": "assessment",
                    **assessment_to_dict(assessment),
                })

            elif msg.get("type") == "stop":
                break

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
    finally:
        if processor:
            processor.close()
