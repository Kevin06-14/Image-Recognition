"""Shared inference service for the stage 3 web application."""

import base64
import mimetypes
import os
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import requests
from dotenv import load_dotenv
from ultralytics import SAM, YOLO


BASE_DIR = Path(__file__).resolve().parents[1]
YOLO_MODEL_PATH = BASE_DIR / "yolo11n.pt"
SAM_MODEL_PATH = BASE_DIR / "mobile_sam.pt"
OUTPUT_DIR = BASE_DIR / "outputs" / "web"


def _image_data_url(image_path: Path) -> str:
    mime_type = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _ask_vision_model(image_path: Path) -> Optional[str]:
    load_dotenv(BASE_DIR / ".env")
    api_key = os.getenv("LLM_API_KEY", "").strip()
    if not api_key:
        return None

    base_url = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    model_name = os.getenv("LLM_MODEL", "gpt-4o-mini").strip()
    question = os.getenv(
        "VISION_QUESTION",
        "请描述图片中的主要物体，并说明它们的大致位置和外观特征。",
    )
    response = requests.post(
        f"{base_url}/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {
                            "type": "image_url",
                            "image_url": {"url": _image_data_url(image_path)},
                        },
                    ],
                }
            ],
            "temperature": 0.2,
        },
        timeout=120,
    )
    response.raise_for_status()
    return str(response.json()["choices"][0]["message"]["content"])


def analyze_image(image_path: Path) -> dict[str, Any]:
    """Run YOLO, Mobile-SAM, and the optional vision model on one image."""
    if not image_path.exists():
        raise FileNotFoundError(f"找不到图片: {image_path}")
    if not YOLO_MODEL_PATH.exists():
        raise FileNotFoundError(f"找不到 YOLO 权重: {YOLO_MODEL_PATH}")
    if not SAM_MODEL_PATH.exists():
        raise FileNotFoundError(f"找不到 Mobile-SAM 权重: {SAM_MODEL_PATH}")

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"无法读取图片: {image_path}")

    detector = YOLO(str(YOLO_MODEL_PATH))
    detection = detector.predict(source=str(image_path), conf=0.25, verbose=False)[0]
    detections: list[dict[str, Any]] = []
    boxes: list[list[float]] = []
    if detection.boxes is not None:
        boxes = detection.boxes.xyxy.cpu().numpy().tolist()
        confidences = detection.boxes.conf.cpu().numpy()
        class_ids = detection.boxes.cls.cpu().numpy().astype(int)
        for index, box in enumerate(boxes):
            detections.append(
                {
                    "class_name": detector.names[class_ids[index]],
                    "confidence": round(float(confidences[index]), 4),
                    "box": [round(float(value), 1) for value in box],
                }
            )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    overlay_path = OUTPUT_DIR / f"{image_path.stem}_overlay.jpg"
    mask_path = OUTPUT_DIR / f"{image_path.stem}_mask.png"
    overlay = image.copy()
    mask_image = np.zeros(image.shape[:2], dtype=np.uint8)

    if boxes:
        segmenter = SAM(str(SAM_MODEL_PATH))
        segmentation = segmenter.predict(
            source=str(image_path),
            bboxes=boxes,
            verbose=False,
        )[0]
        if segmentation.masks is not None:
            colors = [(40, 180, 255), (80, 220, 120), (220, 120, 80)]
            for index, mask in enumerate(segmentation.masks.data.cpu().numpy()):
                mask = cv2.resize(
                    mask.astype(np.uint8),
                    (image.shape[1], image.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
                color = colors[index % len(colors)]
                color_layer = np.zeros_like(image)
                color_layer[:] = color
                overlay[mask] = cv2.addWeighted(
                    overlay[mask], 0.45, color_layer[mask], 0.55, 0
                )
                mask_image[mask] = 255

    cv2.imwrite(str(overlay_path), overlay)
    cv2.imwrite(str(mask_path), mask_image)

    try:
        vision_answer = _ask_vision_model(image_path)
    except requests.RequestException as exc:
        vision_answer = f"视觉大模型调用失败：{exc}"

    return {
        "image_name": image_path.name,
        "detections": detections,
        "segmentation_overlay": str(overlay_path),
        "segmentation_mask": str(mask_path),
        "vision_answer": vision_answer
        or "未配置视觉大模型 API，已完成 YOLO 与 Mobile-SAM 分析。",
    }
