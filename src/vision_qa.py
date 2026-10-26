import base64
import json
import mimetypes
import os
from datetime import datetime
from pathlib import Path

import requests
from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parents[1]
IMAGE_PATH = BASE_DIR / "data" / "test.jpg"
OUTPUT_DIR = BASE_DIR / "outputs" / "vision_qa"


def image_to_data_url(image_path: Path) -> str:
    mime_type = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def ask_vision_model() -> None:
    load_dotenv(BASE_DIR / ".env")
    api_key = os.getenv("LLM_API_KEY", "").strip()
    base_url = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    model_name = os.getenv("LLM_MODEL", "gpt-4o-mini").strip()
    question = os.getenv(
        "VISION_QUESTION",
        "请描述图片中的主要物体，并说明鼠标位于图片的大致位置和外观特征。",
    )

    if not IMAGE_PATH.exists():
        raise FileNotFoundError(f"找不到图片: {IMAGE_PATH}")
    if not api_key:
        raise RuntimeError(
            "没有配置 LLM_API_KEY。请复制 .env.example 为 .env，"
            "再填写视觉模型服务商提供的 API 密钥。"
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
                            "image_url": {"url": image_to_data_url(IMAGE_PATH)},
                        },
                    ],
                }
            ],
            "temperature": 0.2,
        },
        timeout=120,
    )
    response.raise_for_status()
    result = response.json()
    answer = result["choices"][0]["message"]["content"]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().isoformat(timespec="seconds")
    output = {
        "timestamp": timestamp,
        "image": str(IMAGE_PATH),
        "model": model_name,
        "question": question,
        "answer": answer,
    }
    (OUTPUT_DIR / "vision_answer.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (OUTPUT_DIR / "vision_answer.txt").write_text(
        f"图片: {IMAGE_PATH.name}\n模型: {model_name}\n问题: {question}\n\n{answer}\n",
        encoding="utf-8",
    )
    print("视觉大模型问答完成")
    print(f"模型: {model_name}")
    print(f"回答:\n{answer}")
    print(f"结果文件: {OUTPUT_DIR}")


if __name__ == "__main__":
    ask_vision_model()
