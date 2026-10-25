from pathlib import Path
from ultralytics import YOLO

IMAGE_PATH = Path("data/test.jpg")
OUTPUT_DIR = Path("outputs/yolo")
MODEL_PATH = "yolo11n.pt"


def detect_image():
    if not IMAGE_PATH.exists():
        raise FileNotFoundError(f"找不到图片: {IMAGE_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model = YOLO(MODEL_PATH)
    results = model.predict(
        source=str(IMAGE_PATH),
        save=True,
        project=str(OUTPUT_DIR.parent),
        name=OUTPUT_DIR.name,
        exist_ok=True,
        conf=0.25,
    )

    print("检测完成")
    for result in results:
        if result.boxes is None or len(result.boxes) == 0:
            print("没有检测到目标")
            continue

        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = [round(float(v), 2) for v in box.xyxy[0]]
            class_name = model.names[class_id]

            print(
                f"类别: {class_name} | "
                f"置信度: {confidence:.2%} | "
                f"坐标: ({x1}, {y1}, {x2}, {y2})"
            )

    print(f"标注结果目录: {OUTPUT_DIR}")


if __name__ == "__main__":
    detect_image()
