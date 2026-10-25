from pathlib import Path

import cv2
import numpy as np
from ultralytics import SAM, YOLO


BASE_DIR = Path(__file__).resolve().parents[1]
IMAGE_PATH = BASE_DIR / "data" / "test.jpg"
OUTPUT_DIR = BASE_DIR / "outputs" / "sam"
YOLO_MODEL_PATH = BASE_DIR / "yolo11n.pt"
SAM_MODEL_PATH = BASE_DIR / "mobile_sam.pt"


def segment_detected_objects() -> None:
    """Detect objects with YOLO, then segment each detected box with Mobile-SAM."""
    if not IMAGE_PATH.exists():
        raise FileNotFoundError(f"找不到图片: {IMAGE_PATH}")
    if not SAM_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"找不到 SAM 权重: {SAM_MODEL_PATH}\n"
            "首次运行需要联网下载 mobile_sam.pt；"
            "请先解决网络连接后重新运行本脚本。"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    image = cv2.imread(str(IMAGE_PATH))
    if image is None:
        raise ValueError(f"无法读取图片: {IMAGE_PATH}")

    detector = YOLO(str(YOLO_MODEL_PATH))
    detection_results = detector.predict(source=str(IMAGE_PATH), conf=0.25, verbose=False)
    detection = detection_results[0]

    if detection.boxes is None or len(detection.boxes) == 0:
        print("YOLO 没有检测到目标，无法进行分割")
        return

    boxes = detection.boxes.xyxy.cpu().numpy().tolist()
    names = detector.names
    print(f"YOLO 检测到 {len(boxes)} 个目标，开始进行 SAM 分割")

    segmenter = SAM(str(SAM_MODEL_PATH))
    segmentation_results = segmenter.predict(
        source=str(IMAGE_PATH),
        bboxes=boxes,
        verbose=False,
    )
    segmentation = segmentation_results[0]

    if segmentation.masks is None:
        print("SAM 没有生成分割掩膜")
        return

    overlay = image.copy()
    mask_image = np.zeros(image.shape[:2], dtype=np.uint8)
    colors = [(40, 180, 255), (80, 220, 120), (220, 120, 80), (190, 100, 220)]

    masks = segmentation.masks.data.cpu().numpy()
    confidences = detection.boxes.conf.cpu().numpy()
    class_ids = detection.boxes.cls.cpu().numpy().astype(int)

    for index, mask in enumerate(masks):
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

        x1, y1, x2, y2 = [round(value, 1) for value in boxes[index]]
        class_name = names[class_ids[index]]
        label = f"{class_name} {confidences[index]:.2%}"
        cv2.rectangle(
            overlay,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            color,
            2,
        )
        cv2.putText(
            overlay,
            label,
            (int(x1), max(20, int(y1) - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )
        print(
            f"分割目标 {index + 1}: {class_name} | "
            f"YOLO 置信度: {confidences[index]:.2%} | "
            f"边界框: ({x1}, {y1}, {x2}, {y2})"
        )

    overlay_path = OUTPUT_DIR / "segmentation_overlay.jpg"
    mask_path = OUTPUT_DIR / "segmentation_mask.png"
    cv2.imwrite(str(overlay_path), overlay)
    cv2.imwrite(str(mask_path), mask_image)

    print(f"分割叠加图: {overlay_path}")
    print(f"黑白掩膜图: {mask_path}")


if __name__ == "__main__":
    segment_detected_objects()
