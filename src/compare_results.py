import json
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
REPORT_PATH = BASE_DIR / "outputs" / "comparison_report.md"
VISION_RESULT = BASE_DIR / "outputs" / "vision_qa" / "vision_answer.json"
SAM_OVERLAY = BASE_DIR / "outputs" / "sam" / "segmentation_overlay.jpg"
SAM_MASK = BASE_DIR / "outputs" / "sam" / "segmentation_mask.png"


def build_report() -> None:
    lines = [
        "# 同一图片的两种 AI 方案对比",
        "",
        "## 方案一：传统视觉流水线",
        "",
        "- YOLO：输出物体类别、置信度和边界框。",
        "- Mobile-SAM：根据 YOLO 边界框生成像素级分割掩膜。",
        f"- 分割叠加图：`{SAM_OVERLAY}`",
        f"- 黑白掩膜图：`{SAM_MASK}`",
        "",
        "## 方案二：视觉大模型",
        "",
    ]

    if VISION_RESULT.exists():
        result = json.loads(VISION_RESULT.read_text(encoding="utf-8"))
        lines.extend(
            [
                f"- 模型：`{result.get('model', '未记录')}`",
                f"- 提问：{result.get('question', '')}",
                "",
                "### 模型回答",
                "",
                str(result.get("answer", "没有读取到回答")),
            ]
        )
    else:
        lines.extend(
            [
                "视觉大模型结果文件尚未生成。",
                "请先运行 `python src\\vision_qa.py`。",
            ]
        )

    lines.extend(
        [
            "",
            "## 对比结论",
            "",
            "| 维度 | YOLO + Mobile-SAM | 视觉大模型 |",
            "| --- | --- | --- |",
            "| 输出形式 | 类别、置信度、坐标、像素掩膜 | 自然语言描述和问答 |",
            "| 优势 | 结果结构化，适合定位和后续程序处理 | 理解图片语义，表达更灵活 |",
            "| 局限 | 依赖已训练类别，难以解释复杂语义 | 输出可能不稳定，接口需要网络和费用 |",
            "| 适用场景 | 目标检测、分割、自动化流程 | 图片理解、问答、辅助分析 |",
        ]
    )

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"对比报告已生成: {REPORT_PATH}")


if __name__ == "__main__":
    build_report()
