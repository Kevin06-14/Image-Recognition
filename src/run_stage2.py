import subprocess
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]


def run_step(title: str, script: str, allow_failure: bool = False) -> bool:
    print(f"\n===== {title} =====")
    completed = subprocess.run(
        [sys.executable, str(BASE_DIR / "src" / script)],
        cwd=BASE_DIR,
    )
    if completed.returncode != 0:
        print(f"{title}未成功完成，退出码: {completed.returncode}")
        if not allow_failure:
            return False
    return True


def main() -> int:
    if not run_step("YOLO 目标检测", "yolo_detect.py"):
        return 1
    if not run_step("YOLO + Mobile-SAM 分割", "sam_segment.py"):
        return 1
    # API 可能因网络或额度暂时不可用，但不应阻断本地视觉结果汇总。
    run_step("视觉大模型问答", "vision_qa.py", allow_failure=True)
    if not run_step("生成两种方案对比报告", "compare_results.py"):
        return 1
    print("\n第二阶段核心流程执行完成。")
    print("请查看 outputs/ 下的检测、分割、问答和对比报告。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
