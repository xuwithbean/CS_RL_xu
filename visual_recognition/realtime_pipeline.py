# [x]: 实现 yolo 检测推流
"""一键实时识别管道：Windows 推流 + WSL YOLO 检测 + 可选实时预览。

默认流程：
1) 在 Windows 启动 ffmpeg 抓屏并推送到 UDP。
2) 在 WSL 启动 visual_recognition/predict.py 进行实时检测与画框。
3) 可选启动 ffplay 预览带框输出流。

按 Ctrl+C 会自动停止子进程并尝试关闭推流。
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import time
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from get_screenshot import ScreenshotTool


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Realtime CS detection pipeline (Win+WSL)")
    parser.add_argument(
        "--weights",
        type=str,
        default="visual_recognition/runs/ct_t_yolo/weights/best.pt",
        help="YOLO 权重路径",
    )
    parser.add_argument(
        "--in-stream",
        type=str,
        default="udp://192.168.221.36:1234",
        help="输入流地址（Windows 推流到 WSL）",
    )
    parser.add_argument(
        "--out-stream",
        type=str,
        default="udp://127.0.0.1:2234",
        help="带框输出流地址",
    )
    parser.add_argument("--monitor", type=int, default=2, help="Windows 捕获显示器编号（1-based）")
    parser.add_argument("--framerate", type=int, default=60, help="Windows 推流帧率")
    parser.add_argument("--bitrate", type=str, default="2500k", help="Windows 推流码率")
    parser.add_argument("--conf", type=float, default=0.25, help="检测置信度")
    parser.add_argument("--imgsz", type=int, default=640, help="检测输入尺寸")
    parser.add_argument("--device", type=str, default="0", help="检测设备，如 0/cpu")
    parser.add_argument("--project", type=str, default="visual_recognition/runs", help="输出目录")
    parser.add_argument("--name", type=str, default="ct_t_realtime", help="输出实验名")
    parser.add_argument("--show", action="store_true", help="在检测进程内弹窗显示")
    parser.add_argument("--preview", action="store_true", help="使用 ffplay 预览带框输出流")
    parser.add_argument("--skip-win-stream", action="store_true", help="跳过 Windows 推流（当已有输入流时）")
    return parser.parse_args()


def get_run_cmd(cmd: list[str]) -> str:
    return " ".join(shlex.quote(c) for c in cmd)


def main() -> int:
    args = get_args()

    screenshot_tool = None
    preview_proc = None

    try:
        if not args.skip_win_stream:
            screenshot_tool = ScreenshotTool(
                ffmpeg_path="ffmpeg",
                ffplay_path="ffplay",
                framerate=int(args.framerate),
                bitrate=str(args.bitrate),
                ffmpeg_dest=args.in_stream,
                ffplay_source=args.in_stream,
            )
            proc = screenshot_tool.start_stream_monitor(monitor=int(args.monitor), dest=args.in_stream, background=True)
            if proc is None:
                print("Windows 推流启动失败。请检查 ffmpeg 与 monitor 参数。", file=sys.stderr)
                return 1
            print(f"[pipeline] Windows 推流已启动 -> {args.in_stream}")
            # 给输入流一点启动时间，减少检测端首帧超时概率。
            time.sleep(0.8)

        predict_cmd = [
            sys.executable,
            str(ROOT_DIR / "visual_recognition" / "predict.py"),
            "--weights",
            args.weights,
            "--source",
            args.in_stream,
            "--conf",
            str(args.conf),
            "--imgsz",
            str(args.imgsz),
            "--device",
            args.device,
            "--fps",
            str(args.framerate),
            "--save-video",
            "--out-stream",
            args.out_stream,
            "--stream-fps",
            str(args.framerate),
            "--project",
            args.project,
            "--name",
            args.name,
        ]
        if args.show:
            predict_cmd.append("--show")

        print(f"[pipeline] 启动检测: {get_run_cmd(predict_cmd)}")

        if args.preview:
            preview_cmd = [
                "ffplay",
                "-fflags",
                "nobuffer",
                "-flags",
                "low_delay",
                "-framedrop",
                args.out_stream,
            ]
            preview_proc = subprocess.Popen(preview_cmd)
            print(f"[pipeline] 预览已启动 -> {args.out_stream}")

        # 检测进程前台运行，Ctrl+C 可统一退出。
        return subprocess.call(predict_cmd)

    except KeyboardInterrupt:
        print("\n[pipeline] 收到中断，正在清理进程...")
        return 130
    finally:
        if preview_proc is not None:
            try:
                preview_proc.terminate()
            except Exception:
                pass
        if screenshot_tool is not None:
            try:
                screenshot_tool.stop_stream()
                print("[pipeline] Windows 推流已停止")
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
