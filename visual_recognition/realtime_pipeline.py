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
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from opengame import OpenGameTool


def parse_udp_endpoint(url: str) -> tuple[str, int]:
    """从 udp://ip:port URL 解析出 ip 与端口。"""
    m = re.match(r"^udp://([^:/?#]+):(\d+)", (url or "").strip())
    if not m:
        raise ValueError(f"invalid udp endpoint: {url}")
    return m.group(1), int(m.group(2))


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
    parser.add_argument(
        "--detect-roi",
        type=str,
        default="0.00,0.08,1.00,0.84",
        help="YOLO 识别区域（相对坐标 x,y,w,h），用于排除 HUD 干扰",
    )
    parser.add_argument("--print-yolo", action="store_true", help="实时打印每帧 YOLO 四类中心")
    parser.add_argument("--yolo-info-jsonl", type=str, default="", help="YOLO 中心信息 JSONL 输出路径")
    parser.add_argument("--ocr", action="store_true", help="启用 OCR（血量/护甲/弹药）")
    parser.add_argument(
        "--ocr-engine",
        type=str,
        default="pytesseract",
        choices=["easyocr", "pytesseract"],
        help="OCR 引擎",
    )
    parser.add_argument(
        "--ocr-roi",
        action="append",
        default=[],
        help="OCR 区域（相对坐标 x,y,w,h，可重复）",
    )
    parser.add_argument("--show", action="store_true", help="在检测进程内弹窗显示")
    parser.add_argument("--preview", action="store_true", help="使用 ffplay 预览带框输出流")
    parser.add_argument("--skip-win-stream", action="store_true", help="跳过 Windows 推流（当已有输入流时）")
    parser.add_argument(
        "--game-exe",
        type=str,
        default=r"E:\steam\steamapps\common\Counter-Strike Global Offensive\game\bin\win64\cs2.exe",
        help="Windows CS 可执行路径",
    )
    parser.add_argument("--window-title", type=str, default="auto", help="窗口标题（默认自动匹配）")
    parser.add_argument("--wait-game", type=float, default=6.0, help="启动游戏后等待秒数")
    return parser.parse_args()


def get_run_cmd(cmd: list[str]) -> str:
    return " ".join(shlex.quote(c) for c in cmd)


def main() -> int:
    args = get_args()

    stream_tool = None
    preview_proc = None

    try:
        if not args.skip_win_stream:
            try:
                in_ip, in_port = parse_udp_endpoint(args.in_stream)
            except ValueError as e:
                print(f"输入流地址不合法: {e}", file=sys.stderr)
                return 1

            stream_tool = OpenGameTool(
                game_exe=args.game_exe,
                game_args=["-applaunch", "730"],
                linux_ip=in_ip,
                port=in_port,
                framerate=int(args.framerate),
                bitrate=str(args.bitrate),
                window_title=args.window_title,
                stream_outputs=[args.in_stream],
            )

            stream_tool.open_game(wait_seconds=float(args.wait_game))
            proc = stream_tool.start_stream(with_viewer=False)
            if proc is None:
                print("Windows 推流启动失败。请检查窗口匹配与 ffmpeg。", file=sys.stderr)
                return 1
            print(f"[pipeline] Windows 推流已启动（窗口抓取）-> {args.in_stream}")
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
            "--detect-roi",
            args.detect_roi,
            "--project",
            args.project,
            "--name",
            args.name,
        ]
        if args.print_yolo:
            predict_cmd.append("--print-yolo")
        if args.yolo_info_jsonl:
            predict_cmd.extend(["--yolo-info-jsonl", args.yolo_info_jsonl])
        if args.show:
            predict_cmd.append("--show")
        if args.ocr:
            predict_cmd.extend(["--ocr", "--ocr-engine", args.ocr_engine])
            for roi in args.ocr_roi:
                predict_cmd.extend(["--ocr-roi", roi])

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
        if stream_tool is not None:
            try:
                stream_tool.stop_stream()
                print("[pipeline] Windows 推流已停止")
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
