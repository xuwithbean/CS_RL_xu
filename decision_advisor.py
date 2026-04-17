"""实时分析程序：实时输出敌情建议，并通过终端命令触发 OCR/位置分析。"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from visual_recognition.stream_ffplay_pipeline import (
    get_build_image_data_url_from_frame,
    get_frame_ocr_interface,
    get_parse_roi,
    get_resolve_qwen_api_key,
    get_qwen_location_client,
)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Realtime decision advisor (command driven)")
    parser.add_argument("--source", type=str, default="", help="兼容参数：已不再由 advisor 直接拉流")
    parser.add_argument("--weights", type=str, default="", help="兼容参数：已不再由 advisor 内部做 YOLO")
    parser.add_argument("--conf", type=float, default=0.30, help="兼容参数：共享模式下忽略")
    parser.add_argument("--imgsz", type=int, default=128, help="兼容参数：共享模式下忽略")
    parser.add_argument("--device", type=str, default="0", help="兼容参数：共享模式下忽略")
    parser.add_argument("--half", action="store_true", help="兼容参数：共享模式下忽略")
    parser.add_argument("--infer-every", type=int, default=3, help="兼容参数：共享模式下忽略")
    parser.add_argument("--detect-roi", type=str, default="0.00,0.08,1.00,0.84", help="兼容参数：共享模式下忽略")
    parser.add_argument(
        "--shared-frame-path",
        type=str,
        default=str(os.environ.get("CSRL_SHARED_FRAME_PATH") or "/tmp/cs_rl_latest_frame.jpg"),
        help="由 stream_ffplay_pipeline.py 输出的最新原生帧路径",
    )
    parser.add_argument(
        "--shared-state-path",
        type=str,
        default=str(os.environ.get("CSRL_SHARED_STATE_PATH") or "/tmp/cs_rl_runtime_state.json"),
        help="由 stream_ffplay_pipeline.py 输出的运行状态路径",
    )
    parser.add_argument("--poll-interval-sec", type=float, default=0.10, help="共享文件轮询间隔")

    parser.add_argument("--ocr-engine", type=str, default="pytesseract", choices=["easyocr", "pytesseract"], help="OCR 引擎")
    parser.add_argument("--ocr-roi", action="append", default=[], help="OCR ROI，支持重复指定")
    parser.add_argument("--ocr-whitelist", type=str, default="0123456789/%:HPARMOABULLET", help="pytesseract 白名单")
    parser.add_argument("--ocr-min-conf", type=float, default=0.20, help="easyocr 置信度阈值")

    parser.add_argument("--location-roi", type=str, default="0.00,0.0,0.150,0.346", help="位置识别 ROI")
    parser.add_argument("--qwen-model", type=str, default="qwen3.6-plus", help="Qwen 视觉/文本模型")
    parser.add_argument("--api-key", type=str, default="", help="DASHSCOPE_API_KEY / QWEN_API_KEY / OPENAI_API_KEY")
    parser.add_argument("--auto-idle-query-sec", type=float, default=10.0, help="连续无 YOLO 结果达到该秒数后自动询问大模型")
    parser.add_argument("--auto-idle-query-cooldown-sec", type=float, default=5.0, help="自动询问冷却时间（秒）")
    return parser.parse_args()


def get_load_cv2() -> Any:
    try:
        return importlib.import_module("cv2")
    except ImportError as exc:
        raise ImportError("未安装 opencv-python。请先执行: pip install opencv-python") from exc


def get_read_shared_frame(cv2, frame_path: str):
    path = str(frame_path or "").strip()
    if not path:
        return None
    if not os.path.exists(path):
        return None
    frame = cv2.imread(path)
    if frame is None or frame.size == 0:
        return None
    return frame


def get_read_shared_centers(state_path: str) -> list[tuple[str, int, int, float]]:
    path = str(state_path or "").strip()
    if not path or (not os.path.exists(path)):
        return []

    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return []

    out: list[tuple[str, int, int, float]] = []
    for item in list((payload or {}).get("centers") or []):
        if not isinstance(item, dict):
            continue
        out.append(
            (
                str(item.get("name", "")),
                int(item.get("cx", 0)),
                int(item.get("cy", 0)),
                float(item.get("conf", 0.0)),
            )
        )
    return out


def get_start_stdin_thread(cmd_queue: "queue.Queue[str]") -> threading.Thread:
    def _worker() -> None:
        while True:
            try:
                line = sys.stdin.readline()
            except Exception:
                break
            if not line:
                time.sleep(0.05)
                continue
            cmd = str(line).strip().lower()
            if cmd:
                cmd_queue.put(cmd)

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    return t


def get_run_ocr_and_print(
    *,
    frame,
    ocr_interface: get_frame_ocr_interface,
) -> tuple[list[dict], str]:
    def _get_format_ocr_results_line(results: list[dict]) -> str:
        parts: list[str] = []
        for idx, item in enumerate(results):
            rid = int((item or {}).get("id", idx))
            text = str((item or {}).get("text", "") or "")
            parts.append(f'{rid}:{json.dumps(text, ensure_ascii=False)}')
        return "ocr_results=" + ",".join(parts)

    ocr_results = ocr_interface.get_detect(frame)
    ocr_text = ocr_interface.get_compact_text(ocr_results)
    print(_get_format_ocr_results_line(ocr_results), flush=True)
    return ocr_results, ocr_text


def main() -> int:
    args = get_args()
    cv2 = get_load_cv2()
    print(
        f"[advisor] 共享模式: frame={args.shared_frame_path} state={args.shared_state_path}",
        flush=True,
    )

    ocr_args = argparse.Namespace(
        ocr=True,
        ocr_engine=str(args.ocr_engine),
        ocr_roi=";".join([str(v) for v in (args.ocr_roi or [])]),
        ocr_min_conf=float(args.ocr_min_conf),
        ocr_whitelist=str(args.ocr_whitelist),
        ocr_lang="eng",
        ocr_cn_lang="chi_sim+eng",
    )
    ocr_interface = get_frame_ocr_interface(args=ocr_args, cv2_module=cv2)

    qwen_api_key = get_resolve_qwen_api_key(str(args.api_key or ""))
    qwen_client: Optional[get_qwen_location_client] = None
    if qwen_api_key:
        qwen_client = get_qwen_location_client(api_key=qwen_api_key, model=args.qwen_model)
    else:
        print("[advisor] 未配置 API Key，pos 命令将只输出 OCR 信息。", flush=True)

    location_roi = get_parse_roi(args.location_roi)

    cmd_queue: "queue.Queue[str]" = queue.Queue()
    get_start_stdin_thread(cmd_queue)

    print("[advisor] 输入命令: ocr | pos | help | quit", flush=True)

    last_enemy_signature = ""
    latest_frame = None
    latest_centers: list[tuple[str, int, int, float]] = []
    last_enemy_seen_t = time.monotonic()
    last_auto_query_t = 0.0
    auto_query_done_for_idle = False

    def handle_pending_commands() -> bool:
        while True:
            try:
                cmd = cmd_queue.get_nowait()
            except queue.Empty:
                break

            if cmd in {"quit", "exit", "q"}:
                print("[advisor] exit", flush=True)
                return True

            if cmd in {"help", "h", "?"}:
                print("[advisor] 可用命令: ocr | pos | help | quit", flush=True)
                continue

            if latest_frame is None:
                print("[advisor] 暂无可用帧，请稍后重试", flush=True)
                continue

            if cmd == "ocr":
                get_run_ocr_and_print(
                    frame=latest_frame,
                    ocr_interface=ocr_interface,
                )
                continue

            if cmd == "pos":
                pos_frame = latest_frame.copy()
                _, ocr_text = get_run_ocr_and_print(
                    frame=pos_frame,
                    ocr_interface=ocr_interface,
                )

                if qwen_client is None:
                    print("location=unknown", flush=True)
                    print("next_action=未配置 API Key，无法查询位置与大模型建议", flush=True)
                    continue

                location_text = qwen_client.get_query_location_from_frame(cv2, pos_frame, location_roi)
                full_image_data_url = get_build_image_data_url_from_frame(cv2, pos_frame, (0.0, 0.0, 1.0, 1.0))
                summary = (
                    f"敌人中心点={json.dumps(latest_centers, ensure_ascii=False) if latest_centers else 'none'}；"
                    f"位置={location_text or 'unknown'}；"
                    f"OCR={ocr_text or 'empty'}；"
                    "截图=同一时刻原生截图；"
                    "请给出下一步建议，只输出一句话。"
                )
                action_text = qwen_client.get_query_next_action(summary, full_image_data_url)
                if not action_text:
                    action_text = "建议继续观察、微调视角并保持掩体"

                print(f"location={location_text or 'unknown'}", flush=True)
                print(f"next_action={action_text}", flush=True)
                continue

            print(f"[advisor] 未知命令: {cmd}", flush=True)

        return False

    while True:
        if handle_pending_commands():
            return 0

        shared_frame = get_read_shared_frame(cv2, args.shared_frame_path)
        if shared_frame is not None:
            latest_frame = shared_frame

        centers = get_read_shared_centers(args.shared_state_path)
        latest_centers = centers

        if centers:
            now_seen = time.monotonic()
            last_enemy_seen_t = now_seen
            auto_query_done_for_idle = False
            signature = json.dumps(centers, ensure_ascii=False)
            if signature != last_enemy_signature:
                print(f"yolo_centers={signature}", flush=True)
                print("decision=enemy_visible suggestion=建议停止移动，优先瞄准并开火", flush=True)
                last_enemy_signature = signature

        auto_now = time.monotonic()
        idle_sec = max(0.5, float(args.auto_idle_query_sec))
        auto_cooldown_sec = max(0.5, float(args.auto_idle_query_cooldown_sec))
        no_enemy_too_long = (auto_now - last_enemy_seen_t) >= idle_sec
        cooldown_ok = (auto_now - last_auto_query_t) >= auto_cooldown_sec
        if no_enemy_too_long and cooldown_ok and (not auto_query_done_for_idle):
            print(f"auto_strategy_trigger=no_enemy_{int(idle_sec)}s", flush=True)
            if latest_frame is None:
                print("[advisor] 暂无可用帧，跳过本次自动询问", flush=True)
                last_auto_query_t = auto_now
                auto_query_done_for_idle = True
                time.sleep(max(0.02, float(args.poll_interval_sec)))
                continue
            auto_frame = latest_frame.copy()
            _, auto_ocr_text = get_run_ocr_and_print(
                frame=auto_frame,
                ocr_interface=ocr_interface,
            )

            if qwen_client is None:
                print("location=unknown", flush=True)
                print("next_action=未配置 API Key，无法查询位置与大模型建议", flush=True)
                last_auto_query_t = auto_now
                auto_query_done_for_idle = True
                continue

            auto_location_text = qwen_client.get_query_location_from_frame(cv2, auto_frame, location_roi)
            auto_full_image_data_url = get_build_image_data_url_from_frame(cv2, auto_frame, (0.0, 0.0, 1.0, 1.0))
            auto_summary = (
                f"连续{int(idle_sec)}秒未检测到敌人；"
                f"敌人中心点={json.dumps(latest_centers, ensure_ascii=False) if latest_centers else 'none'}；"
                f"位置={auto_location_text or 'unknown'}；"
                f"OCR={auto_ocr_text or 'empty'}；"
                "截图=同一时刻原生截图；"
                "请给出下一步建议，只输出一句话。"
            )
            auto_action_text = qwen_client.get_query_next_action(auto_summary, auto_full_image_data_url)
            if not auto_action_text:
                auto_action_text = "建议继续观察、微调视角并保持掩体"

            print(f"location={auto_location_text or 'unknown'}", flush=True)
            print(f"next_action={auto_action_text}", flush=True)
            last_auto_query_t = auto_now
            auto_query_done_for_idle = True

            time.sleep(max(0.02, float(args.poll_interval_sec)))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
