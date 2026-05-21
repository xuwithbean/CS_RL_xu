#!/usr/bin/env python3
"""实时分析程序：实时输出敌情建议，并通过终端命令触发 OCR/位置分析。"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional
import numpy as np

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

from actions import m_actions
import control as control_mod
try:
    import torch
    from point_aim_trainer import load_model
except Exception:
    torch = None
    load_model = None


ACTION_CODE_TO_NAME = {
    "A": "左平移",
    "B": "右转",
    "C": "左转",
    "D": "右平移",
    "E": "跳跃",
}

ACTION_CODE_TO_KEY = {
    "A": "a",
    "D": "d",
    "E": "space",
}

# 新增蹲下动作 F -> ctrl
ACTION_CODE_TO_NAME["F"] = "蹲下"
ACTION_CODE_TO_KEY["F"] = "ctrl"

ACTION_CODE_TO_MOUSE_DX = {
    "B": 1000,
    "C": -1000,
}


def get_action_choice_label(action_code: str) -> str:
    codes = []
    for part in str(action_code or "").upper().replace(" ", "").split("+"):
        if part in ACTION_CODE_TO_NAME and part not in codes:
            codes.append(part)
    if not codes:
        return "无"
    return "+".join(f"{code}({ACTION_CODE_TO_NAME[code]})" for code in codes)


def normalize_action_code(raw_action: str) -> str:
    text = str(raw_action or "").upper()
    picked: list[str] = []
    for ch in text:
        if ch in ACTION_CODE_TO_NAME and ch not in picked:
            picked.append(ch)
    return "+".join(picked)


def get_action_keys(action_code: str) -> list[str]:
    keys: list[str] = []
    for part in str(action_code or "").upper().replace(" ", "").split("+"):
        key = ACTION_CODE_TO_KEY.get(part)
        if key and key not in keys:
            keys.append(key)
    return keys


def get_action_mouse_dx(action_code: str) -> int:
    mouse_dx = 0
    for part in str(action_code or "").upper().replace(" ", "").split("+"):
        mouse_dx += int(ACTION_CODE_TO_MOUSE_DX.get(part, 0))
    return mouse_dx


def execute_action_choice(action_code: str, controller: m_actions, hold_sec: float = 0.12) -> str:
    keys = get_action_keys(action_code)
    mouse_dx = get_action_mouse_dx(action_code)
    executed_parts: list[str] = []
    if keys:
        controller.key_sender.press_and_release(keys, hold_ms=max(20, int(float(hold_sec) * 1000)))
        executed_parts.append("keys=" + "+".join(keys))
    if mouse_dx != 0:
        controller.mouse_move(mouse_dx, 0)
        executed_parts.append(f"mouse_dx={mouse_dx}")
    return ";".join(executed_parts) if executed_parts else "无"


def get_query_next_action_with_choice(qwen_client, summary_text: str, image_data_url: str) -> dict[str, str]:
    response = qwen_client.client.chat.completions.create(
        model=qwen_client.model,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "你是CS战术助手。请同时输出下一步建议和动作选择。\n"
                            "可选动作只有以下 5 个字母：\n"
                            "A=左平移（A键）\n"
                            "B=右转（鼠标x=+1000）\n"
                            "C=左转（鼠标x=-1000）\n"
                            "D=右平移（D键）\n"
                            "E=跳跃\n"
                            "F=蹲下（Ctrl键）\n"
                            "你可以选择单个动作，也可以选择多个动作组合；组合时用 + 连接，顺序按 A/B/C/D/E/F。\n"
                            "如果当前不需要动作，可将 action 置空字符串。\n"
                            "只输出 JSON，不要输出额外解释，格式必须是：\n"
                            '{"suggestion":"一句话建议","action":"A+B"}\n'
                            f"当前状态：{summary_text}\n"
                            "请严格按 JSON 返回。"
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": image_data_url},
                    },
                ],
            },
        ],
        extra_body={"enable_thinking": False},
        stream=False,
    )
    raw_text = str((response.choices[0].message.content or "").strip())
    suggestion = raw_text
    action_code = ""
    try:
        payload = json.loads(raw_text)
        if isinstance(payload, dict):
            suggestion = str(payload.get("suggestion", "") or raw_text)
            action_code = normalize_action_code(str(payload.get("action", "") or ""))
    except Exception:
        suggestion = raw_text
        action_code = normalize_action_code(raw_text)

    return {
        "suggestion": suggestion,
        "action_code": action_code,
        "action_label": get_action_choice_label(action_code),
        "raw_text": raw_text,
    }


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
    parser.add_argument("--auto-idle-query-sec", type=float, default=3.0, help="连续无 YOLO 结果达到该秒数后自动询问大模型")
    parser.add_argument("--auto-idle-query-cooldown-sec", type=float, default=5.0, help="自动询问冷却时间（秒）")
    parser.add_argument("--aim-model-path", type=str, default="point_aim_net_resume_best.pt", help="瞄准用模型路径")
    parser.add_argument("--aim-move-gain-x", type=float, default=2500.0)
    parser.add_argument("--aim-move-gain-y", type=float, default=500.0)
    parser.add_argument("--aim-max-move-x", type=int, default=1000)
    parser.add_argument("--aim-max-move-y", type=int, default=500)
    parser.add_argument("--debug-aim", action="store_true", help="临时打印瞄准模型输入/输出用于调试")
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


def get_read_shared_centers(state_path: str) -> tuple[list[tuple[str, int, int, float]], tuple[int, int] | None]:
    path = str(state_path or "").strip()
    if not path or (not os.path.exists(path)):
        return [], None

    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return [], None

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
    ref_w = int((payload or {}).get("centers_ref_w") or 0)
    ref_h = int((payload or {}).get("centers_ref_h") or 0)
    ref_size = (ref_w, ref_h) if ref_w > 0 and ref_h > 0 else None
    return out, ref_size


def get_build_aim_target(
    centers: list[tuple[str, int, int, float]],
    frame_shape: tuple[int, int] | None,
) -> dict[str, Any] | None:
    """基于检测中心点选择主目标，并计算相对准星（画面中心）的误差。"""
    if not centers or frame_shape is None:
        return None

    h, w = int(frame_shape[0]), int(frame_shape[1])
    if h <= 0 or w <= 0:
        return None

    cx0 = w // 2
    cy0 = h // 2

    head_alias = {
        "head",
        "enemy_head",
        "person_head",
        "ct_head",
        "t_head",
    }
    body_alias = {
        "person",
        "enemy",
        "ct",
        "t",
        "body",
    }

    typed: list[tuple[str, int, int, float, str]] = []
    for name, x, y, conf in centers:
        lname = str(name or "").strip().lower()
        if lname in head_alias or "head" in lname:
            target_type = "head"
        elif lname in body_alias:
            target_type = "body"
        else:
            target_type = "other"
        typed.append((str(name), int(x), int(y), float(conf), target_type))

    def _pick(candidates: list[tuple[str, int, int, float, str]]) -> tuple[str, int, int, float, str] | None:
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda t: (
                math.hypot(float(t[1] - cx0), float(t[2] - cy0)),
                -float(t[3]),
            ),
        )

    head_candidates = [c for c in typed if c[4] == "head"]
    body_candidates = [c for c in typed if c[4] == "body"]
    other_candidates = [c for c in typed if c[4] == "other"]

    chosen = _pick(head_candidates) or _pick(body_candidates) or _pick(other_candidates)
    if chosen is None:
        return None

    name, tx, ty, conf, target_type = chosen
    dx = int(tx - cx0)
    dy = int(ty - cy0)
    norm = max(1.0, math.hypot(float(w) / 2.0, float(h) / 2.0))
    aim_error = max(0.0, min(1.0, math.hypot(float(dx), float(dy)) / norm))

    return {
        "target_name": name,
        "target_type": target_type,
        "target_x": int(tx),
        "target_y": int(ty),
        "crosshair_x": int(cx0),
        "crosshair_y": int(cy0),
        "dx": int(dx),
        "dy": int(dy),
        "aim_error": float(aim_error),
        "conf": float(conf),
    }


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


def get_query_kill_count_from_frame(
    cv2,
    frame,
    qwen_client: Optional[get_qwen_location_client],
    roi_rel: tuple[float, float, float, float],
    summary_text: str,
) -> dict[str, Any]:
    if qwen_client is None or frame is None:
        return {"kill_count": 0, "reason": "qwen_client_unavailable"}

    image_data_url = get_build_image_data_url_from_frame(cv2, frame, roi_rel)
    try:
        response = qwen_client.client.chat.completions.create(
            model=qwen_client.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "你是CS击杀计数助手。请根据截图和状态读取当前累计击杀数。"
                                "只输出JSON，不要输出额外解释，格式必须是："
                                "{\"kill_count\": 非负整数, \"reason\": \"一句话原因\"}。"
                                "kill_count 必须表示当前总击杀数，不要判断是否击杀。\n"
                                f"当前状态：{summary_text}\n"
                                "请严格按JSON返回。"
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": image_data_url},
                        },
                    ],
                },
            ],
            extra_body={"enable_thinking": False},
            stream=False,
        )
        raw_text = str((response.choices[0].message.content or "").strip())
        try:
            payload = json.loads(raw_text)
            if not isinstance(payload, dict):
                raise ValueError("kill count payload is not a dict")
        except Exception:
            payload = {"kill_count": 0, "reason": raw_text}

        payload["kill_count"] = int(max(0, int(payload.get("kill_count", 0))))
        payload["reason"] = str(payload.get("reason", raw_text) or "")
        return payload
    except Exception as exc:
        return {"kill_count": 0, "reason": f"error:{type(exc).__name__}:{exc}"}


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

    controller = m_actions()

    # held keys between LLM queries (we press KEY_DOWN on new keys and KEY_UP on release)
    held_keys: set[str] = set()

    # helper to send key down/up via controller key sender
    def _key_down(keys: list[str]) -> None:
        if not keys:
            return
        vks = [str(control_mod._char_to_vk(k)) for k in control_mod._normalize_keys(keys)]
        controller.key_sender.client.send_lines([f"KEY_DOWN {vk}" for vk in vks])

    def _key_up(keys: list[str]) -> None:
        if not keys:
            return
        vks = [str(control_mod._char_to_vk(k)) for k in control_mod._normalize_keys(keys)]
        controller.key_sender.client.send_lines([f"KEY_UP {vk}" for vk in vks])

    # smooth mouse move helper: split large moves into small steps to appear smoother
    def smooth_mouse_move(ctrl: m_actions, dx: int, dy: int, max_step: int = 80, delay: float = 0.008) -> None:
        dx = int(dx)
        dy = int(dy)
        max_abs = max(abs(dx), abs(dy), 1)
        steps = int((max_abs + max_step - 1) // max_step)
        if steps <= 1:
            try:
                ctrl.mouse_move(dx, dy)
            except Exception:
                pass
            return
        sx = float(dx) / float(steps)
        sy = float(dy) / float(steps)
        acc_x = 0.0
        acc_y = 0.0
        for i in range(steps):
            acc_x += sx
            acc_y += sy
            step_x = int(round(acc_x))
            step_y = int(round(acc_y))
            # reset accumulators by subtracting applied integer part
            acc_x -= step_x
            acc_y -= step_y
            try:
                ctrl.mouse_move(step_x, step_y)
            except Exception:
                pass
            time.sleep(max(0.0, float(delay)))

    aim_model = None
    aim_device = torch.device("cpu") if torch is not None else None
    if torch is not None and load_model is not None:
        try:
            aim_model = load_model(str(args.aim_model_path), device=aim_device, hidden_dim=64)
        except Exception:
            aim_model = None

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
                    "请给出下一步建议，并同时选择动作。"
                )
                action_result = get_query_next_action_with_choice(qwen_client, summary, full_image_data_url)
                action_text = action_result.get("suggestion") or "建议继续观察、微调视角并保持掩体"

                # 按住新动作键，释放旧的不在新集合中的键（按键保持直到下一次询问）
                new_keys = set(get_action_keys(str(action_result.get("action_code") or "")))
                to_release = list(held_keys - new_keys)
                if to_release:
                    _key_up(to_release)
                to_press = list(new_keys - held_keys)
                if to_press:
                    _key_down(to_press)
                held_keys = new_keys

                # 只执行鼠标移动部分（键盘由按下/释放管理）
                mouse_dx = get_action_mouse_dx(str(action_result.get("action_code") or ""))
                mouse_exec = "无"
                if mouse_dx != 0:
                    smooth_mouse_move(controller, mouse_dx, 0, max_step=80, delay=0.006)
                    mouse_exec = f"mouse_dx={mouse_dx}"

                executed = f"held_keys={'+'.join(sorted(held_keys))};{mouse_exec}"

                print(f"location={location_text or 'unknown'}", flush=True)
                print(f"next_action={action_text}", flush=True)
                print(f"action_choice={action_result.get('action_code') or '无'}", flush=True)
                print(f"action_label={action_result.get('action_label') or '无'}", flush=True)
                print(f"action_executed={executed}", flush=True)
                continue

            print(f"[advisor] 未知命令: {cmd}", flush=True)

        return False

    while True:
        if handle_pending_commands():
            return 0

        shared_frame = get_read_shared_frame(cv2, args.shared_frame_path)
        if shared_frame is not None:
            latest_frame = shared_frame

        centers, centers_ref = get_read_shared_centers(args.shared_state_path)
        latest_centers = centers

        if centers:
            now_seen = time.monotonic()
            last_enemy_seen_t = now_seen
            auto_query_done_for_idle = False
            signature = json.dumps(centers, ensure_ascii=False)
            if signature != last_enemy_signature:
                # suppressed verbose YOLO center log
                # use centers' reference resolution when available to compute aim target
                frame_shape_param = None
                if centers_ref is not None:
                    frame_shape_param = (int(centers_ref[1]), int(centers_ref[0]))
                else:
                    frame_shape_param = (latest_frame.shape[0], latest_frame.shape[1]) if latest_frame is not None else None
                target_payload = get_build_aim_target(
                    centers=centers,
                    frame_shape=frame_shape_param,
                )
                # suppressed aim_target print to reduce verbosity
                if latest_frame is not None:
                    # 见到人，立即停止所有动作并释放已按下的键/鼠标
                    try:
                        controller.stop()
                    except Exception:
                        pass
                    if held_keys:
                        _key_up(list(held_keys))
                        held_keys.clear()

                    # 使用瞄准模型进行快速瞄准（若可用）——不依赖 qwen_client
                    if aim_model is not None and target_payload is not None:
                        try:
                            tx = float(target_payload.get("target_x", 0))
                            ty = float(target_payload.get("target_y", 0))
                            cx = float(target_payload.get("crosshair_x", 1))
                            cy = float(target_payload.get("crosshair_y", 1))
                            aim_err = float(target_payload.get("aim_error", 0.0))
                            ndx = (tx - cx) / max(1.0, float(cx))
                            ndy = (ty - cy) / max(1.0, float(cy))
                            if getattr(args, 'debug_aim', False):
                                print(f"[aim-debug] target(tx,ty)=({tx},{ty}) crosshair=({cx},{cy}) pix_dx={tx-cx} pix_dy={ty-cy} ndx={ndx:.4f} ndy={ndy:.4f}", flush=True)
                            inp = np.array([[ndx, ndy, float(aim_err)]], dtype=np.float32)
                            t_inp = torch.from_numpy(inp).to(aim_device)
                            with torch.no_grad():
                                out_t = aim_model(t_inp)
                            out = out_t.detach().cpu().numpy().squeeze(0)
                            # out expected in [-1,1] for movement fraction
                            aim_scale = 0.5
                            dx = int(np.clip(float(out[0]) * float(args.aim_move_gain_x) * aim_scale, -float(args.aim_max_move_x), float(args.aim_max_move_x)))
                            dy = int(np.clip(float(out[1]) * float(args.aim_move_gain_y) * aim_scale, -float(args.aim_max_move_y), float(args.aim_max_move_y)))
                            # optional debug prints
                            if getattr(args, 'debug_aim', False):
                                print(f"[aim-debug] aim_err={aim_err:.4f} out=({out[0]:.4f},{out[1]:.4f}) dx={dx} dy={dy}", flush=True)
                            # smooth the mouse movement
                            smooth_mouse_move(controller, dx, dy, max_step=80, delay=0.006)
                            # when within 12% error, perform quick clicks (3 clicks, 0.01s interval)
                            if float(aim_err) <= 0.12:
                                try:
                                    if getattr(args, 'debug_aim', False):
                                        print(f"[aim-debug] firing because aim_err={aim_err:.4f} <= 0.12", flush=True)
                                    controller.mouse_click_interval(click_times=3, interval_sec=0.01, hold_sec=0.01)
                                except Exception:
                                    if getattr(args, 'debug_aim', False):
                                        print("[aim-debug] click failed", flush=True)
                        except Exception:
                            pass

                    # 如果配置了 qwen_client，则继续调用大模型询问动作建议
                    if qwen_client is not None:
                        enemy_frame = latest_frame.copy()
                        enemy_summary = (
                            f"敌人已出现；敌人中心点={json.dumps(centers, ensure_ascii=False)}；"
                            f"瞄准信息={json.dumps(target_payload, ensure_ascii=False) if target_payload is not None else 'none'}；"
                            "请给出下一步建议，并同时选择动作。"
                        )
                        enemy_image_data_url = get_build_image_data_url_from_frame(cv2, enemy_frame, (0.0, 0.0, 1.0, 1.0))
                        enemy_action_result = get_query_next_action_with_choice(qwen_client, enemy_summary, enemy_image_data_url)
                        enemy_action_text = enemy_action_result.get("suggestion") or "建议停止移动，优先瞄准并开火"
                        # 将建议动作按持有逻辑执行（鼠标单次移动）
                        new_keys = set(get_action_keys(str(enemy_action_result.get("action_code") or "")))
                        to_release = list(held_keys - new_keys)
                        if to_release:
                            _key_up(to_release)
                        to_press = list(new_keys - held_keys)
                        if to_press:
                            _key_down(to_press)
                        held_keys = new_keys
                        mouse_dx = get_action_mouse_dx(str(enemy_action_result.get("action_code") or ""))
                        mouse_exec = "无"
                        if mouse_dx != 0:
                            smooth_mouse_move(controller, mouse_dx, 0, max_step=80, delay=0.006)
                            mouse_exec = f"mouse_dx={mouse_dx}"
                        enemy_executed = f"held_keys={'+'.join(sorted(held_keys))};{mouse_exec}"
                        print(f"decision=enemy_visible suggestion={enemy_action_text}", flush=True)
                        print(f"action_choice={enemy_action_result.get('action_code') or '无'}", flush=True)
                        print(f"action_label={enemy_action_result.get('action_label') or '无'}", flush=True)
                        print(f"action_executed={enemy_executed}", flush=True)
                    else:
                        print("decision=enemy_visible suggestion=建议停止移动，优先瞄准并开火", flush=True)
                else:
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
                "请给出下一步建议，并同时选择动作。"
            )
            auto_action_result = get_query_next_action_with_choice(qwen_client, auto_summary, auto_full_image_data_url)
            auto_action_text = auto_action_result.get("suggestion") or "建议继续观察、微调视角并保持掩体"

            new_keys = set(get_action_keys(str(auto_action_result.get("action_code") or "")))
            to_release = list(held_keys - new_keys)
            if to_release:
                _key_up(to_release)
            to_press = list(new_keys - held_keys)
            if to_press:
                _key_down(to_press)
            held_keys = new_keys

            mouse_dx = get_action_mouse_dx(str(auto_action_result.get("action_code") or ""))
            mouse_exec = "无"
            if mouse_dx != 0:
                smooth_mouse_move(controller, mouse_dx, 0, max_step=80, delay=0.006)
                mouse_exec = f"mouse_dx={mouse_dx}"

            auto_executed = f"held_keys={'+'.join(sorted(held_keys))};{mouse_exec}"

            print(f"location={auto_location_text or 'unknown'}", flush=True)
            print(f"next_action={auto_action_text}", flush=True)
            print(f"action_choice={auto_action_result.get('action_code') or '无'}", flush=True)
            print(f"action_label={auto_action_result.get('action_label') or '无'}", flush=True)
            print(f"action_executed={auto_executed}", flush=True)
            last_auto_query_t = auto_now
            auto_query_done_for_idle = True

            time.sleep(max(0.02, float(args.poll_interval_sec)))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
