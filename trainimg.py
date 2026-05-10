"""简易点流展示器：从视觉管线的共享状态读取检测中心点，输出纯白背景黑点的视频流。

用法示例：
  python trainimg.py --out-stream udp://127.0.0.1:23000 --fps 20

默认会读取环境变量 `CSRL_SHARED_FRAME_PATH` 与 `CSRL_SHARED_STATE_PATH`（和 `stream_ffplay_pipeline.py` 一致），
并以共享帧的分辨率作为输出分辨率（若共享帧不可用，回退到 `--width/--height`）。
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from visual_recognition.stream_ffplay_pipeline import get_start_ffmpeg_stream_writer, get_latest_frame_stream_writer


def read_state_centers(state_path: str) -> tuple[list[tuple[str, int, int, float]], Optional[tuple[int, int]]]:
    path = str(state_path or "").strip()
    if not path or (not os.path.exists(path)):
        return [], None
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return [], None
    out = []
    for item in list((payload or {}).get("centers") or []):
        try:
            out.append((str(item.get("name", "")), int(item.get("cx", 0)), int(item.get("cy", 0)), float(item.get("conf", 0.0))))
        except Exception:
            continue

    ref_w = int((payload or {}).get("centers_ref_w") or 0)
    ref_h = int((payload or {}).get("centers_ref_h") or 0)
    ref_size = (ref_w, ref_h) if (ref_w > 0 and ref_h > 0) else None
    return out, ref_size


def read_shared_frame_size(frame_path: str) -> Optional[tuple[int, int]]:
    p = str(frame_path or "").strip()
    if not p or (not os.path.exists(p)):
        return None
    try:
        img = cv2.imread(p)
        if img is None:
            return None
        h, w = img.shape[:2]
        return w, h
    except Exception:
        return None


def build_frame(
    w: int,
    h: int,
    centers: list[tuple[str, int, int, float]],
    point_radius: int | None = None,
    centers_ref_size: Optional[tuple[int, int]] = None,
) -> Any:
    # 白色背景
    img = np.full((h, w, 3), 255, dtype=np.uint8)

    def _get_point_color(name: str) -> tuple[int, int, int]:
        lname = str(name or "").upper()
        if "CT_HEAD" in lname:
            return (0, 255, 0)
        if "T_HEAD" in lname:
            return (0, 255, 255)
        if lname.startswith("CT"):
            return (255, 0, 0)
        if lname.startswith("T"):
            return (0, 0, 255)
        return (0, 0, 0)

    if centers:
        default_radius = max(1, int(min(w, h) * 0.015))
        radius = int(point_radius) if (point_radius is not None and int(point_radius) > 0) else default_radius
        scale_x = 1.0
        scale_y = 1.0
        if centers_ref_size is not None:
            ref_w, ref_h = centers_ref_size
            if ref_w > 0 and ref_h > 0:
                scale_x = float(w) / float(ref_w)
                scale_y = float(h) / float(ref_h)
        for name, cx, cy, conf in centers:
            x = max(0, min(w - 1, int(round(float(cx) * scale_x))))
            y = max(0, min(h - 1, int(round(float(cy) * scale_y))))
            cv2.circle(img, (x, y), radius, _get_point_color(name), thickness=-1)
    return img


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stream simple points from shared centers")
    p.add_argument("--shared-frame-path", type=str, default=str(os.environ.get("CSRL_SHARED_FRAME_PATH") or "/tmp/cs_rl_latest_frame.jpg"))
    p.add_argument("--shared-state-path", type=str, default=str(os.environ.get("CSRL_SHARED_STATE_PATH") or "/tmp/cs_rl_runtime_state.json"))
    p.add_argument("--out-stream", type=str, default="", help="输出流地址，如 udp://127.0.0.1:23000；为空则只预览窗口")
    p.add_argument("--ffmpeg", type=str, default="ffmpeg")
    p.add_argument("--out-vcodec", type=str, default="mpeg2video")
    p.add_argument("--out-bitrate", type=str, default="800k")
    p.add_argument("--fps", type=float, default=20.0)
    p.add_argument("--point-radius", type=int, default=0, help="覆盖默认点半径（像素），0 表示自动")
    p.add_argument("--width", type=int, default=0, help="当共享帧不可用时回退宽度")
    p.add_argument("--height", type=int, default=0, help="当共享帧不可用时回退高度")
    p.add_argument("--preview", action="store_true", help="在本地弹窗预览（不输出流）")
    return p.parse_args()


def main() -> int:
    args = get_args()
    shared_frame_path = str(args.shared_frame_path or "").strip()
    shared_state_path = str(args.shared_state_path or "").strip()

    _, init_ref_size = read_state_centers(shared_state_path)
    w_h = read_shared_frame_size(shared_frame_path)
    width, height = (args.width or 0), (args.height or 0)
    # 优先使用 centers 的参考坐标尺寸，使点流与 YOLO 输出流尺寸一致。
    if init_ref_size is not None:
        width, height = init_ref_size
    elif w_h is not None:
        width, height = w_h

    if width <= 0 or height <= 0:
        print("无法确定输出分辨率：请确保共享帧存在，或提供 --width/--height 参数", flush=True)
        return 2

    writer = None
    proc = None
    if args.out_stream:
        proc = get_start_ffmpeg_stream_writer(
            output_url=args.out_stream,
            width=int(width),
            height=int(height),
            fps=float(args.fps),
            ffmpeg_path=str(args.ffmpeg),
            out_vcodec=str(args.out_vcodec),
            out_bitrate=str(args.out_bitrate),
        )
        writer = get_latest_frame_stream_writer(proc)

    try:
        centers_ref_size = init_ref_size
        while True:
            centers, latest_ref_size = read_state_centers(shared_state_path)
            if latest_ref_size is not None:
                centers_ref_size = latest_ref_size

            frame = build_frame(
                width,
                height,
                centers,
                point_radius=args.point_radius,
                centers_ref_size=centers_ref_size,
            )

            if writer is not None:
                try:
                    writer.send(frame.tobytes())
                except Exception:
                    pass

            if args.preview:
                cv2.imshow("trainimg_preview", frame)
                if cv2.waitKey(int(1000.0 / max(1.0, float(args.fps)))) & 0xFF == ord("q"):
                    break
            else:
                # 当不预览且不输出时，仍保持节拍
                time.sleep(max(0.001, 1.0 / max(1.0, float(args.fps))))

    except KeyboardInterrupt:
        pass
    finally:
        try:
            if writer is not None:
                writer.close()
        except Exception:
            pass
        try:
            if proc is not None and proc.stdin is not None:
                proc.stdin.close()
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
