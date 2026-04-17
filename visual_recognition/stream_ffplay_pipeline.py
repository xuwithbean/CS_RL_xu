"""YOLO 实时流管线：输入流 -> YOLO -> 输出流，并仅保留 Windows 侧观测窗口。"""

from __future__ import annotations

import argparse
import base64
import importlib
import json
import os
import queue
import re
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from opengame import OpenGameTool


LATEST_CENTER_POINTS: list[tuple[str, int, int, float]] = []
LATEST_OCR_RESULTS: list[dict] = []
LATEST_LOCATION_RESULT: str = ""


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Realtime YOLO pipeline with Windows viewer only")
    parser.add_argument("--weights", type=str, required=True, help="YOLO 权重路径")

    # 保持与现有脚本兼容。
    parser.add_argument("--game-exe", type=str, default="", help="Windows 游戏路径")
    parser.add_argument("--window-title", type=str, default="auto", help="窗口标题")
    parser.add_argument("--wait-game", type=float, default=6.0, help="启动游戏后等待秒数")
    parser.add_argument("--name", type=str, default="ct_t_yolo_ffplay", help="运行名")
    parser.add_argument("--skip-win-stream", action="store_true", help="跳过启动 opengame 推流")
    parser.add_argument("--linux-ip", type=str, default="auto", help="Windows 推流目标 Linux IP")
    parser.add_argument("--port", type=int, default=12345, help="输入流端口")
    parser.add_argument("--framerate", type=int, default=60, help="输入流帧率")
    parser.add_argument("--bitrate", type=str, default="2500k", help="输入流码率")
    parser.add_argument("--frame-drain", type=int, default=0, help="兼容参数")
    parser.add_argument("--capture-drain", type=int, default=0, help="兼容参数")

    parser.add_argument("--in-stream", type=str, default="", help="输入流地址，如 udp://ip:port")
    parser.add_argument("--source", type=str, default="", help="输入流别名")
    parser.add_argument("--out-stream", type=str, default="", help="输出流地址，如 udp://ip:2234")
    parser.add_argument("--preview", action="store_true", help="启用 Windows 侧观测窗口")

    parser.add_argument("--conf", type=float, default=0.30, help="置信度")
    parser.add_argument("--imgsz", type=int, default=128, help="推理尺寸")
    parser.add_argument("--device", type=str, default="0", help="推理设备")
    parser.add_argument("--half", action="store_true", help="半精度推理")
    parser.add_argument("--infer-every", type=int, default=2, help="每 N 帧推理一次")
    parser.add_argument("--boxes-ttl-ms", type=int, default=350, help="检测框保留时长(毫秒)，超时自动清空")
    parser.add_argument(
        "--family-conflict-iou",
        type=float,
        default=0.35,
        help="CT/T 系列互斥时的 IoU 阈值（超过则仅保留高置信度一方）",
    )
    parser.add_argument(
        "--accel-mode",
        type=str,
        default="auto",
        choices=["auto", "none", "trt", "compile"],
        help="推理加速模式：auto/none/trt/compile",
    )
    parser.add_argument(
        "--infer-worker",
        type=str,
        default="auto",
        choices=["auto", "on", "off"],
        help="推理执行模式：auto=TRT同步/PT异步，on=强制异步，off=强制同步",
    )

    parser.add_argument("--work-size", type=str, default="288x162", help="处理分辨率")
    parser.add_argument("--output-max-size", type=str, default="768x432", help="输出最大分辨率，保持比例")
    parser.add_argument("--detect-roi", type=str, default="0.00,0.08,1.00,0.84", help="ROI 相对坐标")
    parser.add_argument("--line-width", type=int, default=2, help="线宽")
    parser.add_argument("--preview-size", type=str, default="800x450", help="兼容参数")

    parser.add_argument("--stream-fps", type=float, default=30.0, help="输出帧率")
    parser.add_argument("--capture-reconnect-sec", type=float, default=2.0, help="无帧重连秒数")
    parser.add_argument("--capture-timeout-ms", type=int, default=4000, help="读取超时毫秒")
    parser.add_argument("--first-frame-timeout-sec", type=float, default=20.0, help="首帧等待超时")
    parser.add_argument("--udp-fifo-size", type=int, default=32768, help="UDP fifo")
    parser.add_argument("--capture-probesize", type=int, default=131072, help="输入探测字节")
    parser.add_argument("--capture-analyzeduration", type=int, default=300000, help="输入分析时长")
    parser.add_argument("--sender-udp-pkt-size", type=int, default=1316, help="兼容参数")
    parser.add_argument("--sender-udp-buffer-size", type=int, default=262144, help="兼容参数")
    parser.add_argument("--win-vcodec", type=str, default="mpeg2video", help="兼容参数")
    parser.add_argument("--print-yolo", action="store_true", help="兼容参数")
    parser.add_argument("--yolo-info-jsonl", type=str, default="", help="兼容参数")
    parser.add_argument("--ocr", action="store_true", help="兼容参数")
    parser.add_argument("--ocr-engine", type=str, default="pytesseract", help="兼容参数")
    parser.add_argument("--ocr-roi", type=str, default="", help="兼容参数")
    parser.add_argument("--draw-ocr-roi", action="store_true", help="在视频中绘制 OCR ROI 区域")
    parser.add_argument("--ocr-min-conf", type=float, default=0.20, help="兼容参数")
    parser.add_argument("--ocr-whitelist", type=str, default="0123456789/%:HPARMOABULLET", help="兼容参数")
    parser.add_argument("--ocr-info-jsonl", type=str, default="", help="兼容参数")
    parser.add_argument("--ocr-every", type=int, default=10, help="每 N 帧执行一次 OCR")
    parser.add_argument("--ocr-print-interval-sec", type=float, default=0.5, help="OCR 结果打印间隔秒数")
    parser.add_argument("--ocr-lang", type=str, default="eng", help="默认 OCR 语言，如 eng")
    parser.add_argument("--ocr-cn-lang", type=str, default="chi_sim+eng", help="0号ROI中文识别语言")
    location_group = parser.add_mutually_exclusive_group()
    location_group.add_argument("--location-detect", dest="location_detect", action="store_true", help="启用地图位置识别（Qwen）")
    location_group.add_argument("--no-location-detect", dest="location_detect", action="store_false", help="关闭地图位置识别（Qwen）")
    parser.set_defaults(location_detect=True)
    parser.add_argument("--location-every", type=float, default=5.0, help="每 N 秒请求一次位置识别")
    parser.add_argument("--location-roi", type=str, default="0.00,0.0,0.150,0.346", help="位置识别 ROI 相对坐标")
    parser.add_argument("--location-model", type=str, default="qwen3.6-plus", help="位置识别模型")
    parser.add_argument("--location-print-interval-sec", type=float, default=1.0, help="位置结果打印间隔秒")

    parser.add_argument("--out-vcodec", type=str, default="mpeg2video", help="输出编码器")
    parser.add_argument("--out-bitrate", type=str, default="4000k", help="输出码率")
    parser.add_argument("--ffmpeg", type=str, default="ffmpeg", help="ffmpeg 可执行文件")
    parser.add_argument("--ffplay", type=str, default="ffplay", help="Windows 侧 ffplay 可执行文件")
    return parser.parse_args()


def get_pick_accel_weights(weights_path: str, accel_mode: str) -> str:
    """根据加速模式优先选择 TensorRT engine 权重。"""
    mode = str(accel_mode or "auto").lower()
    w = Path(weights_path)

    if w.suffix.lower() == ".engine":
        return str(w)

    engine_path = w.with_suffix(".engine")
    if mode in {"trt", "auto"} and engine_path.exists():
        return str(engine_path)

    return str(w)


def get_enable_torch_runtime_opt(torch_module) -> None:
    """打开对实时推理更友好的 torch 运行时选项。"""
    try:
        if torch_module.cuda.is_available():
            torch_module.backends.cudnn.benchmark = True
            torch_module.backends.cuda.matmul.allow_tf32 = True
            torch_module.backends.cudnn.allow_tf32 = True
    except Exception:
        pass


def get_parse_size(size_spec: str) -> tuple[int, int]:
    m = re.match(r"^\s*(\d+)\s*[xX]\s*(\d+)\s*$", str(size_spec or ""))
    if not m:
        raise ValueError(f"invalid size spec: {size_spec}")
    w = int(m.group(1))
    h = int(m.group(2))
    if w <= 0 or h <= 0:
        raise ValueError(f"invalid size values: {size_spec}")
    return w, h


def get_parse_roi(roi_spec: str) -> tuple[float, float, float, float]:
    parts = [p.strip() for p in str(roi_spec or "").split(",")]
    if len(parts) != 4:
        raise ValueError(f"invalid roi spec: {roi_spec}")
    x, y, w, h = [float(v) for v in parts]
    x = max(0.0, min(1.0, x))
    y = max(0.0, min(1.0, y))
    w = max(0.0, min(1.0 - x, w))
    h = max(0.0, min(1.0 - y, h))
    if w <= 0 or h <= 0:
        raise ValueError(f"invalid roi size: {roi_spec}")
    return x, y, w, h


def get_roi_abs(w_img: int, h_img: int, roi_rel: tuple[float, float, float, float]) -> tuple[int, int, int, int]:
    rx, ry, rw, rh = roi_rel
    x1 = int(rx * w_img)
    y1 = int(ry * h_img)
    x2 = int((rx + rw) * w_img)
    y2 = int((ry + rh) * h_img)
    x1 = max(0, min(w_img - 1, x1))
    y1 = max(0, min(h_img - 1, y1))
    x2 = max(x1 + 1, min(w_img, x2))
    y2 = max(y1 + 1, min(h_img, y2))
    return x1, y1, x2, y2


def get_parse_udp_endpoint(url: str) -> tuple[str, int]:
    m = re.match(r"^udp://([^:/?#]+):(\d+)", str(url or "").strip())
    if not m:
        raise ValueError(f"invalid udp endpoint: {url}")
    return m.group(1), int(m.group(2))


def get_udp_listen_url(url: str, fifo_size: int) -> str:
    _, port = get_parse_udp_endpoint(url)
    return f"udp://@:{port}?fifo_size={int(max(4096, fifo_size))}&overrun_nonfatal=1"


def get_udp_sender_url(url: str, pkt_size: int, buffer_size: int) -> str:
    src = str(url or "").strip()
    if not src.startswith("udp://"):
        return src
    sep = "&" if "?" in src else "?"
    pkt_size = int(max(188, pkt_size))
    buffer_size = int(max(65536, buffer_size))
    return f"{src}{sep}pkt_size={pkt_size}&buffer_size={buffer_size}"


def get_resolve_linux_ip(default_ip: str = "127.0.0.1") -> str:
    try:
        proc = subprocess.run(["hostname", "-I"], check=True, capture_output=True, text=True)
        parts = [item.strip() for item in (proc.stdout or "").split() if item.strip()]
        if parts:
            return parts[0]
    except Exception:
        pass
    return default_ip


def get_windows_host_ip(default_ip: str = "127.0.0.1") -> str:
    try:
        resolv = Path("/etc/resolv.conf")
        if resolv.exists():
            for line in resolv.read_text(encoding="utf-8", errors="ignore").splitlines():
                line = line.strip()
                if line.startswith("nameserver "):
                    parts = line.split()
                    if len(parts) >= 2:
                        return parts[1].strip()
    except Exception:
        pass

    try:
        proc = subprocess.run(["ip", "route", "show", "default"], check=True, capture_output=True, text=True)
        for line in (proc.stdout or "").splitlines():
            parts = line.strip().split()
            if len(parts) >= 3 and parts[0] == "default" and parts[1] == "via":
                return parts[2].strip()
    except Exception:
        pass

    return default_ip


def get_pick_free_udp_port(preferred_port: int, max_tries: int = 32) -> int:
    preferred_port = int(preferred_port)
    for port in range(preferred_port, preferred_port + max(1, int(max_tries))):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.bind(("0.0.0.0", port))
            return port
        except OSError:
            continue
        finally:
            try:
                sock.close()
            except Exception:
                pass

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.bind(("0.0.0.0", 0))
        return int(sock.getsockname()[1])
    finally:
        try:
            sock.close()
        except Exception:
            pass


def get_open_capture(cv2, source, timeout_ms: int):
    cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        return None

    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

    timeout_ms = int(max(500, timeout_ms))
    for prop_name in ("CAP_PROP_OPEN_TIMEOUT_MSEC", "CAP_PROP_READ_TIMEOUT_MSEC"):
        prop_id = getattr(cv2, prop_name, None)
        if prop_id is None:
            continue
        try:
            cap.set(prop_id, timeout_ms)
        except Exception:
            pass
    return cap


def get_start_ffmpeg_stream_writer(
    output_url: str,
    width: int,
    height: int,
    fps: float,
    ffmpeg_path: str,
    out_vcodec: str,
    out_bitrate: str,
) -> subprocess.Popen:
    cmd = [
        ffmpeg_path,
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{width}x{height}",
        "-r",
        f"{fps}",
        "-i",
        "-",
        "-an",
        "-c:v",
        out_vcodec,
        "-bf",
        "0",
        "-pix_fmt",
        "yuv420p",
        "-g",
        "12",
        "-b:v",
        out_bitrate,
        "-maxrate",
        out_bitrate,
        "-bufsize",
        out_bitrate,
        "-flush_packets",
        "1",
        "-muxdelay",
        "0",
        "-muxpreload",
        "0",
    ]

    # mpeg1video 不支持 low_delay 强制，mpeg2video 才启用该选项。
    if out_vcodec == "mpeg2video":
        cmd.extend(["-flags", "low_delay", "-mpegts_flags", "+resend_headers"])

    cmd.extend([
        "-f",
        "mpegts",
        output_url,
    ])

    if out_vcodec == "libx264":
        cmd = [
            ffmpeg_path,
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{width}x{height}",
            "-r",
            f"{fps}",
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-tune",
            "zerolatency",
            "-profile:v",
            "baseline",
            "-bf",
            "0",
            "-g",
            "1",
            "-keyint_min",
            "1",
            "-x264-params",
            "repeat-headers=1:aud=1:scenecut=0:sync-lookahead=0:rc-lookahead=0",
            "-pix_fmt",
            "yuv420p",
            "-b:v",
            out_bitrate,
            "-maxrate",
            out_bitrate,
            "-bufsize",
            out_bitrate,
            "-mpegts_flags",
            "+resend_headers",
            "-flush_packets",
            "1",
            "-muxdelay",
            "0",
            "-muxpreload",
            "0",
            "-f",
            "mpegts",
            output_url,
        ]

    return subprocess.Popen(cmd, stdin=subprocess.PIPE, bufsize=0)


class get_latest_frame_stream_writer:
    """后台写流器：队列仅保留最新一帧，避免输出阻塞拖慢主循环。"""

    def __init__(self, process: subprocess.Popen):
        self.process = process
        self._queue: queue.Queue[Optional[bytes]] = queue.Queue(maxsize=1)
        self._broken = False
        self._stop = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop:
            try:
                item = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if item is None:
                break

            try:
                if self.process.stdin is None:
                    self._broken = True
                    break
                self.process.stdin.write(item)
            except Exception:
                self._broken = True
                break

    def send(self, frame_bytes: bytes) -> bool:
        if self._broken:
            return False

        if self._queue.full():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass

        try:
            self._queue.put_nowait(frame_bytes)
            return True
        except queue.Full:
            return False

    def is_broken(self) -> bool:
        return self._broken

    def close(self) -> None:
        self._stop = True
        if self._queue.full():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        try:
            self._thread.join(timeout=1.0)
        except Exception:
            pass


def get_read_latest_frame(cap, max_drain: int):
    """读取一帧后继续丢弃积压帧，只保留最新画面。"""
    max_drain = max(0, int(max_drain))
    if max_drain <= 0:
        return cap.read()

    grabbed = False
    for _ in range(max_drain):
        try:
            if not cap.grab():
                break
            grabbed = True
        except Exception:
            break

    if grabbed:
        try:
            return cap.retrieve()
        except Exception:
            return False, None

    return cap.read()


def get_extract_centers(boxes: list[tuple[int, int, int, int, float, str]]) -> list[tuple[str, int, int, float]]:
    centers: list[tuple[str, int, int, float]] = []
    for x1, y1, x2, y2, conf, name in boxes:
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        centers.append((str(name), cx, cy, float(conf)))
    return centers


def get_draw_boxes(cv2, frame, boxes, line_width: int) -> None:
    for item in boxes:
        x1, y1, x2, y2, conf, name = item
        class_name = str(name).upper()
        if class_name in {"CT", "CT_HEAD"}:
            color = (255, 120, 0)  # blue in BGR
        else:
            color = (0, 165, 255)  # orange in BGR

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, max(1, line_width))
        label = f"{name} {conf:.2f}"
        cv2.putText(frame, label, (x1, max(16, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def get_class_family(name: str) -> str:
    n = str(name).upper()
    if n.startswith("CT"):
        return "CT"
    if n.startswith("T"):
        return "T"
    return "OTHER"


def get_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = float(iw * ih)
    if inter <= 0:
        return 0.0

    area_a = float(max(0, ax2 - ax1) * max(0, ay2 - ay1))
    area_b = float(max(0, bx2 - bx1) * max(0, by2 - by1))
    denom = area_a + area_b - inter
    if denom <= 0:
        return 0.0
    return inter / denom


def get_family_exclusive_boxes(
    boxes: list[tuple[int, int, int, int, float, str]],
    conflict_iou: float,
) -> list[tuple[int, int, int, int, float, str]]:
    """同一区域 CT/T 系列互斥：保留高置信度框。"""
    if not boxes:
        return boxes

    thr = max(0.0, min(1.0, float(conflict_iou)))
    kept: list[tuple[int, int, int, int, float, str]] = []

    for item in sorted(boxes, key=lambda x: float(x[4]), reverse=True):
        x1, y1, x2, y2, conf, name = item
        fam = get_class_family(name)
        if fam not in {"CT", "T"}:
            kept.append(item)
            continue

        blocked = False
        for k in kept:
            kfam = get_class_family(k[5])
            if kfam == fam or kfam == "OTHER":
                continue
            iou = get_iou((x1, y1, x2, y2), (k[0], k[1], k[2], k[3]))
            if iou >= thr:
                blocked = True
                break

        if not blocked:
            kept.append(item)

    return kept


def get_extract_boxes(result, ox: int, oy: int, names_map: dict[int, str]) -> list[tuple[int, int, int, int, float, str]]:
    out: list[tuple[int, int, int, int, float, str]] = []
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return out

    try:
        xyxy = boxes.xyxy.detach().cpu().numpy()
        conf = boxes.conf.detach().cpu().numpy()
        cls = boxes.cls.detach().cpu().numpy().astype(int)
    except Exception:
        return out

    for idx in range(len(xyxy)):
        x1, y1, x2, y2 = xyxy[idx]
        c = float(conf[idx])
        k = int(cls[idx])
        name = str(names_map.get(k, str(k)))
        out.append((int(x1) + ox, int(y1) + oy, int(x2) + ox, int(y2) + oy, c, name))
    return out


def get_run_infer_once(
    model,
    names_map: dict[int, str],
    args,
    roi_frame,
    roi_x: int,
    roi_y: int,
    inf_w: int,
    inf_h: int,
    out_w: int,
    out_h: int,
) -> list[tuple[int, int, int, int, float, str]]:
    result = model.predict(
        roi_frame,
        conf=float(args.conf),
        imgsz=int(args.imgsz),
        device=args.device,
        half=bool(args.half),
        verbose=False,
    )[0]
    infer_boxes = get_extract_boxes(result, ox=roi_x, oy=roi_y, names_map=names_map)
    return get_scale_boxes(infer_boxes, src_w=inf_w, src_h=inf_h, dst_w=out_w, dst_h=out_h)


class get_latest_infer_worker:
    """异步推理：队列仅保留最新任务，避免阻塞主循环。"""

    def __init__(self, model, names_map: dict[int, str], args, log_func=None):
        self.model = model
        self.names_map = names_map
        self.args = args
        self.log_func = log_func
        self._queue: queue.Queue[Optional[tuple]] = queue.Queue(maxsize=1)
        self._lock = threading.Lock()
        self._latest_boxes: Optional[list[tuple[int, int, int, int, float, str]]] = None
        self._latest_version: int = 0
        self._last_err_log_t = 0.0
        self._stop = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit(
        self,
        roi_frame,
        roi_x: int,
        roi_y: int,
        inf_w: int,
        inf_h: int,
        out_w: int,
        out_h: int,
    ) -> None:
        task = (roi_frame, roi_x, roi_y, inf_w, inf_h, out_w, out_h)
        if self._queue.full():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
        try:
            self._queue.put_nowait(task)
        except queue.Full:
            pass

    def _run(self) -> None:
        while not self._stop:
            try:
                task = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if task is None:
                break

            roi_frame, roi_x, roi_y, inf_w, inf_h, out_w, out_h = task
            try:
                out_boxes = get_run_infer_once(
                    model=self.model,
                    names_map=self.names_map,
                    args=self.args,
                    roi_frame=roi_frame,
                    roi_x=roi_x,
                    roi_y=roi_y,
                    inf_w=inf_w,
                    inf_h=inf_h,
                    out_w=out_w,
                    out_h=out_h,
                )
            except Exception as exc:
                now = time.monotonic()
                if self.log_func is not None and (now - self._last_err_log_t) >= 2.0:
                    self._last_err_log_t = now
                    self.log_func(f"stage=infer_async_error err={type(exc).__name__}:{exc}")
                continue

            with self._lock:
                self._latest_boxes = out_boxes
                self._latest_version += 1

    def get_latest_since(self, seen_version: int) -> tuple[int, Optional[list[tuple[int, int, int, int, float, str]]]]:
        with self._lock:
            if self._latest_boxes is None or self._latest_version <= int(seen_version):
                return self._latest_version, None
            return self._latest_version, list(self._latest_boxes)

    def close(self) -> None:
        self._stop = True
        if self._queue.full():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        try:
            self._thread.join(timeout=1.0)
        except Exception:
            pass


class get_latest_ocr_worker:
    """异步 OCR：每次仅处理最新快照，减少对主循环实时性的影响。"""

    def __init__(self, cv2_module, ocr_rois, ocr_reader, min_conf: float):
        self.cv2 = cv2_module
        self.ocr_rois = list(ocr_rois or [])
        self.ocr_reader = ocr_reader
        self.min_conf = float(min_conf)
        self._queue: queue.Queue[Optional[tuple[int, object]]] = queue.Queue(maxsize=1)
        self._lock = threading.Lock()
        self._latest_version: int = 0
        self._latest_results: list[dict] = []
        self._stop = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit(self, frame_id: int, frame_snapshot) -> None:
        task = (int(frame_id), frame_snapshot)
        if self._queue.full():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
        try:
            self._queue.put_nowait(task)
        except queue.Full:
            pass

    def _run(self) -> None:
        while not self._stop:
            try:
                task = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if task is None:
                break

            frame_id, frame_snapshot = task
            try:
                out = get_run_ocr_on_rois(
                    cv2=self.cv2,
                    frame=frame_snapshot,
                    ocr_rois=self.ocr_rois,
                    ocr_reader=self.ocr_reader,
                    min_conf=self.min_conf,
                )
            except Exception:
                out = []

            with self._lock:
                self._latest_version = int(frame_id)
                self._latest_results = out

    def get_latest_since(self, seen_version: int) -> tuple[int, Optional[list[dict]]]:
        with self._lock:
            if self._latest_version <= int(seen_version):
                return self._latest_version, None
            return self._latest_version, [dict(item) for item in self._latest_results]

    def close(self) -> None:
        self._stop = True
        if self._queue.full():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        try:
            self._thread.join(timeout=1.0)
        except Exception:
            pass


def get_build_image_data_url_from_frame(cv2, frame, roi_rel: tuple[float, float, float, float]) -> str:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = get_roi_abs(w, h, roi_rel)
    crop = frame[y1:y2, x1:x2]
    if crop is None or crop.size == 0:
        raise RuntimeError("location roi crop empty")

    ok, encoded = cv2.imencode(".jpg", crop)
    if not ok:
        raise RuntimeError("location roi encode failed")
    b64 = base64.b64encode(bytes(encoded)).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


class get_qwen_location_client:
    """Qwen 地点识别接口：负责与模型交互。"""

    def __init__(self, api_key: str, model: str):
        try:
            openai_mod = importlib.import_module("openai")
        except Exception as exc:
            raise RuntimeError("openai SDK 不可用，请安装 openai") from exc

        self.model = str(model)
        self.client = openai_mod.OpenAI(api_key=api_key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")

    def get_query_location(self, image_data_url: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "这是CS界面左上角区域截图。请直接判断当前位置，只输出地点名称，不要解释。",
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
        location_text = str((response.choices[0].message.content or "").strip())
        print(location_text or "unknown", flush=True)
        return location_text

    def get_query_location_from_frame(self, cv2, frame, roi_rel: tuple[float, float, float, float]) -> str:
        image_data_url = get_build_image_data_url_from_frame(cv2, frame, roi_rel)
        return self.get_query_location(image_data_url)

    def get_query_next_action(self, summary_text: str, image_data_url: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "你是CS战术助手，只输出下一步建议的一句话，不要输出代码，不要输出控制指令。\n"
                                f"当前状态：{summary_text}\n"
                                "请结合该原生截图内容给出下一步建议。"
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
        return str((response.choices[0].message.content or "").strip())


class get_latest_location_worker:
    """异步位置识别：处理最新 ROI 快照并在后台请求 Qwen。"""

    def __init__(self, cv2_module, location_client: get_qwen_location_client, roi_rel: tuple[float, float, float, float], log_func=None):
        self.cv2 = cv2_module
        self.location_client = location_client
        self.roi_rel = roi_rel
        self.log_func = log_func
        self._queue: queue.Queue[Optional[tuple[int, object]]] = queue.Queue(maxsize=1)
        self._lock = threading.Lock()
        self._latest_version: int = 0
        self._latest_location: str = ""
        self._last_err_log_t = 0.0
        self._stop = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit(self, frame_id: int, frame_snapshot) -> None:
        task = (int(frame_id), frame_snapshot)
        if self._queue.full():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
        try:
            self._queue.put_nowait(task)
        except queue.Full:
            pass

    def _run(self) -> None:
        while not self._stop:
            try:
                task = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if task is None:
                break

            frame_id, frame_snapshot = task
            try:
                location = self.location_client.get_query_location_from_frame(
                    cv2=self.cv2,
                    frame=frame_snapshot,
                    roi_rel=self.roi_rel,
                )
            except Exception as exc:
                now = time.monotonic()
                if self.log_func is not None and (now - self._last_err_log_t) >= 2.0:
                    self._last_err_log_t = now
                    self.log_func(f"stage=location_error err={type(exc).__name__}:{exc}")
                    print(f"location_error={type(exc).__name__}:{exc}", flush=True)
                continue

            with self._lock:
                self._latest_version = int(frame_id)
                self._latest_location = str(location or "")
            if self.log_func is not None:
                shown_location = self._latest_location or "unknown"
                self.log_func(f"stage=location_result frame_id={frame_id} location={shown_location}")
            print(f"location={self._latest_location or 'unknown'}", flush=True)

    def get_latest_since(self, seen_version: int) -> tuple[int, Optional[str]]:
        with self._lock:
            if self._latest_version <= int(seen_version):
                return self._latest_version, None
            return self._latest_version, self._latest_location

    def close(self) -> None:
        self._stop = True
        if self._queue.full():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        try:
            self._thread.join(timeout=1.0)
        except Exception:
            pass


def get_scale_boxes(
    boxes: list[tuple[int, int, int, int, float, str]],
    src_w: int,
    src_h: int,
    dst_w: int,
    dst_h: int,
) -> list[tuple[int, int, int, int, float, str]]:
    if src_w <= 0 or src_h <= 0 or dst_w <= 0 or dst_h <= 0:
        return boxes

    sx = float(dst_w) / float(src_w)
    sy = float(dst_h) / float(src_h)
    mapped: list[tuple[int, int, int, int, float, str]] = []
    for x1, y1, x2, y2, conf, name in boxes:
        nx1 = max(0, min(dst_w - 1, int(x1 * sx)))
        ny1 = max(0, min(dst_h - 1, int(y1 * sy)))
        nx2 = max(nx1 + 1, min(dst_w, int(x2 * sx)))
        ny2 = max(ny1 + 1, min(dst_h, int(y2 * sy)))
        mapped.append((nx1, ny1, nx2, ny2, conf, name))
    return mapped


def get_fit_size(src_w: int, src_h: int, max_w: int, max_h: int) -> tuple[int, int]:
    if src_w <= 0 or src_h <= 0 or max_w <= 0 or max_h <= 0:
        return src_w, src_h

    scale = min(float(max_w) / float(src_w), float(max_h) / float(src_h), 1.0)
    dst_w = max(2, int(round(src_w * scale)))
    dst_h = max(2, int(round(src_h * scale)))

    # yuv420p/libx264 要求宽高为偶数。
    if dst_w % 2 != 0:
        dst_w -= 1
    if dst_h % 2 != 0:
        dst_h -= 1

    dst_w = max(2, dst_w)
    dst_h = max(2, dst_h)
    return dst_w, dst_h


def get_default_ocr_rois() -> list[tuple[float, float, float, float]]:
    """默认 4 个 OCR 区域：金额、生命、子弹、时间。"""
    return [
        # 0) 左下：金钱
        (0.008, 0.930, 0.080, 0.060),
        # 1) 下方中：生命值数字
        (0.345, 0.924, 0.040, 0.066),
        # 2) 下方中右：当前子弹
        (0.615, 0.920, 0.040, 0.055),
        # 3) 上方中间：回合时间
        (0.482, 0.018, 0.042, 0.046),
    ]


def get_parse_ocr_rois(roi_spec: str) -> list[tuple[float, float, float, float]]:
    raw = str(roi_spec or "").strip()
    if not raw:
        return get_default_ocr_rois()

    out: list[tuple[float, float, float, float]] = []
    chunks = [c.strip() for c in raw.split(";") if c.strip()]
    for item in chunks:
        parts = [p.strip() for p in item.split(",")]
        if len(parts) != 4:
            continue
        try:
            x, y, w, h = [float(v) for v in parts]
        except Exception:
            continue
        x = max(0.0, min(1.0, x))
        y = max(0.0, min(1.0, y))
        w = max(0.0, min(1.0 - x, w))
        h = max(0.0, min(1.0 - y, h))
        if w > 0 and h > 0:
            out.append((x, y, w, h))

    if not out:
        return get_default_ocr_rois()
    return out


def get_resolve_location_model(model_name: str) -> str:
    model = str(model_name or "").strip().lower()
    if model in {"", "deepseek-chat", "deepseek-reasoner", "qwen-vl-plus"}:
        return "qwen3.6-plus"
    return str(model_name or "qwen3.6-plus").strip()


def get_ocr_reader(args):
    if not bool(args.ocr):
        return None

    engine = str(args.ocr_engine or "pytesseract").lower()
    if engine != "pytesseract":
        return None

    try:
        pytesseract = importlib.import_module("pytesseract")
    except Exception:
        return None

    whitelist = str(args.ocr_whitelist or "").strip()
    lang_default = str(args.ocr_lang or "eng").strip() or "eng"
    lang_cn = str(args.ocr_cn_lang or "chi_sim+eng").strip() or "chi_sim+eng"

    def _run_reader(img, psm: str, use_whitelist: bool, lang: str, extra_whitelist: str = ""):
        config = f"--oem 1 --psm {psm}"
        active_whitelist = extra_whitelist or whitelist
        if use_whitelist and active_whitelist:
            config += f" -c tessedit_char_whitelist={active_whitelist}"
        try:
            return pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT, config=config, lang=lang)
        except Exception:
            return None

    def _reader_default(img):
        return _run_reader(img, psm="8", use_whitelist=True, lang=lang_default)

    def _reader_time(img):
        return _run_reader(img, psm="7", use_whitelist=True, lang=lang_default, extra_whitelist="0123456789:")

    return {"default": _reader_default, "time": _reader_time}


def get_normalize_ocr_text_for_roi(roi_idx: int, text: str) -> str:
    raw = str(text or "").strip()
    if not raw:
        return ""

    if roi_idx in {0, 1, 2}:
        # 数值区域：只保留数字本身，去掉前缀符号和其它噪声。
        compact = raw.replace(" ", "")
        match = re.search(r"\d[\d,]*", compact)
        if match:
            digits_only = re.sub(r"[^\d]", "", match.group(0))
            return digits_only.lstrip("0") or "0"
        if re.search(r"\d", compact):
            digits = re.sub(r"[^\d]", "", compact)
            return digits.lstrip("0") or "0"

    if roi_idx == 3:
        compact = raw.replace(" ", "")
        if ":" in compact:
            match = re.search(r"(\d{1,2}):?(\d{2})", compact)
            if match:
                return f"{int(match.group(1))}:{match.group(2)}"
        digits = re.sub(r"[^\d]", "", compact)
        if len(digits) == 3:
            return f"{int(digits[0])}:{digits[1:]}"
        if len(digits) == 4:
            return f"{int(digits[:-2])}:{digits[-2:]}"

    return raw


def get_run_ocr_on_rois(cv2, frame, ocr_rois, ocr_reader, min_conf: float) -> list[dict]:
    if ocr_reader is None or frame is None:
        return []

    h, w = frame.shape[:2]
    out: list[dict] = []
    conf_thr = float(min_conf) * 100.0

    for idx, roi_rel in enumerate(ocr_rois):
        if isinstance(ocr_reader, dict):
            if idx == 3:
                reader_fn = ocr_reader.get("time")
            else:
                reader_fn = ocr_reader.get("default")
        else:
            reader_fn = ocr_reader

        x1, y1, x2, y2 = get_roi_abs(w, h, roi_rel)
        crop = frame[y1:y2, x1:x2]
        if crop is None or crop.size == 0:
            out.append(
                {
                    "id": idx,
                    "text": "",
                    "conf": 0.0,
                    "roi_rel": [float(roi_rel[0]), float(roi_rel[1]), float(roi_rel[2]), float(roi_rel[3])],
                    "roi_abs": [x1, y1, x2, y2],
                }
            )
            continue

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)

        # 多预处理候选：优先高置信结果；若都低于阈值则回退到最佳非空文本。
        try:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            clahe_img = clahe.apply(gray)
        except Exception:
            clahe_img = gray

        blur = cv2.GaussianBlur(clahe_img, (3, 3), 0)
        try:
            bin_otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            bin_inv = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        except Exception:
            bin_otsu = blur
            bin_inv = blur

        candidates = [blur, bin_otsu, bin_inv]
        best_text = ""
        best_conf = -1.0
        fallback_text = ""
        fallback_conf = -1.0

        for cand in candidates:
            try:
                data = reader_fn(cand) if callable(reader_fn) else None
            except Exception:
                data = None

            if not isinstance(data, dict):
                continue

            txt_list = data.get("text", [])
            conf_list = data.get("conf", [])

            for i in range(min(len(txt_list), len(conf_list))):
                txt = str(txt_list[i] or "").strip()
                if not txt:
                    continue
                try:
                    c = float(conf_list[i])
                except Exception:
                    c = -1.0

                if c > fallback_conf:
                    fallback_conf = c
                    fallback_text = txt

                if c >= conf_thr and c > best_conf:
                    best_conf = c
                    best_text = txt

        if not best_text and fallback_text:
            best_text = fallback_text
            best_conf = max(0.0, fallback_conf)

        best_text = get_normalize_ocr_text_for_roi(idx, best_text)

        out.append(
            {
                "id": idx,
                "text": best_text,
                "conf": float(best_conf / 100.0 if best_conf > 0 else 0.0),
                "roi_rel": [float(roi_rel[0]), float(roi_rel[1]), float(roi_rel[2]), float(roi_rel[3])],
                "roi_abs": [x1, y1, x2, y2],
            }
        )

    return out


def get_draw_ocr_rois(cv2, frame, ocr_rois, ocr_results: Optional[list[dict]] = None) -> None:
    if frame is None:
        return

    h, w = frame.shape[:2]

    for idx, roi_rel in enumerate(ocr_rois):
        x1, y1, x2, y2 = get_roi_abs(w, h, roi_rel)
        color = (0, 255, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)


def get_latest_ocr_results() -> list[dict]:
    """供外部模块调用：返回最近一次 OCR 结果数组副本。"""
    return [dict(item) for item in LATEST_OCR_RESULTS]


def get_latest_location_result() -> str:
    """供外部模块调用：返回最近一次 Qwen 地点识别结果。"""
    return str(LATEST_LOCATION_RESULT or "")


def get_format_ocr_results_log(ocr_results: list[dict]) -> str:
    """按紧凑格式输出 OCR 结果：ocr_results=[0:"",1:"",...]"""
    parts: list[str] = []
    for idx, item in enumerate(ocr_results):
        if isinstance(item, dict):
            rid = int(item.get("id", idx))
            text = str(item.get("text", "") or "")
        else:
            rid = idx
            text = ""
        parts.append(f'{rid}:{json.dumps(text, ensure_ascii=False)}')
    return "ocr_results=[" + ",".join(parts) + "]"


def get_resolve_qwen_api_key(explicit_key: str = "") -> str:
    key = str(explicit_key or "").strip()
    if key:
        return key
    return str(
        os.environ.get("DASHSCOPE_API_KEY")
        or os.environ.get("QWEN_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or ""
    ).strip()


class get_frame_ocr_interface:
    """供外部模块按帧触发 OCR 检测的接口。"""

    def __init__(self, args, cv2_module):
        self.cv2 = cv2_module
        self.ocr_rois = get_parse_ocr_rois(getattr(args, "ocr_roi", ""))
        self.ocr_reader = get_ocr_reader(args)
        self.min_conf = float(getattr(args, "ocr_min_conf", 0.20))

    def get_detect(self, frame) -> list[dict]:
        if frame is None or self.ocr_reader is None:
            return []
        return get_run_ocr_on_rois(
            cv2=self.cv2,
            frame=frame,
            ocr_rois=self.ocr_rois,
            ocr_reader=self.ocr_reader,
            min_conf=self.min_conf,
        )

    @staticmethod
    def get_compact_text(ocr_results: list[dict]) -> str:
        texts: list[str] = []
        for item in ocr_results:
            text = str((item or {}).get("text", "") or "").strip()
            if text:
                texts.append(text)
        return " | ".join(texts)


def get_write_shared_runtime_artifacts(
    cv2,
    frame,
    centers: list[tuple[str, int, int, float]],
    frame_path: str,
    state_path: str,
) -> None:
    """写入跨进程共享数据：最新原生帧和最新中心点结果。"""
    if frame is not None and str(frame_path or "").strip():
        frame_target = Path(str(frame_path))
        try:
            frame_target.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        # 避免 `cv2.imwrite("*.tmp")` 因无图像扩展名写失败，改为统一编码 JPEG 字节再原子替换。
        ok, encoded = cv2.imencode(".jpg", frame)
        if ok:
            tmp_frame = str(frame_target) + ".tmp"
            with open(tmp_frame, "wb") as f:
                f.write(bytes(encoded))
            os.replace(tmp_frame, str(frame_target))

    if str(state_path or "").strip():
        state_target = Path(str(state_path))
        try:
            state_target.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        payload = {
            "ts": time.time(),
            "centers": [
                {
                    "name": str(name),
                    "cx": int(cx),
                    "cy": int(cy),
                    "conf": float(conf),
                }
                for name, cx, cy, conf in (centers or [])
            ],
        }
        tmp_state = str(state_target) + ".tmp"
        with open(tmp_state, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        os.replace(tmp_state, str(state_target))


def main() -> int:
    global LATEST_CENTER_POINTS, LATEST_OCR_RESULTS, LATEST_LOCATION_RESULT
    args = get_args()
    boot_t0 = time.monotonic()
    infer_worker: Optional[get_latest_infer_worker] = None
    ocr_worker: Optional[get_latest_ocr_worker] = None
    location_worker: Optional[get_latest_location_worker] = None
    location_client: Optional[get_qwen_location_client] = None

    def _log(msg: str) -> None:
        print(f"[yolorun] {msg}", flush=True)

    source_text = (args.in_stream or args.source or "").strip()
    linux_ip = args.linux_ip
    if str(linux_ip).lower() == "auto":
        linux_ip = get_resolve_linux_ip("127.0.0.1")

    preferred_port = int(args.port)
    if not source_text:
        source_text = f"udp://{linux_ip}:{preferred_port}"

    actual_port = preferred_port
    default_source_text = f"udp://{linux_ip}:{preferred_port}"
    if source_text == default_source_text:
        actual_port = get_pick_free_udp_port(preferred_port)
        if actual_port != preferred_port:
            print(f"[yolorun] stage=port_adjusted preferred={preferred_port} actual={actual_port}", flush=True)
            source_text = f"udp://{linux_ip}:{actual_port}"

    try:
        ultralytics = importlib.import_module("ultralytics")
        YOLO = ultralytics.YOLO
    except Exception as exc:
        raise ImportError("未安装 ultralytics，请先安装。") from exc

    try:
        torch = importlib.import_module("torch")
    except Exception as exc:
        raise ImportError("未安装 torch，请先安装。") from exc

    if str(args.device).lower() != "cpu" and not bool(torch.cuda.is_available()):
        raise SystemExit("未检测到 CUDA，请改用 --device cpu 或安装 CUDA 版 torch。")

    try:
        cv2 = importlib.import_module("cv2")
    except Exception as exc:
        raise ImportError("未安装 opencv-python，请先安装。") from exc

    get_enable_torch_runtime_opt(torch)

    picked_weights = get_pick_accel_weights(args.weights, args.accel_mode)
    if picked_weights != args.weights:
        _log(f"stage=accel_weights_selected mode={args.accel_mode} weights={picked_weights}")
    else:
        _log(f"stage=accel_weights_selected mode={args.accel_mode} weights={args.weights}")

    _log(f"stage=model_loading weights={picked_weights}")
    model = YOLO(picked_weights)

    accel_mode = str(args.accel_mode or "auto").lower()
    if accel_mode == "compile":
        try:
            model.model = torch.compile(model.model, mode="reduce-overhead")
            _log("stage=accel_compile_enabled")
        except Exception:
            _log("stage=accel_compile_failed")

    if accel_mode == "trt" and not str(picked_weights).lower().endswith(".engine"):
        _log("stage=accel_trt_engine_missing using_pt_fallback")

    _log(f"stage=model_loaded elapsed={time.monotonic() - boot_t0:.2f}s")
    names_map = getattr(model, "names", {})
    force_infer_worker = str(args.infer_worker or "auto").lower()
    is_engine_weights = str(picked_weights).lower().endswith(".engine")

    if is_engine_weights and int(args.imgsz) != 640:
        _log(f"stage=engine_imgsz_override from={args.imgsz} to=640")
        args.imgsz = 640

    use_async_infer = True
    if force_infer_worker == "on":
        use_async_infer = True
    elif force_infer_worker == "off":
        use_async_infer = False

    _log(
        "stage=infer_mode "
        f"mode={'async' if use_async_infer else 'sync'} "
        f"infer_worker_arg={force_infer_worker} engine_weights={is_engine_weights}"
    )

    if use_async_infer:
        infer_worker = get_latest_infer_worker(model=model, names_map=names_map, args=args, log_func=_log)

    work_w, work_h = get_parse_size(args.work_size)
    out_max_w, out_max_h = get_parse_size(args.output_max_size)
    roi_rel = get_parse_roi(args.detect_roi)
    ocr_rois = get_parse_ocr_rois(args.ocr_roi)
    ocr_reader = get_ocr_reader(args)
    if bool(args.ocr):
        _log("stage=ocr_schedule_disabled reason=moved_to_decision_advisor")

    location_roi_rel = get_parse_roi(args.location_roi)
    location_seen_version = 0
    last_location_submit_t = 0.0
    last_location_text = ""
    print(f"location_status=default_enabled={bool(args.location_detect)}", flush=True)
    if bool(args.location_detect):
        _log("stage=location_schedule_disabled reason=moved_to_decision_advisor")
        print("location_status=disabled reason=moved_to_decision_advisor", flush=True)

    input_listen_url = get_udp_listen_url(source_text, fifo_size=int(args.udp_fifo_size))
    input_send_url = get_udp_sender_url(
        source_text,
        pkt_size=int(args.sender_udp_pkt_size),
        buffer_size=int(args.sender_udp_buffer_size),
    )

    output_url = str(args.out_stream or "").strip()
    if not output_url:
        output_url = f"udp://{get_windows_host_ip()}:2234"
        _log(f"stage=out_stream_defaulted url={output_url}")
    else:
        _log(f"stage=out_stream_configured url={output_url}")

    viewer_source = None
    if args.preview and output_url.startswith("udp://"):
        try:
            _, out_port = get_parse_udp_endpoint(output_url)
            local_viewer_url = f"udp://0.0.0.0:{out_port}?fifo_size=262144&buffer_size=262144&overrun_nonfatal=1"
            viewer_source = local_viewer_url
            _log(f"stage=win_preview_source {local_viewer_url}")
        except Exception:
            viewer_source = None

    capture_opts = (
        "fflags;nobuffer"
        f"|probesize;{int(max(1024, int(args.capture_probesize)))}"
        f"|analyzeduration;{int(max(0, int(args.capture_analyzeduration)))}"
        "|max_delay;0"
    )
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = capture_opts

    _log(f"stage=input_listen url={input_listen_url}")
    _log(f"stage=input_send url={input_send_url}")
    _log(f"stage=capture_opts {capture_opts}")

    stream_tool: Optional[OpenGameTool] = None
    viewer_tool: Optional[OpenGameTool] = None

    if not args.skip_win_stream:
        _log(
            "stage=opengame_start "
            f"linux_ip={linux_ip} port={actual_port} bitrate={args.bitrate} framerate={args.framerate}"
        )
        stream_tool = OpenGameTool(
            game_exe=str(args.game_exe),
            game_args=["-applaunch", "730"],
            linux_ip=str(linux_ip),
            port=int(actual_port),
            framerate=int(args.framerate),
            bitrate=str(args.bitrate),
            window_title=str(args.window_title),
            stream_outputs=[input_send_url],
            ffplay_path=str(args.ffplay),
        )
        stream_tool.open_game(wait_seconds=float(args.wait_game))
        p = stream_tool.start_stream(with_viewer=False)
        if p is None:
            print("[yolorun] opengame start stream failed", file=sys.stderr, flush=True)
            return 1
        _log("stage=opengame_started")
        time.sleep(0.6)

    _log("stage=capture_opening")
    cap = get_open_capture(cv2, input_listen_url, timeout_ms=int(args.capture_timeout_ms))
    if cap is None:
        print(f"[yolorun] open stream failed: {input_listen_url}", file=sys.stderr, flush=True)
        if stream_tool is not None:
            try:
                stream_tool.stop_stream()
            except Exception:
                pass
        return 1
    _log(f"stage=capture_opened elapsed={time.monotonic() - boot_t0:.2f}s")

    out_proc = None
    out_writer: Optional[get_latest_frame_stream_writer] = None
    first_frame_t0 = time.monotonic()
    first_frame_logged = False
    first_infer_logged = False
    last_boxes: list[tuple[int, int, int, int, float, str]] = []
    last_centers: list[tuple[str, int, int, float]] = []
    infer_seen_version = 0
    last_boxes_update_t = 0.0
    boxes_ttl_sec = max(0.05, float(max(1, int(args.boxes_ttl_ms))) / 1000.0)
    sync_last_err_log_t = 0.0
    frame_id = 0
    fps_t0 = time.monotonic()
    fps_count = 0
    wait_log_t0 = time.monotonic()
    out_fps = max(1.0, float(args.stream_fps))
    out_send_interval = 1.0 / out_fps
    next_out_send_t = time.monotonic()
    draw_ocr_roi = bool(args.draw_ocr_roi)
    last_ocr_results: list[dict] = []
    ocr_seen_version = 0
    shared_frame_path = str(os.environ.get("CSRL_SHARED_FRAME_PATH") or "/tmp/cs_rl_latest_frame.jpg").strip()
    shared_state_path = str(os.environ.get("CSRL_SHARED_STATE_PATH") or "/tmp/cs_rl_runtime_state.json").strip()
    shared_write_interval_sec = 0.15
    last_shared_write_t = 0.0
    _log(f"stage=shared_artifacts frame={shared_frame_path} state={shared_state_path}")
    max_frame_drain = max(0, int(args.capture_drain or args.frame_drain or 0))
    if max_frame_drain > 0:
        _log(f"stage=frame_drain_enabled max_drain={max_frame_drain}")

    try:
        while True:
            ok, frame = get_read_latest_frame(cap, max_frame_drain)
            if not ok or frame is None:
                now = time.monotonic()
                if (now - wait_log_t0) >= 5.0:
                    wait_log_t0 = now
                    _log("stage=waiting_frame no new frame yet")
                if (not first_frame_logged) and ((now - first_frame_t0) >= float(max(1.0, args.first_frame_timeout_sec))):
                    print(
                        "[yolorun] first frame timeout: input stream has no decodable frame yet.",
                        file=sys.stderr,
                        flush=True,
                    )
                    return 2
                if stream_tool is not None and float(args.capture_reconnect_sec) > 0:
                    if (now - first_frame_t0) >= float(args.capture_reconnect_sec):
                        _log("stage=opengame_restart_stream")
                        stream_tool.restart_stream(with_viewer=False, cooldown_sec=0.8)
                        first_frame_t0 = now
                time.sleep(0.002)
                continue

            if not first_frame_logged:
                first_frame_logged = True
                _log(f"stage=first_frame elapsed={time.monotonic() - boot_t0:.2f}s")

            # OCR 统一使用原始解码帧，避免后续缩放/绘制影响识别结果。
            native_ocr_frame = frame.copy()

            src_h, src_w = frame.shape[:2]
            out_w, out_h = get_fit_size(src_w, src_h, out_max_w, out_max_h)
            output_frame = frame
            if out_w != src_w or out_h != src_h:
                output_frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)

            infer_frame = output_frame
            if output_frame.shape[1] != work_w or output_frame.shape[0] != work_h:
                infer_frame = cv2.resize(output_frame, (work_w, work_h), interpolation=cv2.INTER_AREA)

            inf_h, inf_w = infer_frame.shape[:2]
            x1, y1, x2, y2 = get_roi_abs(inf_w, inf_h, roi_rel)
            roi_frame = infer_frame[y1:y2, x1:x2]
            now = time.monotonic()

            frame_id += 1
            infer_every = max(1, int(args.infer_every))
            run_infer = (frame_id % infer_every == 0) or (not last_boxes)
            latest_boxes = None

            if run_infer:
                if use_async_infer and infer_worker is not None:
                    # 异步线程必须拷贝 ROI，避免引用后续帧内存导致结果冻结或漂移。
                    infer_worker.submit(
                        roi_frame=roi_frame.copy(),
                        roi_x=x1,
                        roi_y=y1,
                        inf_w=inf_w,
                        inf_h=inf_h,
                        out_w=out_w,
                        out_h=out_h,
                    )
                else:
                    try:
                        latest_boxes = get_run_infer_once(
                            model=model,
                            names_map=names_map,
                            args=args,
                            roi_frame=roi_frame,
                            roi_x=x1,
                            roi_y=y1,
                            inf_w=inf_w,
                            inf_h=inf_h,
                            out_w=out_w,
                            out_h=out_h,
                        )
                    except Exception as exc:
                        if (now - sync_last_err_log_t) >= 2.0:
                            sync_last_err_log_t = now
                            _log(f"stage=infer_sync_error err={type(exc).__name__}:{exc}")
                        latest_boxes = None

            if use_async_infer and infer_worker is not None and latest_boxes is None:
                infer_seen_version, latest_boxes = infer_worker.get_latest_since(infer_seen_version)
            if latest_boxes is not None:
                last_boxes = get_family_exclusive_boxes(
                    latest_boxes,
                    conflict_iou=float(args.family_conflict_iou),
                )
                last_centers = get_extract_centers(last_boxes)
                LATEST_CENTER_POINTS = last_centers
                last_boxes_update_t = now
                if not first_infer_logged:
                    first_infer_logged = True
                    _log(f"stage=first_infer elapsed={time.monotonic() - boot_t0:.2f}s boxes={len(last_boxes)}")
            else:
                # 推理结果长时间未刷新时清空旧框，避免首帧框长期残留。
                if last_boxes and (now - last_boxes_update_t) > boxes_ttl_sec:
                    last_boxes = []
                    last_centers = []
                LATEST_CENTER_POINTS = last_centers

            if (now - last_shared_write_t) >= shared_write_interval_sec:
                last_shared_write_t = now
                try:
                    get_write_shared_runtime_artifacts(
                        cv2=cv2,
                        frame=native_ocr_frame,
                        centers=last_centers,
                        frame_path=shared_frame_path,
                        state_path=shared_state_path,
                    )
                except Exception:
                    pass

            roi_x1, roi_y1, roi_x2, roi_y2 = get_roi_abs(out_w, out_h, roi_rel)
            cv2.rectangle(output_frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (120, 120, 120), 1)
            get_draw_boxes(cv2, output_frame, last_boxes, line_width=int(args.line_width))

            if draw_ocr_roi:
                get_draw_ocr_rois(cv2, output_frame, ocr_rois, ocr_results=last_ocr_results)

            # 定时 OCR 与定时位置识别已迁移到 decision_advisor.py 的终端命令模式。

            if out_writer is not None and out_writer.is_broken():
                _log("stage=out_stream_broken_restart")
                try:
                    out_writer.close()
                except Exception:
                    pass
                out_writer = None
                if out_proc is not None:
                    try:
                        if out_proc.stdin is not None:
                            out_proc.stdin.close()
                    except Exception:
                        pass
                    try:
                        out_proc.terminate()
                    except Exception:
                        pass
                    out_proc = None

            if out_proc is None:
                out_proc = get_start_ffmpeg_stream_writer(
                    output_url=output_url,
                    width=out_w,
                    height=out_h,
                    fps=float(args.stream_fps),
                    ffmpeg_path=args.ffmpeg,
                    out_vcodec=args.out_vcodec,
                    out_bitrate=args.out_bitrate,
                )
                out_writer = get_latest_frame_stream_writer(out_proc)
                _log(f"stage=out_stream_started url={output_url}")

                if args.preview and stream_tool is not None:
                    try:
                        viewer_tool = OpenGameTool(
                            game_exe=str(args.game_exe),
                            ffplay_path=str(args.ffplay),
                            viewer_source=viewer_source,
                        )
                        if viewer_tool.start_windows_viewer():
                            _log(f"stage=win_preview_started url={viewer_source}")
                        else:
                            _log("stage=win_preview_failed")
                    except Exception:
                        _log("stage=win_preview_failed")

            if out_writer is not None:
                send_t = time.monotonic()
                if send_t >= next_out_send_t:
                    if not out_writer.send(output_frame.tobytes()) and out_writer.is_broken():
                        _log("stage=out_stream_broken_restart")
                        try:
                            out_writer.close()
                        except Exception:
                            pass
                        out_writer = None
                        if out_proc is not None:
                            try:
                                if out_proc.stdin is not None:
                                    out_proc.stdin.close()
                            except Exception:
                                pass
                            try:
                                out_proc.terminate()
                            except Exception:
                                pass
                            out_proc = None

                    while next_out_send_t <= send_t:
                        next_out_send_t += out_send_interval

            fps_count += 1
            now = time.monotonic()
            if (now - fps_t0) >= 1.0:
                fps = fps_count / max(1e-6, (now - fps_t0))
                _log(f"fps={fps:.1f} boxes={len(last_boxes)} infer_every={infer_every}")
                fps_t0 = now
                fps_count = 0

        return 0
    except KeyboardInterrupt:
        return 130
    finally:
        if infer_worker is not None:
            try:
                infer_worker.close()
            except Exception:
                pass

        if ocr_worker is not None:
            try:
                ocr_worker.close()
            except Exception:
                pass

        if location_worker is not None:
            try:
                location_worker.close()
            except Exception:
                pass

        try:
            cap.release()
        except Exception:
            pass

        if out_writer is not None:
            try:
                out_writer.close()
            except Exception:
                pass

        if out_proc is not None:
            try:
                if out_proc.stdin is not None:
                    out_proc.stdin.close()
            except Exception:
                pass
            try:
                out_proc.terminate()
            except Exception:
                pass

        if viewer_tool is not None:
            try:
                viewer_tool.stop_windows_viewer()
            except Exception:
                pass

        if stream_tool is not None:
            try:
                stream_tool.stop_stream()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
