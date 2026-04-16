"""YOLO 实时流管线：输入流 -> YOLO -> 输出流，并仅保留 Windows 侧观测窗口。"""

from __future__ import annotations

import argparse
import importlib
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
    parser.add_argument("--imgsz", type=int, default=96, help="推理尺寸")
    parser.add_argument("--device", type=str, default="0", help="推理设备")
    parser.add_argument("--half", action="store_true", help="半精度推理")
    parser.add_argument("--infer-every", type=int, default=4, help="每 N 帧推理一次")

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
    parser.add_argument("--ocr-min-conf", type=float, default=0.20, help="兼容参数")
    parser.add_argument("--ocr-whitelist", type=str, default="0123456789/%:HPARMOABULLET", help="兼容参数")
    parser.add_argument("--ocr-info-jsonl", type=str, default="", help="兼容参数")

    parser.add_argument("--out-vcodec", type=str, default="mpeg2video", help="输出编码器")
    parser.add_argument("--out-bitrate", type=str, default="4000k", help="输出码率")
    parser.add_argument("--ffmpeg", type=str, default="ffmpeg", help="ffmpeg 可执行文件")
    parser.add_argument("--ffplay", type=str, default="ffplay", help="Windows 侧 ffplay 可执行文件")
    return parser.parse_args()


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
        "-pix_fmt",
        "yuv420p",
        "-g",
        "12",
        "-b:v",
        out_bitrate,
        "-f",
        "mpegts",
        output_url,
    ]

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


class get_latest_infer_worker:
    """异步推理：队列仅保留最新任务，避免阻塞主循环。"""

    def __init__(self, model, names_map: dict[int, str], args):
        self.model = model
        self.names_map = names_map
        self.args = args
        self._queue: queue.Queue[Optional[tuple]] = queue.Queue(maxsize=1)
        self._lock = threading.Lock()
        self._latest_boxes: Optional[list[tuple[int, int, int, int, float, str]]] = None
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
                result = self.model.predict(
                    roi_frame,
                    conf=float(self.args.conf),
                    imgsz=int(self.args.imgsz),
                    device=self.args.device,
                    half=bool(self.args.half),
                    verbose=False,
                )[0]
                infer_boxes = get_extract_boxes(result, ox=roi_x, oy=roi_y, names_map=self.names_map)
                out_boxes = get_scale_boxes(infer_boxes, src_w=inf_w, src_h=inf_h, dst_w=out_w, dst_h=out_h)
            except Exception:
                continue

            with self._lock:
                self._latest_boxes = out_boxes

    def get_latest(self) -> Optional[list[tuple[int, int, int, int, float, str]]]:
        with self._lock:
            if self._latest_boxes is None:
                return None
            return list(self._latest_boxes)

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


def main() -> int:
    global LATEST_CENTER_POINTS
    args = get_args()
    boot_t0 = time.monotonic()
    infer_worker: Optional[get_latest_infer_worker] = None

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

    _log(f"stage=model_loading weights={args.weights}")
    model = YOLO(args.weights)
    _log(f"stage=model_loaded elapsed={time.monotonic() - boot_t0:.2f}s")
    names_map = getattr(model, "names", {})
    infer_worker = get_latest_infer_worker(model=model, names_map=names_map, args=args)

    work_w, work_h = get_parse_size(args.work_size)
    out_max_w, out_max_h = get_parse_size(args.output_max_size)
    roi_rel = get_parse_roi(args.detect_roi)

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
            local_viewer_url = f"udp://0.0.0.0:{out_port}?fifo_size=16384&buffer_size=16384&overrun_nonfatal=1"
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
    frame_id = 0
    fps_t0 = time.monotonic()
    fps_count = 0
    wait_log_t0 = time.monotonic()
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

            frame_id += 1
            infer_every = max(1, int(args.infer_every))
            run_infer = (frame_id % infer_every == 0) or (not last_boxes)

            if run_infer:
                infer_worker.submit(
                    roi_frame=roi_frame,
                    roi_x=x1,
                    roi_y=y1,
                    inf_w=inf_w,
                    inf_h=inf_h,
                    out_w=out_w,
                    out_h=out_h,
                )

            latest_boxes = infer_worker.get_latest()
            if latest_boxes is not None:
                last_boxes = latest_boxes
                last_centers = get_extract_centers(last_boxes)
                LATEST_CENTER_POINTS = last_centers
                if not first_infer_logged:
                    first_infer_logged = True
                    _log(f"stage=first_infer elapsed={time.monotonic() - boot_t0:.2f}s boxes={len(last_boxes)}")
            else:
                LATEST_CENTER_POINTS = last_centers

            roi_x1, roi_y1, roi_x2, roi_y2 = get_roi_abs(out_w, out_h, roi_rel)
            cv2.rectangle(output_frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (120, 120, 120), 1)
            get_draw_boxes(cv2, output_frame, last_boxes, line_width=int(args.line_width))

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
