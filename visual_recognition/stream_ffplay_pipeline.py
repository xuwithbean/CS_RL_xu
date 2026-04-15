"""实时管道：Windows 推流 -> OpenCV 读流 -> YOLO 识别 -> ffmpeg 输出 -> ffplay 预览。

这个脚本不改动现有 `predict.py` 的行为，而是单独处理视频流输入，
避免 Ultralytics 自己去解析 udp:// 作为数据源时出现的 FileNotFoundError。
"""

from __future__ import annotations

import argparse
import importlib
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from opengame import OpenGameTool

try:
    from visual_recognition.yolor import get_draw_yolo_and_rows
except Exception:
    from yolor import get_draw_yolo_and_rows


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Realtime YOLO stream pipeline with ffplay preview")
    parser.add_argument("--weights", type=str, required=True, help="YOLO 权重路径")
    parser.add_argument("--source", type=str, default="", help="输入流地址，例如 udp://192.168.x.x:12345")
    parser.add_argument("--in-stream", type=str, default="", help="输入流地址别名，兼容旧启动脚本")
    parser.add_argument("--out-stream", type=str, default="", help="输出流地址（为空时自动发到 Windows 主机）")
    parser.add_argument("--preview", action="store_true", help="使用 ffplay 预览输出流")
    parser.add_argument("--show", action="store_true", help="是否在 Python 进程内显示窗口")
    parser.add_argument("--skip-win-stream", action="store_true", help="跳过 Windows 推流，适用于已有输入流")
    parser.add_argument("--game-exe", type=str, default=r"E:\steam\steamapps\common\Counter-Strike Global Offensive\game\bin\win64\cs2.exe", help="Windows 游戏可执行路径")
    parser.add_argument("--window-title", type=str, default="auto", help="游戏窗口标题")
    parser.add_argument("--wait-game", type=float, default=6.0, help="启动游戏后等待秒数")
    parser.add_argument("--linux-ip", type=str, default="auto", help="WSL 接收地址")
    parser.add_argument("--port", type=int, default=12345, help="原始流端口")
    parser.add_argument("--framerate", type=int, default=60, help="原始推流帧率")
    parser.add_argument("--bitrate", type=str, default="2500k", help="原始推流码率")
    parser.add_argument("--conf", type=float, default=0.30, help="检测置信度")
    parser.add_argument("--imgsz", type=int, default=256, help="推理尺寸")
    parser.add_argument("--device", type=str, default="0", help="设备：0 或 cpu")
    parser.add_argument("--project", type=str, default="visual_recognition/runs", help="输出目录")
    parser.add_argument("--name", type=str, default="ct_t_yolo_ffplay", help="运行名")
    parser.add_argument("--head-ratio", type=float, default=0.30, help="头部高度占身体框比例")
    parser.add_argument("--head-width-ratio", type=float, default=0.45, help="头部宽度占身体框比例")
    parser.add_argument("--line-width", type=int, default=2, help="绘图线宽")
    parser.add_argument("--detect-roi", type=str, default="0.00,0.08,1.00,0.84", help="YOLO 识别区域")
    parser.add_argument("--work-size", type=str, default="704x396", help="检测处理分辨率，例如 704x396")
    parser.add_argument("--preview-size", type=str, default="800x450", help="预览窗口分辨率，例如 800x450")
    parser.add_argument("--stream-fps", type=float, default=60.0, help="输出流帧率")
    parser.add_argument("--frame-drain", type=int, default=4, help="每轮最多丢弃的积压帧数，越大越低延迟")
    parser.add_argument("--infer-every", type=int, default=1, help="每 N 帧跑一次推理，其余帧复用最近结果")
    parser.add_argument("--half", action="store_true", help="使用 FP16 半精度推理（需 GPU 支持）")
    parser.add_argument("--udp-fifo-size", type=int, default=262144, help="UDP 接收缓存大小，越小延迟越低")
    parser.add_argument("--ffmpeg", type=str, default="ffmpeg", help="ffmpeg 可执行文件")
    parser.add_argument("--ffplay", type=str, default="ffplay", help="ffplay 可执行文件（Linux 或 Windows 均可）")
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


def get_parse_single_roi(roi_spec: str) -> tuple[float, float, float, float]:
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


def get_parse_input_source(source: str):
    source = str(source or "").strip()
    if source.isdigit():
        return int(source)
    return source


def parse_udp_endpoint(url: str) -> tuple[str, int]:
    """从 udp://ip:port URL 解析出 ip 与端口。"""
    m = re.match(r"^udp://([^:/?#]+):(\d+)", (url or "").strip())
    if not m:
        raise ValueError(f"invalid udp endpoint: {url}")
    return m.group(1), int(m.group(2))


def get_udp_listen_url(url: str, fifo_size: int) -> str:
    """把 udp://host:port 转换为本地监听用的 udp://@:port。"""
    _, port = parse_udp_endpoint(url)
    return f"udp://@:{port}?fifo_size={int(max(4096, fifo_size))}&overrun_nonfatal=1"


def get_windows_host_ip() -> str:
    """获取 WSL 中可访问到的 Windows 主机 IP。"""
    resolv_conf = Path("/etc/resolv.conf")
    try:
        if resolv_conf.exists():
            for line in resolv_conf.read_text(encoding="utf-8", errors="ignore").splitlines():
                line = line.strip()
                if line.startswith("nameserver "):
                    parts = line.split()
                    if len(parts) >= 2:
                        return parts[1].strip()
    except Exception:
        pass

    try:
        proc = subprocess.run(["sh", "-lc", "ip route | awk '/default/ {print $3; exit}'"], check=True, capture_output=True, text=True)
        ip = (proc.stdout or "").strip()
        if ip:
            return ip
    except Exception:
        pass

    return "127.0.0.1"


def get_build_udp_url(host: str, port: int) -> str:
    return f"udp://{host}:{int(port)}"


def get_udp_listen_url_from_port(port: int, fifo_size: int) -> str:
    return f"udp://@:{int(port)}?fifo_size={int(max(4096, fifo_size))}&overrun_nonfatal=1"


def get_start_windows_ffplay(ffplay_path: str, input_url: str) -> bool:
    """在 Windows 桌面启动 ffplay 预览。"""
    ffplay_exe = str(ffplay_path or "").replace("\\\\", "\\").strip()
    win_args = [
        "-x",
        "800",
        "-y",
        "450",
        "-fflags",
        "nobuffer",
        "-flags",
        "low_delay",
        "-framedrop",
        "-probesize",
        "32",
        "-analyzeduration",
        "0",
        input_url,
    ]
    if not ffplay_exe:
        ffplay_exe = "ffplay.exe"

    ffplay_exe_ps = ffplay_exe.replace("'", "''")
    args_ps = ",".join("'" + a.replace("'", "''") + "'" for a in win_args)
    ps = (
        f"$pref='{ffplay_exe_ps}'; "
        "$exe=''; "
        "if ($pref -and (Test-Path $pref)) { $exe=$pref } "
        "if (-not $exe) { "
        "  $cmd=Get-Command ffplay.exe -ErrorAction SilentlyContinue; "
        "  if ($cmd -and $cmd.Source) { $exe=$cmd.Source } "
        "} "
        "if (-not $exe) { Write-Output '__ERR__NOEXE__'; exit 2 }; "
        f"$args=@({args_ps}); "
        "Start-Process -FilePath $exe -ArgumentList $args -WindowStyle Normal; "
        "Start-Sleep -Milliseconds 500; "
        "if (Get-Process -Name ffplay -ErrorAction SilentlyContinue) { Write-Output ('__OK__|' + $exe) } else { Write-Output ('__ERR__NOPROC__|' + $exe) }"
    )

    proc = subprocess.run(
        ["powershell.exe", "-NoProfile", "-Command", ps],
        check=False,
        capture_output=True,
        text=True,
    )
    out = (proc.stdout or "").strip()
    err = (proc.stderr or "").strip()
    if out.startswith("__OK__|"):
        exe_used = out.split("|", 1)[1]
        print(f"[yolo_ffplay] winffplay started: {exe_used}")
        return True
    if out.startswith("__ERR__NOEXE__"):
        print(f"[yolo_ffplay] winffplay failed: ffplay not found (preferred={ffplay_path})", file=sys.stderr)
        return False
    if out.startswith("__ERR__NOPROC__|"):
        exe_used = out.split("|", 1)[1]
        print(f"[yolo_ffplay] winffplay failed to stay running: {exe_used}", file=sys.stderr)
    if err:
        print(f"[yolo_ffplay] winffplay stderr: {err}", file=sys.stderr)
    return False


def get_start_ffmpeg_stream_writer(output_url: str, width: int, height: int, fps: float, ffmpeg_path: str) -> subprocess.Popen:
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
        "-g",
        "30",
        "-keyint_min",
        "30",
        "-x264-params",
        "repeat-headers=1:scenecut=0",
        "-muxdelay",
        "0",
        "-muxpreload",
        "0",
        "-f",
        "mpegts",
        output_url,
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, bufsize=0)


def get_start_ffplay_raw_preview(ffplay_path: str, width: int, height: int, fps: float) -> subprocess.Popen:
    """启动本机 ffplay，直接从 stdin 读取原始 BGR 帧进行预览。"""
    cmd = [
        ffplay_path,
        "-x",
        "800",
        "-y",
        "450",
        "-fflags",
        "nobuffer",
        "-flags",
        "low_delay",
        "-framedrop",
        "-f",
        "rawvideo",
        "-pixel_format",
        "bgr24",
        "-video_size",
        f"{width}x{height}",
        "-framerate",
        f"{fps}",
        "-i",
        "-",
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, bufsize=0)


def get_start_ffplay_preview(ffplay_path: str, input_url: str) -> subprocess.Popen:
    cmd = [
        ffplay_path,
        "-x",
        "800",
        "-y",
        "450",
        "-fflags",
        "nobuffer",
        "-flags",
        "low_delay",
        "-framedrop",
        "-probesize",
        "32",
        "-analyzeduration",
        "0",
        input_url,
    ]
    return subprocess.Popen(cmd)


def get_make_status_frame(cv2, width: int, height: int, message: str):
    import numpy as np

    frame = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(frame, message, (24, max(36, height // 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
    return frame


def main() -> int:
    args = get_args()

    try:
        ultralytics = importlib.import_module("ultralytics")
        YOLO = ultralytics.YOLO
    except ImportError as exc:
        raise ImportError("未安装 ultralytics。请先执行: pip install ultralytics") from exc

    try:
        torch = importlib.import_module("torch")
    except ImportError as exc:
        raise ImportError("未安装 torch。请先安装带 CUDA 的 torch 版本") from exc

    if str(args.device).lower() != "cpu":
        if not bool(torch.cuda.is_available()):
            raise SystemExit("当前未检测到可用 CUDA。请安装 CUDA 版 torch，或检查 GPU 驱动。")
        try:
            torch.backends.cudnn.benchmark = True
        except Exception:
            pass

    try:
        cv2 = importlib.import_module("cv2")
    except ImportError as exc:
        raise ImportError("未安装 opencv-python。请先执行: pip install opencv-python") from exc

    model = YOLO(args.weights)
    work_w, work_h = get_parse_size(args.work_size)
    preview_w, preview_h = get_parse_size(args.preview_size)
    detect_roi_rel = get_parse_single_roi(args.detect_roi)
    source_text = args.source or args.in_stream
    if not source_text:
        raise SystemExit("需要提供 --source 或 --in-stream")
    source_url = get_udp_listen_url(source_text, int(args.udp_fifo_size)) if source_text.startswith("udp://") else source_text
    source = get_parse_input_source(source_url)

    stream_tool = None
    preview_started = False
    preview_proc = None
    ffmpeg_stream_proc = None
    cap = None
    cap_thread = None
    cap_lock = None
    cap_state = None

    windows_host_ip = get_windows_host_ip()
    send_url = args.out_stream.strip() or ""
    preview_url = get_udp_listen_url_from_port(2234, int(args.udp_fifo_size)) if send_url else ""

    try:
        if not args.skip_win_stream:
            in_ip = args.linux_ip
            if in_ip == "auto":
                try:
                    proc = subprocess.run(["hostname", "-I"], check=True, capture_output=True, text=True)
                    parts = [item.strip() for item in proc.stdout.split() if item.strip()]
                    if parts:
                        in_ip = parts[0]
                except Exception:
                    in_ip = "127.0.0.1"

            stream_tool = OpenGameTool(
                game_exe=args.game_exe,
                game_args=["-applaunch", "730"],
                linux_ip=in_ip,
                port=int(args.port),
                framerate=int(args.framerate),
                bitrate=str(args.bitrate),
                window_title=args.window_title,
                stream_outputs=[source_text],
            )
            stream_tool.open_game(wait_seconds=float(args.wait_game))
            proc = stream_tool.start_stream(with_viewer=False)
            if proc is None:
                print("[yolo_ffplay] Windows 推流启动失败。", file=sys.stderr)
                return 1
            print(f"[yolo_ffplay] Windows 推流已启动 -> {source_text}")
            time.sleep(0.8)

        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "fflags;nobuffer|flags;low_delay|probesize;32|analyzeduration;0|max_delay;0"
        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            print(f"[yolo_ffplay] 无法打开输入流: {source_url}", file=sys.stderr)
            return 1

        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        cap_lock = threading.Lock()
        cap_state = {"running": True, "seq": 0, "frame": None}
        def _capture_worker() -> None:
            while bool(cap_state["running"]):
                ok_read, frame_read = cap.read()
                if not ok_read or frame_read is None:
                    time.sleep(0.001)
                    continue
                with cap_lock:
                    cap_state["frame"] = frame_read
                    cap_state["seq"] = int(cap_state["seq"]) + 1

        cap_thread = threading.Thread(target=_capture_worker, daemon=True)
        cap_thread.start()

        if args.show:
            try:
                cv2.namedWindow("yolo_ffplay", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("yolo_ffplay", preview_w, preview_h)
                cv2.imshow("yolo_ffplay", get_make_status_frame(cv2, preview_w, preview_h, "waiting for frames..."))
                cv2.waitKey(1)
            except Exception:
                pass

        frame_id = 0
        last_seq = 0
        last_result = None
        last_yolo_rows: list[list[str]] = []
        while True:
            with cap_lock:
                cur_seq = int(cap_state["seq"])
                frame = cap_state["frame"]

            if frame is None or cur_seq <= last_seq:
                if args.show:
                    try:
                        cv2.imshow("yolo_ffplay", get_make_status_frame(cv2, preview_w, preview_h, "waiting for frames..."))
                        cv2.waitKey(1)
                    except Exception:
                        pass
                time.sleep(0.001)
                continue
            last_seq = cur_seq
            frame = frame.copy()

            if frame is None:
                continue
            if frame.shape[1] != work_w or frame.shape[0] != work_h:
                frame = cv2.resize(frame, (work_w, work_h), interpolation=cv2.INTER_AREA)

            frame_id += 1
            h_img, w_img = frame.shape[:2]
            detect_roi_abs = get_roi_abs(w_img, h_img, detect_roi_rel)
            drx1, dry1, drx2, dry2 = detect_roi_abs
            cv2.rectangle(frame, (drx1, dry1), (drx2, dry2), (180, 180, 0), 1)

            infer_every = max(1, int(args.infer_every))
            run_infer = (frame_id % infer_every == 0) or (last_result is None)
            if run_infer:
                last_result = model.predict(
                    frame,
                    conf=float(args.conf),
                    imgsz=int(args.imgsz),
                    device=args.device,
                    half=bool(args.half),
                    verbose=False,
                )[0]

            if last_result is not None:
                yolo_rows = get_draw_yolo_and_rows(
                    result=last_result,
                    img=frame,
                    w_img=w_img,
                    h_img=h_img,
                    model_names=model.names,
                    head_ratio=float(args.head_ratio),
                    head_width_ratio=float(args.head_width_ratio),
                    line_width=int(args.line_width),
                    cv2=cv2,
                    detect_roi_abs=detect_roi_abs,
                )
                last_yolo_rows = yolo_rows
            else:
                yolo_rows = last_yolo_rows

            if send_url and ffmpeg_stream_proc is None:
                ffmpeg_stream_proc = get_start_ffmpeg_stream_writer(
                    output_url=send_url,
                    width=w_img,
                    height=h_img,
                    fps=float(args.stream_fps),
                    ffmpeg_path=args.ffmpeg,
                )

            if ffmpeg_stream_proc is not None and ffmpeg_stream_proc.stdin is not None:
                try:
                    ffmpeg_stream_proc.stdin.write(frame.tobytes())
                except (BrokenPipeError, OSError):
                    ffmpeg_stream_proc = None

            if args.preview and preview_proc is None:
                try:
                    preview_proc = get_start_ffplay_raw_preview(
                        ffplay_path=args.ffplay,
                        width=w_img,
                        height=h_img,
                        fps=float(args.stream_fps),
                    )
                    preview_started = True
                    print("[yolo_ffplay] ffplay 原始帧预览已启动")
                except Exception as exc:
                    print(f"[yolo_ffplay] 本机 ffplay 启动失败: {exc}", file=sys.stderr)
                    if get_start_windows_ffplay(args.ffplay, preview_url):
                        preview_started = True
                        print(f"[yolo_ffplay] Windows ffplay 预览已启动 -> {preview_url}")
                    else:
                        print("[yolo_ffplay] ffplay 启动失败，继续仅输出流", file=sys.stderr)

            if preview_proc is not None and preview_proc.stdin is not None:
                try:
                    preview_proc.stdin.write(frame.tobytes())
                except (BrokenPipeError, OSError):
                    preview_proc = None

            if args.show:
                try:
                    cv2.namedWindow("yolo_ffplay", cv2.WINDOW_NORMAL)
                except Exception:
                    pass
                cv2.imshow("yolo_ffplay", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    break

            if frame_id % 60 == 0:
                print(f"[yolo_ffplay] processed frame {frame_id}, detections={len(yolo_rows)}")

        return 0
    except KeyboardInterrupt:
        return 130
    finally:
        if cap_state is not None:
            try:
                cap_state["running"] = False
            except Exception:
                pass
        if cap_thread is not None:
            try:
                cap_thread.join(timeout=1.0)
            except Exception:
                pass
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
        if args.show:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        if ffmpeg_stream_proc is not None:
            try:
                if ffmpeg_stream_proc.stdin is not None:
                    ffmpeg_stream_proc.stdin.close()
            except Exception:
                pass
            try:
                ffmpeg_stream_proc.terminate()
            except Exception:
                pass
        if preview_proc is not None:
            try:
                if preview_proc.stdin is not None:
                    preview_proc.stdin.close()
            except Exception:
                pass
            try:
                preview_proc.terminate()
            except Exception:
                pass
        if preview_started:
            try:
                subprocess.run(
                    [
                        "powershell.exe",
                        "-NoProfile",
                        "-Command",
                        "Get-Process -Name ffplay -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.Id -Force }",
                    ],
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except Exception:
                pass
        if stream_tool is not None:
            try:
                stream_tool.stop_stream()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())