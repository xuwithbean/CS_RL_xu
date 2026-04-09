"""YOLO 人体检测与头部估计逻辑。"""

from __future__ import annotations

import argparse
import csv
import importlib
import subprocess
from pathlib import Path
from typing import Any


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".m4v"}
STREAM_PREFIXES = ("udp://", "rtsp://", "rtmp://", "http://", "https://")


def get_class_name(names: Any, cls_idx: int) -> str:
    if isinstance(names, dict):
        return str(names.get(cls_idx, cls_idx))
    if isinstance(names, list) and 0 <= cls_idx < len(names):
        return str(names[cls_idx])
    return str(cls_idx)


def get_color(sub_type: str) -> tuple[int, int, int]:
    # BGR
    if sub_type.upper() == "CT":
        return (255, 120, 40)
    if sub_type.upper() == "T":
        return (40, 180, 255)
    return (0, 255, 0)


def get_clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def get_head_box(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    head_ratio: float,
    head_width_ratio: float,
) -> tuple[float, float, float, float]:
    body_w = max(1.0, x2 - x1)
    body_h = max(1.0, y2 - y1)
    cx = x1 + body_w * 0.5
    head_w = body_w * head_width_ratio
    head_h = body_h * head_ratio
    hx1 = cx - head_w * 0.5
    hy1 = y1
    hx2 = cx + head_w * 0.5
    hy2 = y1 + head_h
    return hx1, hy1, hx2, hy2


def get_parse_main_and_sub(class_name: str) -> tuple[str, str]:
    """将类别名归一化到主类(person/head)与子类(CT/T)。"""
    raw = str(class_name or "").strip().upper()
    token = raw.replace("-", "_").replace(" ", "_")

    if token in {"CT", "COUNTER_TERRORIST", "COUNTERTERRORIST"}:
        return "person", "CT"
    if token in {"T", "TERRORIST"}:
        return "person", "T"

    if token in {"CT_HEAD", "HEAD_CT", "CTHEAD"}:
        return "head", "CT"
    if token in {"T_HEAD", "HEAD_T", "THEAD"}:
        return "head", "T"

    # 未知类别默认按 person 显示，避免中断流程。
    return "person", raw or "UNKNOWN"


def get_draw_yolo_and_rows(
    *,
    result: Any,
    img: Any,
    w_img: int,
    h_img: int,
    model_names: Any,
    head_ratio: float,
    head_width_ratio: float,
    line_width: int,
    cv2: Any,
) -> list[list[str]]:
    """在图像上绘制检测框，并返回用于 CSV 的行。"""
    rows: list[list[str]] = []
    boxes = result.boxes
    if boxes is None:
        return rows

    for det_id, box in enumerate(boxes):
        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
        conf = float(box.conf[0].item()) if box.conf is not None else 0.0
        cls_idx = int(box.cls[0].item()) if box.cls is not None else -1

        class_name = get_class_name(model_names, cls_idx)
        main_label, sub_type = get_parse_main_and_sub(class_name)
        body_cx = (x1 + x2) * 0.5
        body_cy = (y1 + y2) * 0.5

        if main_label == "head":
            # 对显式头部类别，直接使用检测框本身作为 head 信息。
            hx1, hy1, hx2, hy2 = x1, y1, x2, y2
        else:
            # 两类模式下从身体框几何估计头部框。
            hx1, hy1, hx2, hy2 = get_head_box(
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                head_ratio=float(head_ratio),
                head_width_ratio=float(head_width_ratio),
            )
        hx1 = get_clip(hx1, 0.0, w_img - 1.0)
        hy1 = get_clip(hy1, 0.0, h_img - 1.0)
        hx2 = get_clip(hx2, 0.0, w_img - 1.0)
        hy2 = get_clip(hy2, 0.0, h_img - 1.0)
        head_cx = (hx1 + hx2) * 0.5
        head_cy = (hy1 + hy2) * 0.5

        color = get_color(sub_type)
        cv2.rectangle(
            img,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            color,
            int(line_width),
        )
        cv2.rectangle(
            img,
            (int(hx1), int(hy1)),
            (int(hx2), int(hy2)),
            (0, 255, 255),
            int(line_width),
        )

        cv2.circle(img, (int(body_cx), int(body_cy)), 4, (0, 255, 0), -1)
        cv2.circle(img, (int(head_cx), int(head_cy)), 4, (255, 255, 0), -1)

        label = f"{main_label} {sub_type} {conf:.2f}"
        coord_text = f"B({int(body_cx)},{int(body_cy)}) H({int(head_cx)},{int(head_cy)})"
        cv2.putText(
            img,
            label,
            (int(x1), max(20, int(y1) - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
        )
        cv2.putText(
            img,
            coord_text,
            (int(x1), min(h_img - 10, int(y2) + 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

        rows.append(
            [
                str(det_id),
                main_label,
                sub_type,
                f"{conf:.4f}",
                f"{body_cx:.2f}",
                f"{body_cy:.2f}",
                f"{head_cx:.2f}",
                f"{head_cy:.2f}",
                f"{x1:.2f}",
                f"{y1:.2f}",
                f"{x2:.2f}",
                f"{y2:.2f}",
                f"{hx1:.2f}",
                f"{hy1:.2f}",
                f"{hx2:.2f}",
                f"{hy2:.2f}",
            ]
        )

    return rows


def get_is_video_like(source: str) -> bool:
    if source.isdigit():
        return True
    return Path(source).suffix.lower() in VIDEO_SUFFIXES


def get_is_stream_like(source: str) -> bool:
    return source.lower().startswith(STREAM_PREFIXES)


def get_is_image_like(path: str) -> bool:
    return Path(path).suffix.lower() in IMAGE_SUFFIXES


def get_output_dir(project: str, name: str) -> Path:
    out_dir = Path(project) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def get_start_ffmpeg_stream_writer(output_url: str, width: int, height: int, fps: float) -> subprocess.Popen:
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-re",
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
        "-f",
        "mpegts",
        output_url,
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO only: person + CT/T + head")
    parser.add_argument("--weights", type=str, required=True, help="模型权重路径")
    parser.add_argument("--source", type=str, required=True, help="输入源：图片/视频/目录/摄像头/流")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    parser.add_argument("--imgsz", type=int, default=640, help="推理尺寸")
    parser.add_argument("--device", type=str, default="0", help="设备：0 或 cpu")
    parser.add_argument("--project", type=str, default="visual_recognition/runs", help="输出目录")
    parser.add_argument("--name", type=str, default="yolo_only", help="输出实验名")
    parser.add_argument("--head-ratio", type=float, default=0.30, help="头部高度占身体框比例")
    parser.add_argument("--head-width-ratio", type=float, default=0.45, help="头部宽度占身体框比例")
    parser.add_argument("--line-width", type=int, default=2, help="绘图线宽")
    parser.add_argument("--fps", type=float, default=30.0, help="输出视频帧率")
    parser.add_argument("--show", action="store_true", help="是否实时显示")
    parser.add_argument("--save-video", action="store_true", help="强制保存连续视频")
    parser.add_argument("--stream-fps", type=float, default=30.0, help="输出流帧率")
    parser.add_argument("--out-stream", type=str, default="", help="实时输出流地址，如 udp://127.0.0.1:2234")
    return parser.parse_args()


def main() -> None:
    args = get_args()

    try:
        ultralytics = importlib.import_module("ultralytics")
        YOLO = ultralytics.YOLO
    except ImportError as exc:
        raise ImportError("未安装 ultralytics。请先执行: pip install ultralytics") from exc

    try:
        cv2 = importlib.import_module("cv2")
    except ImportError as exc:
        raise ImportError("未安装 opencv-python。请先执行: pip install opencv-python") from exc

    model = YOLO(args.weights)

    out_dir = get_output_dir(args.project, args.name)
    csv_path = out_dir / "yolo_centers.csv"
    is_video_like = get_is_video_like(args.source)
    is_stream_like = get_is_stream_like(args.source)
    should_write_video = bool(args.save_video or is_video_like or is_stream_like)
    video_writer = None
    ffmpeg_stream_proc = None

    results = model.predict(
        source=args.source,
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
        stream=True,
        save=False,
        verbose=False,
    )

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "source",
                "frame_id",
                "det_id",
                "label",
                "sub_type",
                "confidence",
                "body_cx",
                "body_cy",
                "head_cx",
                "head_cy",
                "body_x1",
                "body_y1",
                "body_x2",
                "body_y2",
                "head_x1",
                "head_y1",
                "head_x2",
                "head_y2",
            ]
        )

        frame_id = 0
        for result in results:
            frame_id += 1
            img = result.orig_img.copy()
            h_img, w_img = img.shape[:2]
            src_path = str(result.path)

            if should_write_video and video_writer is None:
                out_video = out_dir / "yolo_realtime.mkv"
                fourcc = cv2.VideoWriter_fourcc(*"XVID")
                video_writer = cv2.VideoWriter(str(out_video), fourcc, float(args.fps), (w_img, h_img))

            if args.out_stream and ffmpeg_stream_proc is None:
                ffmpeg_stream_proc = get_start_ffmpeg_stream_writer(
                    output_url=args.out_stream,
                    width=w_img,
                    height=h_img,
                    fps=float(args.stream_fps),
                )

            rows = get_draw_yolo_and_rows(
                result=result,
                img=img,
                w_img=w_img,
                h_img=h_img,
                model_names=model.names,
                head_ratio=float(args.head_ratio),
                head_width_ratio=float(args.head_width_ratio),
                line_width=int(args.line_width),
                cv2=cv2,
            )
            for row in rows:
                writer.writerow([src_path, frame_id] + row)

            if should_write_video and video_writer is not None:
                video_writer.write(img)
            else:
                if get_is_image_like(src_path):
                    out_img = out_dir / Path(src_path).name
                else:
                    out_img = out_dir / f"frame_{frame_id:06d}.jpg"
                cv2.imwrite(str(out_img), img)

            if ffmpeg_stream_proc is not None and ffmpeg_stream_proc.stdin is not None:
                try:
                    ffmpeg_stream_proc.stdin.write(img.tobytes())
                except (BrokenPipeError, OSError):
                    ffmpeg_stream_proc = None

            if args.show:
                cv2.imshow("yolo_only", img)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    break

    if video_writer is not None:
        video_writer.release()
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
    if args.show:
        cv2.destroyAllWindows()

    print(f"YOLO 识别完成，输出目录: {out_dir}")
    print(f"YOLO CSV: {csv_path}")
    if args.out_stream:
        print(f"实时输出流: {args.out_stream}")


if __name__ == "__main__":
    main()
