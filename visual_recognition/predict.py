# [x]: 使用 yolo和ocr等技术对画面中的人物进行识别和位置信息判定以及读取图片中的数字等信息

from __future__ import annotations

import argparse
import csv
import importlib
import json
import subprocess
from pathlib import Path

try:
    from visual_recognition.ocrr import get_parse_roi_specs as get_parse_ocr_rois
    from visual_recognition.ocrr import get_run_ocr_on_frame
    from visual_recognition.yolor import get_draw_yolo_and_rows
except Exception:
    from ocrr import get_parse_roi_specs as get_parse_ocr_rois
    from ocrr import get_run_ocr_on_frame
    from yolor import get_draw_yolo_and_rows


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".m4v"}
STREAM_PREFIXES = ("udp://", "rtsp://", "rtmp://", "http://", "https://")


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict CT/T and visualize person+head with centers")
    parser.add_argument("--weights", type=str, required=True, help="模型权重路径，例如 best.pt")
    parser.add_argument("--source", type=str, required=True, help="输入源：图片/视频/目录/摄像头")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    parser.add_argument("--imgsz", type=int, default=640, help="推理尺寸")
    parser.add_argument("--device", type=str, default="0", help="设备：0 或 cpu")
    parser.add_argument("--project", type=str, default="visual_recognition/runs", help="输出目录")
    parser.add_argument("--name", type=str, default="ct_t_predict_person", help="输出实验名")
    parser.add_argument("--head-ratio", type=float, default=0.30, help="头部高度占身体框比例")
    parser.add_argument("--head-width-ratio", type=float, default=0.45, help="头部宽度占身体框比例")
    parser.add_argument("--line-width", type=int, default=2, help="绘图线宽")
    parser.add_argument("--fps", type=float, default=30.0, help="视频源输出帧率（摄像头/未知时使用）")
    parser.add_argument("--show", action="store_true", help="是否实时显示窗口")
    parser.add_argument("--save-video", action="store_true", help="强制保存连续带框视频")
    parser.add_argument("--stream-fps", type=float, default=30.0, help="输出流帧率")
    parser.add_argument("--out-stream", type=str, default="", help="实时输出流地址，如 udp://127.0.0.1:2234")
    parser.add_argument("--ocr", action="store_true", help="启用 OCR（识别血量/护甲/弹药等 HUD 文本）")
    parser.add_argument(
        "--ocr-engine",
        type=str,
        default="pytesseract",
        choices=["easyocr", "pytesseract"],
        help="OCR 引擎：easyocr 或 pytesseract",
    )
    parser.add_argument(
        "--ocr-roi",
        action="append",
        default=[],
        help=(
            "OCR 区域，格式: x,y,w,h（0~1 相对坐标，可重复传多个）。"
            "例如左下 HUD: --ocr-roi 0.00,0.78,0.42,0.22"
        ),
    )
    parser.add_argument(
        "--ocr-whitelist",
        type=str,
        default="0123456789/%:HPARMOABULLET",
        help="OCR 白名单字符（pytesseract 使用）",
    )
    parser.add_argument("--ocr-min-conf", type=float, default=0.20, help="OCR 最低置信度（easyocr 使用）")
    parser.add_argument(
        "--ocr-info-jsonl",
        type=str,
        default="",
        help="按帧输出 OCR 信息的 JSONL 文件路径（默认输出到运行目录）",
    )
    parser.add_argument("--print-ocr", action="store_true", help="实时打印每帧 OCR 信息")
    return parser.parse_args()


def get_is_video_like(source: str) -> bool:
    if source.isdigit():
        return True
    suffix = Path(source).suffix.lower()
    return suffix in VIDEO_SUFFIXES


def get_is_stream_like(source: str) -> bool:
    source_lower = source.lower()
    return source_lower.startswith(STREAM_PREFIXES)


def get_is_image_like(path: str) -> bool:
    return Path(path).suffix.lower() in IMAGE_SUFFIXES


def get_output_dir(project: str, name: str) -> Path:
    out_dir = Path(project) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def get_start_ffmpeg_stream_writer(
    output_url: str,
    width: int,
    height: int,
    fps: float,
) -> subprocess.Popen:
    """启动 ffmpeg 进程，将原始 BGR 帧实时编码并推送到输出流。"""
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

    ocr_obj = None
    ocr_rois = get_parse_ocr_rois(args.ocr_roi)
    if args.ocr:
        if args.ocr_engine == "easyocr":
            try:
                easyocr = importlib.import_module("easyocr")
            except ImportError as exc:
                raise ImportError("未安装 easyocr。请先执行: pip install easyocr") from exc
            ocr_obj = easyocr.Reader(["en"], gpu=(str(args.device).lower() != "cpu"))
        else:
            try:
                ocr_obj = importlib.import_module("pytesseract")
            except ImportError as exc:
                raise ImportError("未安装 pytesseract。请先执行: pip install pytesseract") from exc

    model = YOLO(args.weights)

    out_dir = get_output_dir(args.project, args.name)
    csv_path = out_dir / "detections_centers.csv"
    ocr_csv_path = out_dir / "ocr_hud.csv"
    is_video_like = get_is_video_like(args.source)
    is_stream_like = get_is_stream_like(args.source)

    # 流输入场景默认输出 YOLO 带框流，便于下游直接消费。
    if is_stream_like and not args.out_stream:
        args.out_stream = "udp://127.0.0.1:2234"
        print(f"[predict] 检测到流输入，自动启用 out-stream: {args.out_stream}")

    should_write_video = bool(args.save_video or is_video_like or is_stream_like)
    video_writer = None
    ffmpeg_stream_proc = None

    ocr_jsonl_path = Path(args.ocr_info_jsonl) if args.ocr_info_jsonl else (out_dir / "ocr_info.jsonl")

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

        with ocr_csv_path.open("w", newline="", encoding="utf-8") as f_ocr:
            ocr_jsonl = ocr_jsonl_path.open("w", encoding="utf-8") if args.ocr else None
            ocr_writer = csv.writer(f_ocr)
            ocr_writer.writerow(["source", "frame_id", "roi_id", "raw_text", "numbers", "hp_guess"])

            frame_id = 0
            for result in results:
                frame_id += 1
                img = result.orig_img.copy()
                h_img, w_img = img.shape[:2]
                src_path = str(result.path)

                if args.ocr and ocr_obj is not None:
                    ocr_rows, ocr_lines = get_run_ocr_on_frame(
                        img=img,
                        src_path=src_path,
                        frame_id=frame_id,
                        rois=ocr_rois,
                        engine=args.ocr_engine,
                        ocr_obj=ocr_obj,
                        min_conf=float(args.ocr_min_conf),
                        whitelist=str(args.ocr_whitelist),
                        cv2=cv2,
                    )
                    for row in ocr_rows:
                        ocr_writer.writerow(row)

                    if ocr_jsonl is not None:
                        roi_items = []
                        for row in ocr_rows:
                            num_values = []
                            nums_str = str(row[4] or "")
                            if nums_str:
                                for n in nums_str.split("|"):
                                    if n.strip():
                                        try:
                                            num_values.append(int(n.strip()))
                                        except Exception:
                                            pass
                            roi_items.append(
                                {
                                    "roi_id": int(row[2]),
                                    "raw_text": str(row[3]),
                                    "numbers": num_values,
                                    "hp_guess": int(row[5]),
                                }
                            )

                        payload = {
                            "source": src_path,
                            "frame_id": frame_id,
                            "ocr": roi_items,
                            "summary": ocr_lines,
                        }
                        ocr_jsonl.write(json.dumps(payload, ensure_ascii=False) + "\n")
                        ocr_jsonl.flush()

                        if args.print_ocr:
                            print(f"[ocr][frame {frame_id}] {payload['summary']}")

                    if ocr_lines:
                        cv2.putText(
                            img,
                            " | ".join(ocr_lines),
                            (10, 28),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 255),
                            2,
                        )

                if should_write_video:
                    if video_writer is None:
                        # MKV 在长时间运行中更稳，更适合边写边看。
                        out_video = out_dir / "predicted_realtime.mkv"
                        fourcc = cv2.VideoWriter_fourcc(*"XVID")
                        video_writer = cv2.VideoWriter(str(out_video), fourcc, float(args.fps), (w_img, h_img))

                if args.out_stream and ffmpeg_stream_proc is None:
                    ffmpeg_stream_proc = get_start_ffmpeg_stream_writer(
                        output_url=args.out_stream,
                        width=w_img,
                        height=h_img,
                        fps=float(args.stream_fps),
                    )

                boxes = result.boxes
                yolo_rows = get_draw_yolo_and_rows(
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
                for row in yolo_rows:
                    writer.writerow([src_path, frame_id] + row)

                if should_write_video:
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
                        # 输出端断开后不中断主检测流程。
                        ffmpeg_stream_proc = None

                if args.show:
                    cv2.imshow("person ct/t with head + ocr", img)
                    key = cv2.waitKey(1) & 0xFF
                    if key == 27:
                        break

            if ocr_jsonl is not None:
                ocr_jsonl.close()

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

    print(f"推理完成。可视化输出目录: {out_dir}")
    print(f"中心坐标 CSV: {csv_path}")
    if args.ocr:
        print(f"OCR CSV: {ocr_csv_path}")
        print(f"OCR JSONL: {ocr_jsonl_path}")
    if args.out_stream:
        print(f"实时输出流: {args.out_stream}")


if __name__ == "__main__":
    main()
