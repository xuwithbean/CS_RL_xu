"""OCR HUD 信息识别逻辑（血量/护甲/弹药等）。"""

from __future__ import annotations

import argparse
import csv
import importlib
import re
import subprocess
from pathlib import Path
from typing import Any


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".m4v"}
STREAM_PREFIXES = ("udp://", "rtsp://", "rtmp://", "http://", "https://")


def get_clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def get_parse_roi_specs(roi_specs: list[str]) -> list[tuple[float, float, float, float]]:
    rois: list[tuple[float, float, float, float]] = []
    for spec in roi_specs:
        parts = [p.strip() for p in (spec or "").split(",")]
        if len(parts) != 4:
            continue
        try:
            x, y, w, h = [float(v) for v in parts]
        except Exception:
            continue
        x = get_clip(x, 0.0, 1.0)
        y = get_clip(y, 0.0, 1.0)
        w = get_clip(w, 0.0, 1.0 - x)
        h = get_clip(h, 0.0, 1.0 - y)
        if w > 0 and h > 0:
            rois.append((x, y, w, h))

    if not rois:
        # 默认取左下 HUD 区域，通常包含血量/护甲/弹药信息。
        rois.append((0.00, 0.78, 0.42, 0.22))
    return rois


def get_preprocess_for_ocr(img_bgr: Any, cv2: Any) -> Any:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return th


def get_extract_numbers(text: str) -> list[int]:
    nums = re.findall(r"\d{1,3}", text or "")
    out = []
    for n in nums:
        try:
            out.append(int(n))
        except Exception:
            pass
    return out


def get_ocr_text(
    roi_img: Any,
    *,
    cv2: Any,
    engine: str,
    ocr_obj: Any,
    min_conf: float,
    whitelist: str,
) -> str:
    proc = get_preprocess_for_ocr(roi_img, cv2)

    if engine == "easyocr":
        results = ocr_obj.readtext(proc, detail=1, paragraph=False)
        kept = []
        for item in results:
            if len(item) < 3:
                continue
            txt = str(item[1]).strip()
            conf = float(item[2])
            if txt and conf >= float(min_conf):
                kept.append(txt)
        return " ".join(kept).strip()

    config = (
        "--oem 1 --psm 6 "
        f"-c tessedit_char_whitelist={whitelist} "
        "-c preserve_interword_spaces=1"
    )
    txt = ocr_obj.image_to_string(proc, config=config)
    return (txt or "").strip()


def get_run_ocr_on_frame(
    *,
    img: Any,
    src_path: str,
    frame_id: int,
    rois: list[tuple[float, float, float, float]],
    engine: str,
    ocr_obj: Any,
    min_conf: float,
    whitelist: str,
    cv2: Any,
) -> tuple[list[list[Any]], list[str]]:
    """执行 OCR，并将 ROI 与文本画到图像上。"""
    h_img, w_img = img.shape[:2]
    rows: list[list[Any]] = []
    ocr_lines: list[str] = []

    for roi_id, (rx, ry, rw, rh) in enumerate(rois):
        x1r = int(rx * w_img)
        y1r = int(ry * h_img)
        x2r = int((rx + rw) * w_img)
        y2r = int((ry + rh) * h_img)
        x1r = int(get_clip(x1r, 0, w_img - 1))
        y1r = int(get_clip(y1r, 0, h_img - 1))
        x2r = int(get_clip(x2r, x1r + 1, w_img))
        y2r = int(get_clip(y2r, y1r + 1, h_img))

        roi_img = img[y1r:y2r, x1r:x2r]
        text = get_ocr_text(
            roi_img,
            cv2=cv2,
            engine=engine,
            ocr_obj=ocr_obj,
            min_conf=float(min_conf),
            whitelist=str(whitelist),
        )
        numbers = get_extract_numbers(text)
        hp_guess = numbers[0] if numbers else -1

        rows.append(
            [
                src_path,
                frame_id,
                roi_id,
                text,
                "|".join(str(n) for n in numbers),
                hp_guess,
            ]
        )

        cv2.rectangle(img, (x1r, y1r), (x2r, y2r), (255, 255, 0), 2)
        short_text = text[:28] if text else "-"
        cv2.putText(
            img,
            f"OCR[{roi_id}] {short_text}",
            (x1r, max(18, y1r - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 0),
            1,
        )
        if hp_guess >= 0:
            ocr_lines.append(f"R{roi_id}:HP~{hp_guess}")

    return rows, ocr_lines


def get_is_video_like(source: str) -> bool:
    if source.isdigit():
        return True
    return Path(source).suffix.lower() in VIDEO_SUFFIXES


def get_is_stream_like(source: str) -> bool:
    return source.lower().startswith(STREAM_PREFIXES)


def get_is_image_like(source: str) -> bool:
    return Path(source).suffix.lower() in IMAGE_SUFFIXES


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
    parser = argparse.ArgumentParser(description="OCR only: HUD text/number recognition")
    parser.add_argument("--source", type=str, required=True, help="输入源：图片/视频/摄像头/流")
    parser.add_argument("--project", type=str, default="visual_recognition/runs", help="输出目录")
    parser.add_argument("--name", type=str, default="ocr_only", help="输出实验名")
    parser.add_argument("--show", action="store_true", help="是否实时显示")
    parser.add_argument("--save-video", action="store_true", help="强制保存连续视频")
    parser.add_argument("--fps", type=float, default=30.0, help="输出视频帧率")
    parser.add_argument("--stream-fps", type=float, default=30.0, help="输出流帧率")
    parser.add_argument("--out-stream", type=str, default="", help="实时输出流地址，如 udp://127.0.0.1:2234")
    parser.add_argument(
        "--ocr-engine",
        type=str,
        default="pytesseract",
        choices=["easyocr", "pytesseract"],
        help="OCR 引擎",
    )
    parser.add_argument("--ocr-roi", action="append", default=[], help="OCR 区域 x,y,w,h，可重复")
    parser.add_argument(
        "--ocr-whitelist",
        type=str,
        default="0123456789/%:HPARMOABULLET",
        help="OCR 白名单字符（pytesseract 使用）",
    )
    parser.add_argument("--ocr-min-conf", type=float, default=0.20, help="OCR 最低置信度（easyocr 使用）")
    parser.add_argument("--device", type=str, default="0", help="设备：0 或 cpu（easyocr 用）")
    return parser.parse_args()


def main() -> None:
    args = get_args()

    try:
        cv2 = importlib.import_module("cv2")
    except ImportError as exc:
        raise ImportError("未安装 opencv-python。请先执行: pip install opencv-python") from exc

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

    rois = get_parse_roi_specs(args.ocr_roi)
    out_dir = get_output_dir(args.project, args.name)
    ocr_csv_path = out_dir / "ocr_hud.csv"

    is_video_like = get_is_video_like(args.source)
    is_stream_like = get_is_stream_like(args.source)
    should_write_video = bool(args.save_video or is_video_like or is_stream_like)

    video_writer = None
    ffmpeg_stream_proc = None

    with ocr_csv_path.open("w", newline="", encoding="utf-8") as f_ocr:
        writer = csv.writer(f_ocr)
        writer.writerow(["source", "frame_id", "roi_id", "raw_text", "numbers", "hp_guess"])

        if get_is_image_like(args.source):
            img = cv2.imread(args.source)
            if img is None:
                raise RuntimeError(f"无法读取图片: {args.source}")

            rows, lines = get_run_ocr_on_frame(
                img=img,
                src_path=args.source,
                frame_id=1,
                rois=rois,
                engine=args.ocr_engine,
                ocr_obj=ocr_obj,
                min_conf=float(args.ocr_min_conf),
                whitelist=str(args.ocr_whitelist),
                cv2=cv2,
            )
            for row in rows:
                writer.writerow(row)
            if lines:
                cv2.putText(img, " | ".join(lines), (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            out_img = out_dir / Path(args.source).name
            cv2.imwrite(str(out_img), img)
            if args.show:
                cv2.imshow("ocr_only", img)
                cv2.waitKey(0)
        else:
            source_arg: Any = int(args.source) if args.source.isdigit() else args.source
            cap = cv2.VideoCapture(source_arg)
            if not cap.isOpened():
                raise RuntimeError(f"无法打开视频/流: {args.source}")

            frame_id = 0
            while True:
                ok, img = cap.read()
                if not ok or img is None:
                    break
                frame_id += 1
                h_img, w_img = img.shape[:2]

                if should_write_video and video_writer is None:
                    out_video = out_dir / "ocr_realtime.mkv"
                    fourcc = cv2.VideoWriter_fourcc(*"XVID")
                    video_writer = cv2.VideoWriter(str(out_video), fourcc, float(args.fps), (w_img, h_img))

                if args.out_stream and ffmpeg_stream_proc is None:
                    ffmpeg_stream_proc = get_start_ffmpeg_stream_writer(
                        output_url=args.out_stream,
                        width=w_img,
                        height=h_img,
                        fps=float(args.stream_fps),
                    )

                rows, lines = get_run_ocr_on_frame(
                    img=img,
                    src_path=args.source,
                    frame_id=frame_id,
                    rois=rois,
                    engine=args.ocr_engine,
                    ocr_obj=ocr_obj,
                    min_conf=float(args.ocr_min_conf),
                    whitelist=str(args.ocr_whitelist),
                    cv2=cv2,
                )
                for row in rows:
                    writer.writerow(row)
                if lines:
                    cv2.putText(img, " | ".join(lines), (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                if should_write_video and video_writer is not None:
                    video_writer.write(img)
                else:
                    out_img = out_dir / f"frame_{frame_id:06d}.jpg"
                    cv2.imwrite(str(out_img), img)

                if ffmpeg_stream_proc is not None and ffmpeg_stream_proc.stdin is not None:
                    try:
                        ffmpeg_stream_proc.stdin.write(img.tobytes())
                    except (BrokenPipeError, OSError):
                        ffmpeg_stream_proc = None

                if args.show:
                    cv2.imshow("ocr_only", img)
                    key = cv2.waitKey(1) & 0xFF
                    if key == 27:
                        break

            cap.release()

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

    print(f"OCR 识别完成，输出目录: {out_dir}")
    print(f"OCR CSV: {ocr_csv_path}")
    if args.out_stream:
        print(f"实时输出流: {args.out_stream}")


if __name__ == "__main__":
    main()
