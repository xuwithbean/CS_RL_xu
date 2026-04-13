"""批量裁剪图片。

支持：
- 单张图片裁剪
- 目录批量裁剪（可递归）
- 使用相对 ROI 坐标进行裁剪，适合统一裁掉 HUD/边缘区域
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch crop images by ROI")
    parser.add_argument("--input", required=True, help="输入图片或图片目录")
    parser.add_argument("--output", required=True, help="输出目录")
    parser.add_argument(
        "--roi",
        type=str,
        default="0.00,0.08,1.00,0.84",
        help="裁剪区域（相对坐标 x,y,w,h），默认适合裁掉上下 HUD",
    )
    parser.add_argument("--recursive", action="store_true", help="递归处理子目录")
    parser.add_argument("--preserve-tree", action="store_true", help="输出时保留输入目录结构")
    parser.add_argument("--suffix", type=str, default="_crop", help="输出文件名后缀")
    parser.add_argument("--overwrite", action="store_true", help="允许覆盖同名输出文件")
    return parser.parse_args()


def parse_roi(roi: str) -> tuple[float, float, float, float]:
    parts = [p.strip() for p in str(roi or "").split(",")]
    if len(parts) != 4:
        raise ValueError(f"invalid roi spec: {roi}")
    x, y, w, h = [float(v) for v in parts]
    x = max(0.0, min(1.0, x))
    y = max(0.0, min(1.0, y))
    w = max(0.0, min(1.0 - x, w))
    h = max(0.0, min(1.0 - y, h))
    if w <= 0 or h <= 0:
        raise ValueError(f"invalid roi size: {roi}")
    return x, y, w, h


def get_image_files(root: Path, recursive: bool) -> list[Path]:
    if root.is_file():
        return [root]
    if not root.is_dir():
        raise FileNotFoundError(f"input not found: {root}")

    if recursive:
        files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]
    else:
        files = [p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]
    return sorted(files)


def crop_image(img, roi_rel: tuple[float, float, float, float], cv2):
    h, w = img.shape[:2]
    rx, ry, rw, rh = roi_rel
    x1 = int(rx * w)
    y1 = int(ry * h)
    x2 = int((rx + rw) * w)
    y2 = int((ry + rh) * h)
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(x1 + 1, min(w, x2))
    y2 = max(y1 + 1, min(h, y2))
    return img[y1:y2, x1:x2], (x1, y1, x2, y2)


def build_output_path(src: Path, input_root: Path, output_root: Path, suffix: str, preserve_tree: bool) -> Path:
    if preserve_tree and input_root.is_dir():
        rel = src.relative_to(input_root)
        return output_root / rel.parent / f"{rel.stem}{suffix}{rel.suffix}"
    return output_root / f"{src.stem}{suffix}{src.suffix}"


def main() -> int:
    args = get_args()

    try:
        cv2 = __import__("cv2")
    except ImportError as exc:
        raise ImportError("未安装 opencv-python。请先执行: pip install opencv-python") from exc

    input_root = Path(args.input).expanduser().resolve()
    output_root = Path(args.output).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    roi_rel = parse_roi(args.roi)
    files = get_image_files(input_root, args.recursive)
    if not files:
        print(f"[crop] no images found in {input_root}")
        return 1

    print(f"[crop] input={input_root}")
    print(f"[crop] output={output_root}")
    print(f"[crop] roi={roi_rel}")
    print(f"[crop] files={len(files)}")

    done = 0
    for src in files:
        dst = build_output_path(src, input_root, output_root, args.suffix, args.preserve_tree)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() and not args.overwrite:
            print(f"[crop] skip exists: {dst}")
            continue

        img = cv2.imread(str(src))
        if img is None:
            print(f"[crop] failed to read: {src}")
            continue

        cropped, rect = crop_image(img, roi_rel, cv2)
        if cropped.size == 0:
            print(f"[crop] empty crop: {src}")
            continue

        if not cv2.imwrite(str(dst), cropped):
            print(f"[crop] failed to write: {dst}")
            continue

        done += 1
        print(f"[crop] {src.name} -> {dst.name} rect={rect}")

    print(f"[crop] done={done}/{len(files)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
