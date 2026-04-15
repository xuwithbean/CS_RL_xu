# [x]: 实现并优化 yolo 检测训练。
"""训练 YOLO 检测模型用于 CT/T（可选 CT_HEAD/T_HEAD）识别。

数据集需采用 YOLO 检测格式：
- images/train, images/val, images/test（可选）
- labels/train, labels/val, labels/test（可选）

标签类别约定：
- 两类模式：0: CT, 1: T
- 四类模式：0: CT, 1: T, 2: CT_HEAD, 3: T_HEAD
"""

from __future__ import annotations

import argparse
import importlib
import tempfile
from pathlib import Path
from typing import Any

import yaml


def get_patch_ultralytics_imread_if_needed(force_patch: bool = False) -> None:
	"""在 OpenCV 读图异常时，给 Ultralytics 注入 PIL 读图兜底。"""
	np = importlib.import_module("numpy")
	Image = importlib.import_module("PIL.Image")
	cv2 = importlib.import_module("cv2")

	need_patch = bool(force_patch)
	if not need_patch:
		try:
			probe = np.zeros((8, 8, 3), dtype=np.uint8)
			encoded = cv2.imencode(".png", probe)[1]
			_ = cv2.imdecode(encoded, 1)
			_ = cv2.resize(probe, (16, 16))
		except Exception:
			need_patch = True

	if not need_patch:
		return

	def _pil_bgr_imread(path: str, flags: int = 1):
		img = Image.open(path)
		if flags == 0:
			return np.array(img.convert("L"))
		return np.array(img.convert("RGB"))[:, :, ::-1]

	up = importlib.import_module("ultralytics.utils.patches")
	udb = importlib.import_module("ultralytics.data.base")
	up.imread = _pil_bgr_imread
	udb.imread = _pil_bgr_imread
	print("检测到 OpenCV 读图异常，已启用 PIL 读图兜底补丁。")


def get_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Train YOLO for CT/T detection")
	parser.add_argument("--data", type=str, default="visual_recognition/data_ct_t.yaml", help="数据集 yaml 路径")
	parser.add_argument("--model", type=str, default="yolo11n.pt", help="初始模型权重")
	parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
	parser.add_argument("--imgsz", type=int, default=640, help="输入尺寸")
	parser.add_argument("--batch", type=int, default=16, help="batch size")
	parser.add_argument("--device", type=str, default="0", help="设备，例：0/cpu")
	parser.add_argument("--project", type=str, default="visual_recognition/runs", help="输出目录")
	parser.add_argument("--name", type=str, default="ct_t_yolo", help="实验名")
	parser.add_argument("--workers", type=int, default=4, help="dataloader workers")
	parser.add_argument("--patience", type=int, default=50, help="早停 patience")
	parser.add_argument("--seed", type=int, default=42, help="随机种子")
	parser.add_argument("--cache", type=str, default="false", choices=["false", "ram", "disk"], help="是否缓存图像")
	parser.add_argument("--amp", action="store_true", help="启用 AMP 混合精度（默认关闭）")
	parser.add_argument("--exist-ok", action="store_true", help="允许复用已有实验目录")
	parser.add_argument("--force-pil-imread", action="store_true", help="强制启用 PIL 读图兜底")
	return parser.parse_args()


def get_check_dataset_yaml(data_yaml: Path) -> None:
	if not data_yaml.exists():
		raise FileNotFoundError(
			f"未找到数据配置文件: {data_yaml}\n"
			"请先参考 visual_recognition/data_ct_t.yaml 准备数据集。"
		)

	with data_yaml.open("r", encoding="utf-8") as f:
		cfg = yaml.safe_load(f)

	names: Any = cfg.get("names", None)
	if isinstance(names, dict):
		ordered = list(names.values())
	elif isinstance(names, list):
		ordered = names
	else:
		raise ValueError("数据集 yaml 缺少 names 字段，或格式错误。")

	norm = [str(x).strip().upper() for x in ordered]
	allowed_2 = ["CT", "T"]
	allowed_4 = ["CT", "T", "CT_HEAD", "T_HEAD"]
	if norm not in (allowed_2, allowed_4):
		raise ValueError(
			f"类别配置应为 {allowed_2} 或 {allowed_4}，当前为 {ordered}。\n"
			"请确认标签编号：0->CT, 1->T，（可选）2->CT_HEAD, 3->T_HEAD。"
		)

	raw_path = cfg.get("path", None)
	if raw_path and not Path(str(raw_path)).is_absolute():
		path_obj = Path(str(raw_path))
		candidates = [
			(data_yaml.parent / path_obj).resolve(),
			(data_yaml.parent.parent / path_obj).resolve(),
			(Path.cwd() / path_obj).resolve(),
		]
		if not any(p.exists() for p in candidates):
			raise FileNotFoundError(
				f"数据集根目录不存在: {path_obj}\n"
				f"已尝试: {', '.join(str(p) for p in candidates)}\n"
				"请先准备数据集目录，或在 yolorun.sh 中切换到 smoke 数据集。"
			)


def get_resolved_dataset_yaml(data_yaml: Path) -> Path:
	"""生成临时 yaml，将数据集 path 规范为绝对路径。"""
	with data_yaml.open("r", encoding="utf-8") as f:
		cfg = yaml.safe_load(f)

	raw_path = cfg.get("path", None)
	if raw_path:
		path_obj = Path(str(raw_path))
		if path_obj.is_absolute():
			resolved_path = path_obj
		else:
			candidates = [
				(data_yaml.parent / path_obj).resolve(),
				(data_yaml.parent.parent / path_obj).resolve(),
				(Path.cwd() / path_obj).resolve(),
			]
			resolved_path = next((p for p in candidates if p.exists()), candidates[1])
		cfg["path"] = str(resolved_path)

	with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8") as f:
		yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=False)
		return Path(f.name)


def main() -> None:
	args = get_args()
	data_yaml = Path(args.data).resolve()
	get_check_dataset_yaml(data_yaml)
	resolved_data_yaml = get_resolved_dataset_yaml(data_yaml)

	# 延迟导入，方便在未安装依赖时给出更清晰报错。
	try:
		ultralytics = importlib.import_module("ultralytics")
		YOLO = ultralytics.YOLO
	except ImportError as exc:
		raise ImportError(
			"未安装 ultralytics。请先执行: pip install ultralytics"
		) from exc

	get_patch_ultralytics_imread_if_needed(force_patch=args.force_pil_imread)

	model = YOLO(args.model)
	train_kwargs = dict(
		data=str(resolved_data_yaml),
		epochs=args.epochs,
		imgsz=args.imgsz,
		batch=args.batch,
		device=args.device,
		project=str(Path(args.project).resolve()),
		name=args.name,
		workers=args.workers,
		patience=args.patience,
		seed=args.seed,
		cache=(False if args.cache == "false" else args.cache),
		amp=args.amp,
		exist_ok=args.exist_ok,
		val=True,
		plots=True,
	)
	try:
		model.train(**train_kwargs)
	finally:
		if resolved_data_yaml.exists():
			resolved_data_yaml.unlink()

	print("训练完成。最佳权重通常在 runs/.../weights/best.pt")


if __name__ == "__main__":
	main()
