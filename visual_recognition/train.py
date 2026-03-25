"""训练 YOLO 检测模型用于 CT/T 识别。

数据集需采用 YOLO 检测格式：
- images/train, images/val, images/test（可选）
- labels/train, labels/val, labels/test（可选）

标签类别约定：
- 0: CT
- 1: T
"""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path

import yaml


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
	parser.add_argument("--workers", type=int, default=8, help="dataloader workers")
	parser.add_argument("--patience", type=int, default=50, help="早停 patience")
	parser.add_argument("--seed", type=int, default=42, help="随机种子")
	return parser.parse_args()


def get_check_dataset_yaml(data_yaml: Path) -> None:
	if not data_yaml.exists():
		raise FileNotFoundError(
			f"未找到数据配置文件: {data_yaml}\n"
			"请先参考 visual_recognition/data_ct_t.yaml 准备数据集。"
		)

	with data_yaml.open("r", encoding="utf-8") as f:
		cfg = yaml.safe_load(f)

	names = cfg.get("names", None)
	if isinstance(names, dict):
		ordered = [names[k] for k in sorted(names.keys())]
	elif isinstance(names, list):
		ordered = names
	else:
		raise ValueError("数据集 yaml 缺少 names 字段，或格式错误。")

	if ordered != ["CT", "T"]:
		raise ValueError(
			f"类别配置应为 ['CT', 'T']，当前为 {ordered}。\n"
			"请确认标签编号 0->CT, 1->T。"
		)


def main() -> None:
	args = get_args()
	data_yaml = Path(args.data)
	get_check_dataset_yaml(data_yaml)

	# 延迟导入，方便在未安装依赖时给出更清晰报错。
	try:
		ultralytics = importlib.import_module("ultralytics")
		YOLO = ultralytics.YOLO
	except ImportError as exc:
		raise ImportError(
			"未安装 ultralytics。请先执行: pip install ultralytics"
		) from exc

	model = YOLO(args.model)
	model.train(
		data=str(data_yaml),
		epochs=args.epochs,
		imgsz=args.imgsz,
		batch=args.batch,
		device=args.device,
		project=args.project,
		name=args.name,
		workers=args.workers,
		patience=args.patience,
		seed=args.seed,
		val=True,
		plots=True,
	)

	print("训练完成。最佳权重通常在 runs/.../weights/best.pt")


if __name__ == "__main__":
	main()
