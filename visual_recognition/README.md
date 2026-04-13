# visual_recognition

本目录用于训练与推理 CS 视觉识别模型，支持 YOLO 与 OCR。

## 1. 能力范围

- YOLO 检测并输出中心点。
- 支持两类数据：CT、T。
- 支持四类数据：CT、T、CT_HEAD、T_HEAD。
- 支持识别区域 ROI 过滤，规避 HUD 干扰。
- OCR 默认使用 pytesseract（Python 3.13 更稳）。

## 2. 数据集格式

YOLO 检测格式：

- images/train
- images/val
- images/test（可选）
- labels/train
- labels/val
- labels/test（可选）

每个 labels 文件每行格式：

class_id x_center y_center width height

坐标为相对值（0 到 1）。

如果你希望先对截图/采集图做统一裁剪（例如裁掉 HUD），可以使用根目录脚本 [crop_images.sh](../crop_images.sh)：

```bash
INPUT=./screenshots \
OUTPUT=./screenshots_crop \
ROI=0.00,0.08,1.00,0.84 \
bash crop_images.sh
```

默认会保留输入目录结构，并为输出文件添加 `_crop` 后缀。

## 3. 数据配置文件

- 两类配置：[data_ct_t.yaml](data_ct_t.yaml)
  - 0: CT
  - 1: T
- 四类配置：[data_ct_t_head.yaml](data_ct_t_head.yaml)
  - 0: CT
  - 1: T
  - 2: CT_HEAD
  - 3: T_HEAD
- 冒烟配置：[data_ct_t_smoke.yaml](data_ct_t_smoke.yaml)

## 4. 训练

两类训练：

```bash
python3 visual_recognition/train.py \
  --data visual_recognition/data_ct_t.yaml \
  --model yolo11n.pt \
  --epochs 100 \
  --imgsz 640 \
  --batch 16 \
  --device 0
```

四类训练：

```bash
python3 visual_recognition/train.py \
  --data visual_recognition/data_ct_t_head.yaml \
  --model yolo11n.pt \
  --epochs 100 \
  --imgsz 640 \
  --batch 16 \
  --device 0
```

## 5. 联合推理（YOLO + OCR）

主入口是 [predict.py](predict.py)。

示例：

```bash
python3 visual_recognition/predict.py \
  --weights visual_recognition/runs/ct_t_yolo/weights/best.pt \
  --source udp://192.168.221.36:1234 \
  --ocr \
  --detect-roi 0.00,0.08,1.00,0.84 \
  --out-stream udp://127.0.0.1:2234 \
  --stream-fps 60
```

输出内容：

- detections_centers.csv
- yolo_info.jsonl（四类中心信息）
- ocr_hud.csv（启用 OCR 时）
- ocr_info.jsonl（启用 OCR 时）
- predicted_realtime.mkv（视频输入或启用 save-video 时）

## 6. YOLO-only 与 OCR-only

- YOLO-only 模块：[yolor.py](yolor.py)
- OCR-only 模块：[ocrr.py](ocrr.py)

根目录快捷脚本：

- yolo_pic.sh
- yolo_video.sh
- ocr_pic.sh
- ocr_video.sh
- recognize.sh（联合识别）

## 7. 实时编排脚本

[realtime_pipeline.py](realtime_pipeline.py) 与根目录 [start_realtime.sh](../start_realtime.sh) 用于一键编排：

- Windows 推流（可选）
- WSL 推理
- 带框流预览（可选）

常用参数：

- detect-roi：识别区域 ROI
- print-yolo：打印每帧四类中心
- yolo-info-jsonl：指定 YOLO JSONL 输出路径

## 8. 依赖

建议在项目根目录安装：

```bash
pip install -r requirements.txt
```
