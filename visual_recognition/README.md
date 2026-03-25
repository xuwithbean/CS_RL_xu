# visual_recognition

本目录用于训练 YOLO 来识别两类目标：CT 和 T。

如果你希望展示为 person，并区分子类 CT/T，推荐做法是：
- 训练标签仍用两类（CT/T）
- 推理时把显示名映射为 `person(CT)` 和 `person(T)`

## 1. 数据集格式

请使用 YOLO 检测格式：

- `images/train/*.jpg|png`
- `images/val/*.jpg|png`
- `images/test/*.jpg|png`（可选）
- `labels/train/*.txt`
- `labels/val/*.txt`
- `labels/test/*.txt`（可选）

每个 `labels/*.txt` 每行格式：

`class_id x_center y_center width height`

其中坐标为相对值（0~1）。

类别编号固定：
- `0 -> CT`
- `1 -> T`

## 2. 数据配置

默认配置文件是 [data_ct_t.yaml](data_ct_t.yaml)。

你需要把 `path` 指向你的数据集根目录。

## 3. 安装依赖

```bash
pip install ultralytics pyyaml
```

## 4. 开始训练

在仓库根目录执行：

```bash
python3 visual_recognition/train.py \
  --data visual_recognition/data_ct_t.yaml \
  --model yolo11n.pt \
  --epochs 100 \
  --imgsz 640 \
  --batch 16 \
  --device 0
```

训练结果会输出到：
- `visual_recognition/runs/ct_t_yolo/...`

最佳权重通常为：
- `.../weights/best.pt`

## 5. 推理时显示为 person(CT)/person(T)

在仓库根目录执行：

```bash
python3 visual_recognition/predict.py \
  --weights visual_recognition/runs/ct_t_yolo/weights/best.pt \
  --source path/to/image_or_video \
  --conf 0.25 \
  --device 0
```

输出可视化结果中，标签会显示：
- `person(CT)`
- `person(T)`

脚本还会额外输出：
- 头部框（基于身体框比例估计）
- 身体中心坐标 `B(x,y)`
- 头部中心坐标 `H(x,y)`

并在输出目录生成坐标文件：
- `detections_centers.csv`

可选参数示例：

```bash
python3 visual_recognition/predict.py \
  --weights visual_recognition/runs/ct_t_yolo/weights/best.pt \
  --source path/to/video.mp4 \
  --head-ratio 0.30 \
  --head-width-ratio 0.45 \
  --show
```

说明：YOLO 检测头本质是单层类别预测，不直接支持“父类+子类”层级结构。
因此工程上通常采用“训练用细分类 + 展示时映射为父类(子类)”的方式实现。

## 6. Win + WSL 实时带框视频

你的场景是 Windows 跑 CS，WSL 跑识别。推荐使用 UDP 流输入：

```bash
python3 visual_recognition/predict.py \
  --weights visual_recognition/runs/ct_t_yolo/weights/best.pt \
  --source udp://192.168.221.36:1234 \
  --conf 0.25 \
  --imgsz 640 \
  --device 0 \
  --fps 60 \
  --save-video
```

执行后会实时生成：
- 带框视频：`visual_recognition/runs/<name>/predicted_realtime.mkv`
- 坐标文件：`visual_recognition/runs/<name>/detections_centers.csv`

如果你有可用 GUI/X11，也可以加 `--show` 实时预览窗口。

### 实时输出带框流（推荐）

如果你希望在 Linux 中实时得到带框视频流（而不仅是落盘文件），可加 `--out-stream`：

```bash
python3 visual_recognition/predict.py \
  --weights visual_recognition/runs/ct_t_yolo/weights/best.pt \
  --source udp://192.168.221.36:1234 \
  --conf 0.25 \
  --imgsz 640 \
  --device 0 \
  --fps 60 \
  --save-video \
  --out-stream udp://127.0.0.1:2234 \
  --stream-fps 60
```

然后在同一台 Linux/WSL 上可用 ffplay 查看带框输出流：

```bash
ffplay -fflags nobuffer -flags low_delay -framedrop udp://127.0.0.1:2234
```

## 7. 一键启动脚本

仓库根目录提供了 [start_realtime.sh](../start_realtime.sh)，可一键启动：
- Windows 推流（可选）
- WSL 实时检测
- 带框流预览（可选）

直接运行：

```bash
bash start_realtime.sh
```

常用环境变量（按需覆盖）：

```bash
PYTHON_BIN=/home/xu/anaconda3/envs/condacommon/bin/python \
WEIGHTS=visual_recognition/runs/ct_t_yolo/weights/best.pt \
IN_STREAM=udp://192.168.221.36:1234 \
OUT_STREAM=udp://127.0.0.1:2234 \
MONITOR=2 FPS=60 BITRATE=2500k \
PREVIEW=1 SKIP_WIN_STREAM=0 SHOW_WINDOW=0 \
bash start_realtime.sh
```

说明：
- 若你已在别处启动 Windows 推流，可设 `SKIP_WIN_STREAM=1` 仅启动识别链路。
- 按 `Ctrl+C` 可统一退出并触发清理流程。
