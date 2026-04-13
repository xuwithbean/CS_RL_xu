# CS_RL_xu

毕业设计项目：使用强化学习与视觉识别辅助游玩 CS。

## 项目概览

- 推流与截图统一在 [opengame.py](opengame.py)。
- 识别主入口为 [visual_recognition/predict.py](visual_recognition/predict.py)。
- 已支持四类中心信息输出：CT、T、CT_HEAD、T_HEAD。
- 已支持识别区域 ROI 过滤，用于规避 HUD 干扰。
- OCR 默认引擎为 pytesseract（Python 3.13 兼容更好）。

## 环境说明

- 运行环境：Windows + WSL2。
- 需要 Windows 侧可用 ffmpeg/ffplay，并可从 WSL 调用 powershell.exe。
- Python 依赖见 [requirements.txt](requirements.txt)。

## 快速开始

1. 安装依赖

```bash
pip install -r requirements.txt
```

2. 启动游戏并推流

```bash
bash run.sh
```

3. 在同一终端执行运行时命令

```text
screenshot
```

```text
screenshot_100
```

```text
p
```

说明：
- screenshot：抓取 1 张到 screenshots/。
- screenshot_100：每 2 秒抓 1 张，共 100 张。
- p：停止 screenshot_100 任务。

## 识别脚本

### 联合识别（YOLO + OCR）

```bash
bash recognize.sh
```

常用环境变量：
- SOURCE：输入图片/视频/流地址。
- WEIGHTS：YOLO 权重路径。
- DETECT_ROI：识别区域（相对坐标 x,y,w,h）。
- OCR、OCR_ENGINE、OCR_ROI：OCR 开关与参数。
- OUT_STREAM：带框输出流地址。
- YOLO_INFO_JSONL：YOLO 四类中心信息输出文件。
- OCR_INFO_JSONL：OCR 信息输出文件。

### YOLO-only / OCR-only

```bash
bash yolo_pic.sh
bash yolo_video.sh
bash ocr_pic.sh
bash ocr_video.sh
```

## 图片批量裁剪

如果你要先把截图/采集图片统一裁掉 HUD 或边缘区域，可直接用：

```bash
bash crop_images.sh
```

常用环境变量：
- INPUT：输入图片或目录。
- OUTPUT：输出目录。
- ROI：裁剪区域（相对坐标 x,y,w,h，默认 0.00,0.08,1.00,0.84）。
- RECURSIVE=1：递归处理子目录。
- PRESERVE_TREE=1：输出时保留目录结构。
- SUFFIX：输出文件名后缀。

示例：批量裁掉上下 HUD

```bash
INPUT=./screenshots \
OUTPUT=./screenshots_crop \
ROI=0.00,0.08,1.00,0.84 \
bash crop_images.sh
```

## run.sh 常用环境变量

- GAME_EXE：Windows 游戏可执行路径。
- LINUX_IP：WSL 接收地址。
- PORT：推流端口。
- WINDOW_TITLE：窗口标题关键字（默认 auto）。
- STREAM_OUTPUT：原始流输出地址。
- VIEWER_SOURCE：预览输入源（可指向处理后流）。
- VIEW_W / VIEW_H：预览窗口大小。
- NO_VIEWER=1：只推流不启动预览。

## 视觉实时管道

[visual_recognition/realtime_pipeline.py](visual_recognition/realtime_pipeline.py) 用于一键编排：
- Windows 推流（可选）
- WSL 识别
- 可选 ffplay 预览

对应封装脚本为 [start_realtime.sh](start_realtime.sh)。

## 训练与测试

- 强化学习训练：train.py
- 强化学习测试：test.py
- 视觉训练：visual_recognition/train.py
- 视觉推理：visual_recognition/predict.py