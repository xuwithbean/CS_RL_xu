# CS_RL_xu

毕业设计项目：使用强化学习与视觉识别辅助游玩 CS。

## 项目现状

- 核心推流与截图统一到 [opengame.py](opengame.py)。
- 旧的 get_screenshot.py 已移除。
- 支持窗口捕获推流、Windows 侧预览、运行中终端命令截图。
- 支持将原始流接入 YOLO/OCR 等后处理，并播放处理后流。

## 环境说明

- 运行环境：Windows + WSL2。
- 需要 Windows 侧可用 `ffmpeg` / `ffplay`，并可从 WSL 调用 `powershell.exe`。
- Python 依赖见 [requirements.txt](requirements.txt)。

## 快速开始

1. 安装依赖

```bash
pip install -r requirements.txt
```

2. 启动游戏并推流（默认命令）

```bash
bash run.sh
```

3. 运行中截图

- 在运行 `run.sh` 的同一终端输入：

```text
screenshot
```

- 图片保存到 `./screenshots/`。

## run.sh 常用环境变量

- `GAME_EXE`：Windows 游戏可执行路径。
- `LINUX_IP`：WSL 接收地址。
- `PORT`：推流端口。
- `WINDOW_TITLE`：窗口标题关键字（默认 `auto`）。
- `STREAM_OUTPUT`：自定义原始流输出地址（例如 `udp://192.168.221.36:2234`）。
- `VIEWER_SOURCE`：播放输入源地址（可指向 YOLO/OCR 处理后的流）。
- `VIEW_W` / `VIEW_H`：预览窗口大小（默认 `800x450`）。
- `NO_VIEWER=1`：只推流不打开预览窗口。

示例：输出原始流并播放处理后流

```bash
STREAM_OUTPUT='udp://192.168.221.36:2234' \
VIEWER_SOURCE='udp://127.0.0.1:3333?fifo_size=1000000&overrun_nonfatal=1' \
bash run.sh
```

## 视觉识别流水线

- 入口脚本：[visual_recognition/realtime_pipeline.py](visual_recognition/realtime_pipeline.py)
- 流程：
	- `opengame` 推送原始流
	- `visual_recognition/predict.py` 做检测与画框
	- 可选 `ffplay` 预览处理后流

## 训练与测试

- 强化学习训练：`train.py`
- 强化学习测试：`test.py`
- 视觉训练：`visual_recognition/train.py`
- 视觉推理：`visual_recognition/predict.py`