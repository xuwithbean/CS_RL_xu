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

## 实时决策与辅助（Advisor）

- 主入口：`decision_advisor.py`，用于读取 `visual_recognition` 的共享状态（YOLO centers 等），并在无敌人时异步询问 LLM 给出策略，或在见到人时使用本地瞄准模型快速瞄准并开火。
- 常用参数示例：

```bash
# 使用本地瞄准模型打印调试信息（需激活 conda 环境 condacommon）
conda activate condacommon
python decision_advisor.py --debug-aim --aim-model-path point_aim_net_resume_best.pt

# 启用 API Key 调用大模型（Qwen/OpenAI 等）并在空闲时自动询问
python decision_advisor.py --api-key YOUR_KEY --auto-idle-query-sec 3.0
```

## 瞄准模型（Point Aim）

- 模型训练/加载：`point_aim_trainer.py` 提供 `load_model`，仓库包含若干模型权重（如 `point_aim_net_resume_best.pt`、`point_aim_net.pt` 等）。
- 在 `decision_advisor.py` 中会以优先级使用本地瞄准模型（如果可用），以降低对 LLM 的实时依赖。

## 控制接口

- `actions.py` 提供高层动作映射（按键/鼠标）；`control.py` 提供底层键盘/鼠标发送实现（socket client）。
- 常用函数：`m_actions().mouse_click_interval()`、`m_actions().mouse_move()`、`m_actions().stop()` 等。

## 常用脚本

- `run.sh`：启动游戏推流（WSL+Windows 配合）。
- `start_realtime.sh`：一键启动视觉实时管道（`visual_recognition/realtime_pipeline.py` + ffplay）。
- `yolo_ffplay.sh`, `yolorun.sh`：YOLO 预览与推流脚本。
- `decision_advisor.sh`：封装运行 `decision_advisor.py` 的脚本（可查看用于调试的默认参数）。

## 运行与调试小贴士

- 推荐在 `condacommon` 环境下运行：

```bash
conda activate condacommon
pip install -r requirements.txt
```

- 若需要调试瞄准行为，使用 `--debug-aim` 来打印模型输入/输出与开火触发信息。
- 若要强制只使用本地瞄准并跳过 LLM，请不要提供 `--api-key`，advisor 会在检测到人物时直接进入瞄准模式。

## 模型与输出目录

- YOLO 训练/输出位于 `visual_recognition/runs/`，模型权重存放在 `visual_recognition/runs/*/weights/`。
- 截图目录：`screenshots/`，裁剪后图片：`screenshots_crop/`。

---

如果需要我把 README 再精简为一页快速参考，或把运行示例分成 Windows/WSL 两部分，我可以继续调整。