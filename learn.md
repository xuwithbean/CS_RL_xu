# CS_RL_xu 项目学习文档

## 项目概览

本项目实现了一个**基于大模型（LLM）决策和强化学习（RL）控制的CS2（Counter-Strike 2）游戏智能体系统**。系统运行在 **Windows + WSL2** 混合环境下，**不依赖游戏内部接口**，仅通过纯视觉输入（屏幕画面）实现端到端的游戏控制。

### 核心架构：LLM + RL 分层混合

```
┌─────────────────────────────────────────────────────┐
│                  大模型决策层 (LLM)                   │
│  Qwen-VL: 态势感知、地图识别、战略建议、击杀确认        │
├─────────────────────────────────────────────────────┤
│              强化学习控制层 (TD3 Agent)               │
│  高层管理器: search / fight / take_cover             │
│  低层TD3: 19维状态→3维连续动作(水平/垂直/射击)        │
├─────────────────────────────────────────────────────┤
│               视觉感知流水线 (Vision)                 │
│  YOLO(目标检测) + OCR(HUD读取) + Qwen-VLM(语义理解)   │
├─────────────────────────────────────────────────────┤
│        数据采集 & 跨平台控制 (Windows + WSL2)         │
│  ffmpeg gdigrab → UDP流 → Socket → Win32 API        │
└─────────────────────────────────────────────────────┘
```

---

## 目录结构

```
CS_RL_xu/
├── *.sh                     # 启动脚本（bash封装）
├── *.py                     # 核心Python模块
├── visual_recognition/      # 视觉识别模块
│   ├── predict.py           # YOLO+OCR联合预测
│   ├── stream_ffplay_pipeline.py  # 流式处理流水线
│   ├── realtime_pipeline.py # 实时管道（一键编排）
│   ├── ocrr.py              # OCR识别模块
│   ├── yolor.py             # YOLO识别模块
│   ├── train.py             # YOLO模型训练
│   ├── data_ct_t*.yaml      # YOLO数据集配置
│   ├── runs/                # YOLO训练输出
│   └── datasets/            # 数据集
├── screenshots/             # 游戏截图
├── screenshots_crop/        # 裁剪后的截图
├── list/                    # 列表文件
└── *.pt                     # 模型权重文件
```

---

## Python 核心模块详解

### 1. actions.py — 动作组合封装

**作用**：提供高层游戏操作接口，封装键盘和鼠标控制。

**原理**：通过 `m_actions` 类组合 `KeySender`（键盘）和 `MouseController`（鼠标）的底层控制，提供语义化的游戏动作函数。

**关键类**：
```python
class m_actions:
    def __init__(self, key_sender=None, mouse_controller=None)
```

**主要方法**：

| 方法 | 参数 | 作用 | 原理 |
|------|------|------|------|
| `move_forward()` | hold_sec=None | 按W前进 | hold_sec=None时一直按住 |
| `move_back()` | hold_sec=None | 按S后退 | 同上 |
| `move_left/right()` | hold_sec=None | 按A/D平移 | 同上 |
| `jump()` | hold_sec=None | 空格跳跃 | 同上 |
| `reload()` | hold_sec=None | R换弹 | 同上 |
| `crouch()` | hold_sec=None | Ctrl蹲下 | 持续按住 |
| `crouch_end()` | — | 取消蹲下 | 释放Ctrl |
| `mouse_move(dx, dy)` | dx, dy | 相对移动鼠标 | 调用Win32 mouse_event |
| `mouse_click()` | hold_sec=None | 鼠标左键点击 | 默认50ms按下 |
| `mouse_click_interval(click_times, interval_sec)` | 点击次数、间隔 | 连点 | 可被X2侧键中断 |
| `mouse_move_click(dx, dy)` | dx, dy | 移动+点击 | 组合动作 |
| `stop()` | — | 紧急停止 | 释放所有按键+鼠标 |
| `is_interrupt_x2_pressed()` | — | 检测中断 | 读取鼠标X2侧键状态 |
| `stop_if_interrupt_x2()` | — | 条件停止 | X2按下则stop → True |

**hold_sec 参数**：为 `None` 时表示"持续按住直到释放"；设为数值时表示"按住N秒后自动释放"。

---

### 2. control.py — WSL→Windows 低延迟控制模块

**作用**：实现 WSL（Linux）到 Windows 的游戏控制通道，是系统的"执行手臂"。

**原理**：通过 TCP Socket 长连接，向 Windows 端监听程序发送文本命令，Windows 端调用 Win32 API（`keybd_event`, `mouse_event`）执行实际的键盘鼠标操作。

**关键类**：

#### WinControlClient — 客户端连接
```python
WinControlClient(host=None, port=None, timeout=0.5)
```
- `host`：Windows主机IP，自动探测（默认路由网关→resolv.conf→127.0.0.1）
- `port`：监听端口，默认60000
- `timeout`：连接超时

**核心方法**：
- `send(line)`：发送单行命令
- `send_lines(lines)`：批量发送，减少TCP往返次数

#### KeySender — 键盘控制
```python
KeySender(default_hold_ms=50, client=None)
```
- `press(key)`：按下并释放（快捷操作）
- `press_and_release(key, hold_ms, inter_ms)`：按住指定毫秒后释放
- `release(key)`：释放按键

#### MouseController — 鼠标控制
```python
MouseController(default_hold_ms=50, client=None)
```
- `move(dx, dy)`：相对移动鼠标（deltax, deltay像素）
- `click(button, hold_ms)`：点击（left/right/middle）
- `move_and_click(dx, dy, button, hold_ms, inter_ms)`：移动+点击组合
- `press/release(button)`：按住/释放
- `scroll(delta)`：滚轮
- `is_button_pressed(button)`：查询按键状态（x2侧键用于紧急停止）

**控制协议**（文本行格式，逐行发送）：

| 命令 | 参数 | 作用 |
|------|------|------|
| `KEY_DOWN vk` | 虚拟键码 | 按键按下 |
| `KEY_UP vk` | 虚拟键码 | 按键释放 |
| `MOUSE_MOVE dx dy` | 像素偏移 | 鼠标相对移动 |
| `MOUSE_PRESS button` | left/right/middle | 鼠标按下 |
| `MOUSE_RELEASE button` | left/right/middle | 鼠标释放 |
| `MOUSE_CLICK button hold_ms` | 按钮+时长 | 点击操作 |
| `MOUSE_SCROLL delta` | 滚轮量 | 滚动 |
| `SLEEP ms` | 毫秒 | 延时等待 |
| `PING` | — | 心跳检测（回复PONG） |
| `IS_BUTTON vk` | 虚拟键码 | 查询按键状态（返回1/0） |
| `SHUTDOWN` | — | 关闭服务器 |

**VK_MAP（虚拟键码映射）**：
```
w→0x57, a→0x41, s→0x53, d→0x44, space→0x20
ctrl→0x11, shift→0x10, alt→0x12, tab→0x09, esc→0x1B
```

**Windows端监听服务**：通过PowerShell内联C#代码编译Win32 API绑定，启动TCP监听器。支持端口自动回退（60000→60100→55000→...）。`StartWinListener` 类负责在WSL中启动这个PowerShell进程。

---

### 3. td3_agent.py — TD3 强化学习智能体

**作用**：实现 **Twin Delayed DDPG (TD3)** 算法，用于连续动作空间（鼠标瞄准+射击）的强化学习。

**原理**：TD3是DDPG的改进版本，通过三个关键技术解决Q值过估计问题：
1. **双Q网络**：两个独立Critic，取较小Q值作为目标
2. **延迟策略更新**：每2步Critic更新对应1步Actor更新
3. **目标策略平滑**：目标动作加高斯噪声正则化

**网络结构**：

```
Actor网络（策略网络）：
  状态(19维) → Linear(256) → ReLU → Linear(256) → ReLU → Linear(3) → Tanh → 动作[-1,1]

Critic网络（Q值网络，×2）：
  [状态(19)+动作(3)] → Linear(256) → ReLU → Linear(256) → ReLU → Linear(1) → Q值
```

**关键类**：

#### Actor — 策略网络
- 输入：state_dim=19（状态维度）
- 输出：action_dim=3（动作维度）：[水平移动, 垂直移动, 射击得分]
- 隐藏层：256×256，ReLU激活
- 输出层：Tanh激活，范围[-1, 1]

#### Critic — 价值网络（两个独立副本）
- 输入：state_dim + action_dim = 22
- 输出：1维Q值
- 提供 `q1_value()` 方法（Actor更新时用）

#### ReplayBuffer — 经验回放缓冲池
```python
ReplayBuffer(state_dim, action_dim, capacity=50000)
```
- 容量：50,000条
- 存储格式：numpy数组（高效随机采样）
- 循环队列：满时覆盖最旧数据
- `sample(batch_size=128)`：随机采样

#### TD3Agent — 主智能体
```python
TD3Agent(
    state_dim=19,           # 状态维度
    action_dim=3,           # 动作维度  
    action_limit=1.0,       # 动作范围
    actor_lr=1e-4,          # Actor学习率
    critic_lr=1e-3,         # Critic学习率
    gamma=0.99,             # 折扣因子
    tau=0.005,              # 软更新系数(Polyak平均)
    policy_noise=0.2,       # 目标策略噪声(标准差)
    noise_clip=0.5,         # 噪声裁剪范围
    policy_delay=2,         # 策略延迟更新(步数)
    device="cuda",          # 计算设备
    hidden_dim=256,         # 隐藏层维度
)
```

**核心方法**：

| 方法 | 作用 | 算法细节 |
|------|------|----------|
| `select_action(state, noise_scale, deterministic)` | 选择动作 | 有噪声(训练)/无噪声(测试)/确定性三种模式 |
| `train_step(replay_buffer, batch_size)` | 训练一步 | 返回 `TD3TrainStats` |
| `save(path, replay_buffer, extra_meta)` | 保存模型 | torch.save完整状态 |
| `load(path, device)` | 加载模型 | 类方法，返回(agent, replay_buffer, meta) |

**训练流程**（`train_step`内部）：
1. 从经验池采样batch
2. 计算目标动作（目标Actor + 裁剪噪声）
3. 计算目标Q值（两个目标Critic的最小值）
4. 计算Critic损失（MSE）→ 更新Critic
5. 每policy_delay步：更新Actor（最大化Q值）→ 软更新目标网络

---

### 4. train.py — 强化学习训练主程序

**作用**：训练TD3智能体的主循环，包含仿真环境和真实环境适配。

**原理**：实现分层强化学习（HRL），Manager选择高层目标，TD3 Agent执行底层连续控制。

**两种训练环境**：

#### SimpleCombatEnv（简化仿真环境）
用于算法联调，纯Python模拟，不连接真实游戏。

状态维度：9维观测
- hit/kill/death：命中/击杀/死亡事件
- 弹药、血量、目标可见性、瞄准误差、危险等级

动作空间：离散动作
- idle, aim_left, aim_right, aim_up, aim_down, shoot, reload, move_back, strafe_left, strafe_right

**环境模拟逻辑**：
- 视野随机变化，search模式下更易看到敌人
- 瞄准动作减少误差，射击命中概率与瞄准精度相关
- 击杀有一定概率，危险等级随时间变化
- 血量归零或击杀成功时回合结束

#### SharedPointEnv（真实游戏环境）
通过共享状态文件读取YOLO检测结果，在真实CS2中训练。

**配置参数**（TrainConfig）：
```python
@dataclass
class TrainConfig:
    episodes: int = 200            # 训练回合数
    max_steps: int = 200           # 每回合最大步数
    manager_interval: int = 10     # 管理器决策间隔(步数)
    env_mode: str = "auto"         # 环境模式(simple/shared/auto)
    step_dt_sec: float = 0.03     # 每步时间间隔(秒)
    shared_state_path: str = "/tmp/cs_rl_runtime_state.json"
    apply_actions: bool = True     # 是否实际执行动作
    batch_size: int = 128          # 训练批大小
    replay_size: int = 50000       # 经验池容量
    start_steps: int = 400         # 预热步数(随机探索)
    exploration_noise: float = 0.15  # 探索噪声
    shoot_threshold: float = 0.12  # 射击触发阈值
    shoot_center_error: float = 0.04  # 瞄准中心误差阈值
    move_gain: float = 400.0       # 鼠标移动增益
    checkpoint_every: int = 10     # 保存间隔(回合)
    gamma: float = 0.99
    tau: float = 0.005
    policy_noise: float = 0.20
    noise_clip: float = 0.50
    policy_delay: int = 2
    auto_measure_stream_delay: bool = True  # 自动测量流延迟
    qwen_api_key: str = ""          # Qwen API密钥（可选）
```

**管理器决策逻辑**（`get_manager_goal`）：
- 敌人可见 → `"fight"`（战斗）
- 敌人不可见 → `"search"`（搜索）
- 未来支持 → `"take_cover"`（隐蔽，预留LLM决策）

**训练主循环**：
```
for episode in range(episodes):
    重置环境
    manager选择初始目标
    for step in range(max_steps):
        Actor + 探索噪声 → 动作
        执行动作(鼠标/键盘)
        计算奖励(get_reward)
        存储经验到缓冲池
        if 缓冲池足够: train_step()
        if step%manager_interval==0: 管理器重新决策
```

---

### 5. get_reward.py — 奖励函数

**作用**：计算强化学习的即时奖励，引导智能体学习期望行为。

**原理**：密集奖励设计，包含10+个分量，每个分量有不同权重。

**参数**：
```python
get_reward(prev_obs, curr_obs, action_name, manager_goal, kill_count_reader, kill_count_state)
```

**奖励分量表**：

| 分量 | 权重 | 触发条件 | 设计目的 |
|------|------|----------|----------|
| `hit` | +3.20 | 命中敌人 | 鼓励准确射击 |
| `kill` | +14.00 | 击杀敌人 | 追求最终目标 |
| `kill_speed` | +4.00×kill×(1-time/4) | 击杀时计算 | 鼓励快速击杀（4秒内满分） |
| `death` | -6.00 | 被击杀 | 避免死亡 |
| `aim` | +3.00×(prev_aim-curr_aim) | 瞄准改善时 | 引导准星逼近目标 |
| `center_lock` | +3.50×(1-curr_aim) | 准星靠近目标 | 持续维持瞄准 |
| `center_snap` | +5.00×max(0, 0.15-curr_aim) | 误差<0.15 | 精确瞄准奖励 |
| `waste_fire` | -0.50×(shot_fired-hit) | 无效射击 | 减少浪费弹药 |
| `center_shot_bonus` | +4.00 | 准星对准时射击 | 精确射击额外奖励 |
| `early_shot_penalty` | -0.40×min(1, aim/0.35) | 未瞄准就开枪 | 惩罚盲射 |
| `aim_miss_penalty` | -1.20×min(1, aim/0.5) | 目标可见但未瞄准 | 督促瞄准 |
| `hide_penalty` | -0.80×min(1, time/1.5) | 目标消失 | 防止躲开敌人 |
| `goal_align` | 动态 | 根据manager_goal | 确保下层策略配合上层目标 |
| `lost_target_penalty` | -0.60×min(1, time/1.5) | 目标突然消失 | 防止"甩出画面"投机 |

**Manager目标对齐**：
- `fight`模式：鼓励命中+瞄准+射击
- `search`模式：发现敌人时给少量奖励
- `take_cover`模式：暂不激励

---

### 6. decision_advisor.py — 实时决策与辅助（Advisor）

**作用**：读取视觉感知结果，在无敌人时异步调用LLM给出策略，有敌人时使用本地瞄准模型快速瞄准并开火。

**原理**：结合YOLO检测结果、共享状态和LLM，实现实时游戏辅助决策。

**动作映射表**：
```
A → 左平移(a键)        B → 右转(鼠标右移1000px)
C → 左转(鼠标左移1000px)  D → 右平移(d键)
E → 跳跃(space)        F → 蹲下(ctrl)
G → 前行(w键)
```

**核心功能**：
1. **LLM决策**：`get_query_next_action_with_choice()` — 向Qwen发送当前画面+检测结果，获取战略建议
2. **本地瞄准**：使用point_aim_trainer加载的模型进行快速瞄准
3. **击杀检测**：`get_query_kill_count_from_frame()` — 通过Qwen判断击杀数
4. **动作执行**：`execute_action_choice()` — 解析LLM返回的动作码并执行

**关键参数**（命令行）：
```
--debug-aim             打印瞄准调试信息
--aim-model-path PATH   本地瞄准模型路径
--api-key KEY           Qwen/OpenAI API密钥
--auto-idle-query-sec N  空闲时自动询问间隔(秒)
--no-idle-query         关闭空闲询问
```

---

### 7. point_aim_trainer.py — 本地瞄准模型训练器（核心控制模型）

**作用**：训练一个轻量级Actor-Critic神经网络，直接从YOLO检测结果中的目标中心坐标预测鼠标移动量。这是本项目**实际使用**的瞄准控制核心模型。

**原理**：读取共享状态JSON中的目标中心点坐标（来自YOLO检测），构建3维状态向量，通过一个自定义的Actor-Critic架构学习从"目标位置偏差"到"鼠标移动量"的映射关系。训练采用**监督式策略更新**和**强化学习Actor-Critic**两种互补的机制。

#### 神经网络结构

```
┌─────────────────────────────────────────────────────┐
│                   PointAimNet (Actor)                │
├─────────────────────────────────────────────────────┤
│  输入: 3维状态向量                                    │
│    state[0] = ndx: 水平偏差 (目标x-中心x)/中心x [-1,1] │
│    state[1] = ndy: 垂直偏差 (目标y-中心y)/中心y [-1,1] │
│    state[2] = aim_error: 欧几里得距离 sqrt(ndx²+ndy²) │
├─────────────────────────────────────────────────────┤
│  Linear(3 → 2): 全连接层，无偏置                       │
│    W·x = [w11·ndx + w12·ndy + w13·err,               │
│           w21·ndx + w22·ndy + w23·err]               │
├─────────────────────────────────────────────────────┤
│  Tanh 激活 → 输出: 2维连续动作 [-1,1]                 │
│    action[0] = move_x: 水平鼠标移动量(归一化)          │
│    action[1] = move_y: 垂直鼠标移动量(归一化)          │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│                    CriticNet                         │
├─────────────────────────────────────────────────────┤
│  输入: state(3) + action(2) = 5维                    │
├─────────────────────────────────────────────────────┤
│  Linear(5 → 64) → ReLU                               │
│  Linear(64 → 64) → ReLU                              │
│  Linear(64 → 1) → Q值                                │
└─────────────────────────────────────────────────────┘
```

#### 关键设计特点

**Actor为什么用单层线性网络？**
- 瞄准控制本质上是"点到中心"的几何问题，存在明确的线性映射关系：目标在屏幕左侧，鼠标应该左移，偏移量大致与距离成正比
- 单层Linear网络足以拟合这种线性或近似线性的映射
- 参数少，推理快，适合实时控制场景
- 初始化为零权重零偏置，策略从"不移动"开始，在交互中逐步学习

**两种互补训练机制**：

| 机制 | 函数 | 原理 | 优势 |
|------|------|------|------|
| 监督式策略更新 | `policy_update()` | 以目标方向为监督信号，加权MSE损失 | 利用几何先验快速学习正确方向 |
| Actor-Critic强化学习 | `train_step()` | 通过Critic的Q值引导Actor优化 | 适应未知环境，学习最优策略 |

**监督式策略更新（policy_update）**：
```python
def policy_update(model, optimizer, states, targets, weights, device):
    pred = model(states)                         # 模型预测动作
    loss = ((pred - targets)² × weights).mean()  # 加权MSE损失
    optimizer.zero_grad()
    loss.backward()
    clip_grad_norm_(5.0)                         # 梯度裁剪
    optimizer.step()
    return loss.item()
```
- `targets` 由 `make_policy_target()` 生成：直接将当前状态的前两维（ndx, ndy）作为理想动作
- `weights` 由回合奖励动态调整：奖励越高，该样本的权重越大

**Actor-Critic强化学习（train_step）**：
```python
def train_step(actor, target_actor, critic, target_critic, replay, ...):
    # 1. 从经验池采样
    states, actions, rewards, next_states, dones = replay.sample(batch_size)
    
    # 2. 更新Critic：最小化Q值预测与目标值的MSE
    q_values = critic(states, actions)
    next_actions = target_actor(next_states)
    target = rewards + γ × target_critic(next_states, next_actions) × (1 - dones)
    critic_loss = MSE(q_values, target)
    
    # 3. 更新Actor：最大化Critic的Q值输出
    pred_actions = actor(states)
    actor_loss = -critic(states, pred_actions).mean()  # 梯度上升
    
    # 4. 软更新目标网络
    target_p ← τ × p + (1-τ) × target_p    # Polyak平均
```

#### 完整训练循环

```
while True:
    1. 读取共享状态JSON → 获取目标中心坐标
    2. 若无目标 → 左右搜索(每0.2秒移动500-1200像素)
    3. 若有目标 → 构建状态向量 [ndx, ndy, aim_error]
    4. 判断：若 aim_error ≤ shoot_center_error(0.1)
       → 标记回合成功，随机移开目标，等待2秒
    5. 否则：模型预测动作 → 转换为鼠标移动 → 执行
    6. 等待 round_wait_sec 秒 → 读取新状态
    7. compute_reward() 计算奖励
    8. 记录 (state, target_action, weight) 到回合列表
    9. _finish_episode() → 调用 policy_update() 学习
    10. 定期保存模型 + 绘制奖励曲线
```

#### 奖励函数（compute_reward）

**输入**：回合起始状态、回合结束状态
**公式**：
```
directional = 0.5 × (score_x + score_y)    # 方向正确性
progress = 1 - center_dist / start_dist     # 距离缩小比例
reward = reward_scale × (0.6×directional + 0.4×progress)

# 到达中心时的高额奖励
if center_dist ≤ shoot_center_error:
    reward += reward_scale × (4.0 + 6.0 × center_ratio)

# 远离中心的惩罚
if center_dist > start_dist:
    reward -= reward_scale × (center_dist/start_dist - 1)
```

**axis_score 计算**：评估单轴上的移动方向是否正确
- 若起始位置已在中心(≤eps)，偏离则惩罚
- 若起始位置在非中心，朝着中心方向移动得正分，远离得负分

#### 动作映射（action_to_command）

```
raw_move_x = action[0] × move_gain_x    # 默认2500
raw_move_y = action[1] × move_gain_y    # 默认500
move_x = clip(raw_move_x, -max_move_x, max_move_x)  # 默认最大1000
move_y = clip(raw_move_y, -max_move_y, max_move_y)  # 默认最大500

# 到达中心误差范围时停止移动
if aim_error ≤ shoot_center_error(0.1):
    move_x = 0, move_y = 0
```

#### 搜索策略

当画面中无检测目标（centers为空或target为None）时：
- 每隔 `search_interval_sec`（默认0.2秒）执行一次搜索动作
- 随机选择一个搜索方向（+1或-1）
- 向该方向移动 `search_step` 像素（默认500）
- 找到目标后重置搜索状态

#### 关键参数详解

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--shared-state` | /tmp/cs_rl_runtime_state.json | YOLO检测结果共享状态文件 |
| `--save-path` | point_aim_net.pt | 模型保存路径 |
| `--load-path` | (空) | 加载已有模型继续训练 |
| `--move-gain-x` | 2500.0 | X方向增益：网络输出[-1,1] × 2500 = 实际像素 |
| `--move-gain-y` | 500.0 | Y方向增益：网络输出[-1,1] × 500 = 实际像素 |
| `--max-move-x` | 1000 | 单次X方向最大移动像素 |
| `--max-move-y` | 500 | 单次Y方向最大移动像素 |
| `--shoot-center-error` | 0.1 | 归一化误差<此值视为瞄准成功（不射击，仅标记完成） |
| `--max-step` | 400 | 最大训练步数 |
| `--search-step` | 500 | 无目标时搜索移动的像素量 |
| `--search-interval-sec` | 0.20 | 搜索动作的间隔秒数 |
| `--round-wait-sec` | 2.0 | 每轮发出移动后等待时间（等待画面反馈） |
| `--round-reward-scale` | 10.0 | 奖励缩放系数 |
| `--batch-size` | 64 | Actor-Critic训练批大小 |
| `--buffer-size` | 1024 | 经验回放池容量 |
| `--gamma` | 0.98 | 折扣因子 |
| `--noise-start` | 0.08 | 探索噪声初始值（训练步数增加后衰减） |
| `--noise-end` | 0.01 | 探索噪声最小值 |
| `--noise-decay` | 0.999 | 噪声衰减率 |
| `--tau` | 0.01 | 目标网络软更新系数 |
| `--actor-lr` | 1e-3 | Actor学习率 |
| `--critic-lr` | 1e-3 | Critic学习率 |
| `--hidden-dim` | 64 | Critic隐藏层维度（Actor始终为Linear3→2） |
| `--train-only` | false | 仅训练不动鼠标（安全调试模式） |
| `--step-shift-threshold` | 0.008 | 判定动作生效的最小归一化位移变化 |
| `--step-wait-timeout-sec` | 0.22 | 等待动作生效的最长时间 |
| `--action-settle-sec` | 0.06 | 鼠标动作后额外等待画面反馈的最短时间 |

#### 与td3_agent.py的关系

项目中存在两个独立的强化学习实现：
- **point_aim_trainer.py**（实际使用）：轻量级Actor-Critic，3维输入→2维输出，实时在线学习
- **td3_agent.py**（独立模块，未实际集成）：标准TD3算法实现，19维输入→3维输出，适用于train.py中的仿真/真实环境训练

point_aim_trainer.py是项目实际运行的核心瞄准控制模型，通过decision_advisor.py加载并在实时游戏中使用。

---

### 8. get_action.py — 短期决策模块（Q-learning原型）

**作用**：提供基于Q-learning的离散动作空间解决方案，用于早期原型验证。

**原理**：将连续观测离散化为桶编号（bin），使用ε-greedy策略选择动作。

**离散动作空间**（6个）：
```python
ACTIONS = ["idle", "aim_left", "aim_right", "aim_up", "aim_down", "shoot"]
```

**状态离散化**：9维状态键：
```
(manager_goal, target_visible, enemy_visible, target_dx_bin, 
 target_dy_bin, aim_bin, hp_bin, ammo_bin, danger_bin)
```

每个连续特征按阈值分桶：
- `_get_bin(value, thresholds)`：将value按thresholds切分为桶索引

**关键函数**：
- `get_action(q_table, state_key, epsilon)`：ε-greedy选动作
- `get_q_update(q_table, state_key, action_idx, reward, next_state_key, alpha, gamma)`：Q表更新
- `get_action_command(action_name)`：动作名→控制命令映射

---

### 9. find_enemy.py — 敌人特征抽取

**作用**：统一敌人信息数据接口，便于训练代码联调。

**原理**：将原始观测字典标准化为统一的 `EnemyFeature` 数据结构。

```python
@dataclass
class EnemyFeature:
    enemy_visible: bool    # 是否可见敌人
    enemy_distance: float  # 距离[0,1]，越小越近
    aim_error: float       # 瞄准误差[0,1]，越小越准
    danger_level: float    # 危险程度[0,1]
```

---

### 10. get_policy.py — LLM策略决策（早期原型）

**作用**：调用DeepSeek API进行策略决策（早期原型，已被更完善的decision_advisor.py取代）。

**原理**：通过OpenAI兼容API调用DeepSeek Chat，传入当前状态，让LLM从选项中决策。

---

### 11. opengame.py — 游戏启动与推流

**作用**：在WSL中通过PowerShell启动Windows游戏，并使用ffmpeg捕获窗口画面进行UDP推流。

**原理**：利用ffmpeg的gdigrab（Windows桌面捕获）抓取指定窗口标题的游戏画面，通过UDP推流到WSL端供后续视觉处理。

**类**：`OpenGameTool`

**关键参数**：
```
--game-exe PATH      游戏可执行路径（如cs2.exe）
--game-arg ARG       游戏启动参数
--linux-ip IP        WSL端IP地址（自动探测）
--port PORT          UDP推流端口（默认12345）
--window-title TITLE 窗口标题（auto=自动探测）
--framerate N       推流帧率（默认60）
--bitrate RATE      推流码率（默认8M）
--view-width/height  预览窗口大小
--no-viewer         不启动预览
--ffplay PATH       ffplay路径（Windows侧预览用）
```

---

### 12. test.py — 控制测试脚本

**作用**：简单测试键盘鼠标控制通道是否正常。

**功能**：持续按W前进并向右前方向点击，检测X2侧键可中断退出。

---

## Shell 启动脚本详解

### run.sh — 启动游戏推流
```bash
bash run.sh
```
启动游戏并开始UDP推流。

**关键环境变量**：
| 变量 | 默认值 | 说明 |
|------|--------|------|
| `GAME_EXE` | cs2.exe路径 | Windows游戏可执行文件 |
| `LINUX_IP` | 自动获取 | WSL端接收地址 |
| `PORT` | 12345 | UDP推流端口 |
| `WINDOW_TITLE` | auto | 窗口标题 |
| `NO_VIEWER` | 0 | 设为1不启动预览 |
| `FFPLAY_BIN` | ffplay.exe路径 | Windows侧ffplay |

### start_realtime.sh — 启动视觉实时管道
```bash
bash start_realtime.sh
```
一键启动：Windows推流 + WSL识别 + 可选预览。

**关键环境变量**：
| 变量 | 默认值 | 说明 |
|------|--------|------|
| `WEIGHTS` | best.pt路径 | YOLO模型权重 |
| `IN_STREAM` | UDP:1234 | 输入流地址 |
| `OUT_STREAM` | UDP:2234 | 处理后输出流 |
| `CONF` | 0.25 | YOLO检测置信度阈值 |
| `IMGSZ` | 640 | 检测图像尺寸 |
| `DEVICE` | 0 | GPU设备号 |
| `DETECT_ROI` | 0.00,0.08,1.00,0.84 | 检测区域(相对坐标) |
| `OCR` | 1 | 开启OCR |
| `PREVIEW` | 1 | 显示预览窗口 |

### yolorun.sh — YOLO实时识别启动器
```bash
bash yolorun.sh
```
功能更丰富的YOLO检测启动器，支持半精度推理、批量参数配置。

**特有参数**：
| 变量 | 默认值 | 说明 |
|------|--------|------|
| `HALF` | 1 | FP16半精度推理 |
| `INFER_EVERY` | 1 | 每N帧推理一次 |
| `WORK_SIZE` | 704x396 | 处理分辨率 |
| `LINE_WIDTH` | 2 | 检测框线宽 |
| `OUT_VCODEC` | mpeg2video | 输出编码 |
| `CAPTURE_DRAIN` | 0 | 帧丢弃策略 |
| `UDP_FIFO_SIZE` | 1048576 | UDP缓冲大小 |

### recognize.sh — 联合识别（YOLO+OCR）
```bash
bash recognize.sh
```
同时运行YOLO检测和OCR识别。

**特有参数**：
| 变量 | 默认值 | 说明 |
|------|--------|------|
| `OCR_ENGINE` | pytesseract | OCR引擎 |
| `OCR_ROI` | 0.00,0.78,0.42,0.22 | OCR识别区域 |
| `OCR_WHITELIST` | 0123456789... | OCR白名单字符 |
| `HEAD_RATIO` | 0.30 | 头部占身体高度比 |
| `HEAD_WIDTH_RATIO` | 0.45 | 头部占身体宽度比 |
| `SOURCE` | UDP:1234 | 输入来源 |

### trainsl.sh — 强化学习训练启动器
```bash
bash trainsl.sh
```
启动TD3强化学习训练。

**所有关键参数**（均可通过环境变量覆盖）：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `ENV_MODE` | shared | 环境模式(simple/shared) |
| `EPISODES` | 200 | 训练回合数 |
| `MAX_STEPS` | 200 | 每回合最大步数 |
| `MANAGER_INTERVAL` | 10 | 管理器决策间隔 |
| `SAVE_PATH` | td3_checkpoint.pt | 模型保存路径 |
| `LOAD_PATH` | 同SAVE_PATH | 模型加载路径 |
| `RESUME` | 1 | 继续训练已有模型 |
| `APPLY_ACTIONS` | 1 | 实际执行动作 |
| `MOVE_GAIN` | 120.0 | 鼠标移动增益 |
| `GAMMA` | 0.99 | 折扣因子 |
| `BATCH_SIZE` | 128 | 训练批大小 |
| `REPLAY_SIZE` | 50000 | 经验池容量 |
| `START_STEPS` | 400 | 预热步数 |
| `EXPLORATION_NOISE` | 0.15 | 探索噪声 |
| `POLICY_NOISE` | 0.20 | 策略噪声 |
| `POLICY_DELAY` | 2 | 策略延迟更新 |
| `TAU` | 0.005 | 软更新系数 |
| `SHOOT_THRESHOLD` | 0.35 | 射击阈值 |
| `SHOOT_CENTER_ERROR` | 0.02 | 瞄准中心误差 |
| `CHECKPOINT_EVERY` | 10 | 保存间隔(回合) |
| `REWARD_PLOT_PATH` | reward_curve.png | 奖励曲线图路径 |

### trainrled.sh — RL训练封装（环境变量自动透传）
```bash
bash trainrled.sh
```
在trainsl.sh基础上增加参数回显和透传，便于调试。

### point_aim_train.sh — 本地瞄准模型训练
```bash
bash point_aim_train.sh
```
启动point_aim_trainer.py的训练流程。

**特有参数**：
| 变量 | 默认值 | 说明 |
|------|--------|------|
| `MOVE_GAIN_X` | 2500 | X方向移动增益 |
| `MOVE_GAIN_Y` | 500 | Y方向移动增益 |
| `MAX_MOVE_X` | 1000 | 最大X移动像素 |
| `MAX_MOVE_Y` | 500 | 最大Y移动像素 |
| `SHOOT_CENTER_ERROR` | 0.1 | 开火中心误差 |
| `TRAIN_ONLY` | 0 | 仅训练不动鼠标 |
| `ACTOR_LR` | 1e-3 | Actor学习率 |
| `CRITIC_LR` | 1e-3 | Critic学习率 |
| `HIDDEN_DIM` | 64 | 隐藏层维度 |
| `BUFFER_SIZE` | 1024 | 经验池容量 |
| `SEARCH_STEP` | 500 | 搜索步数上限 |

---

## 数据流全景

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Windows 侧                                    │
│                                                                     │
│  CS2 Game ──→ ffmpeg gdigrab ──→ UDP Stream ─────────────────┐     │
│      ↑                               (port 12345)            │     │
│      │                                                        │     │
│  Win32 API ←── PowerShell Listener ←── TCP Socket ←──────────┤     │
│  (keybd_event,    (port 60000)               命令文本         │     │
│   mouse_event)                                               │     │
└──────────────────────────────────────────────────────────────┘     │
                                                                      │
                              │ WSL 侧(Ubuntu)                        │
                              ▼                                       │
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  ffmpeg接收UDP流 ──→ 视觉感知流水线 ──→ 共享状态JSON                  │
│                        (YOLO + OCR + Qwen)   /tmp/cs_rl_*.json      │
│                                                     │               │
│  决策控制层 ←─────────────────────────────────────────┘               │
│  ├─ Manager (LLM/规则): 选择search/fight/take_cover                  │
│  ├─ TD3 Agent: 19维状态 → 3维动作                                   │
│  └─ 奖励计算: get_reward()                                          │
│                                                     │               │
│  动作执行层 ──→ Socket ──→ Windows PowerShell ──→ Win32 API          │
│  (actions.py)                                                       │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 模型文件说明

| 文件 | 说明 |
|------|------|
| `yolo11n.pt` | YOLOv11 Nano预训练权重（Ultralytics官方） |
| `point_aim_net.pt` | 本地瞄准模型（从头训练） |
| `point_aim_net_best.pt` | 本地瞄准模型（最佳版本） |
| `point_aim_net_resume.pt` | 本地瞄准模型（恢复训练） |
| `point_aim_net_resume_best.pt` | 本地瞄准模型（恢复训练+最佳） |
| `visual_recognition/runs/*/weights/best.pt` | YOLO训练产出的最佳权重 |
| `td3_checkpoint.pt` | TD3强化学习模型检查点 |

---

## 常见运行模式

### 模式1：纯仿真训练（不需要游戏）
```bash
ENV_MODE=simple EPISODES=100 bash trainsl.sh
```

### 模式2：真实游戏训练
```bash
# 终端1：启动游戏推流
bash run.sh

# 终端2：启动视觉管道
bash start_realtime.sh

# 终端3：启动强化学习训练
bash trainsl.sh
```

### 模式3：仅视觉识别测试
```bash
SOURCE=udp://192.168.221.36:1234 PYTHON_BIN=python bash yolorun.sh
```

### 模式4：实时决策辅助
```bash
python decision_advisor.py --debug-aim --aim-model-path point_aim_net_resume_best.pt
```

### 模式5：本地瞄准模型训练
```bash
bash point_aim_train.sh
```

---

## 奖励曲线解读

训练过程中生成的 `reward_curve.png` 显示奖励变化趋势：

- **持续上升**：策略在改进，学习有效
- **震荡但趋势向上**：正常，强化学习固有波动
- **长期不上升/下降**：可能需要调整学习率、噪声参数或奖励权重
- **突然暴跌**：可能是环境变化或参数设置问题

---

## 调试技巧

1. **X2侧键紧急停止**：任何循环中按下鼠标侧键X2即可安全停止
2. **流延迟测量**：系统自动测量端到端延迟（移动鼠标→观察到准星移动），用于同步
3. **日志分析**：训练日志包含每个分量的奖励值，便于调参
4. **--train-only**：在不动鼠标的情况下训练模型，安全调试
5. **ROI过滤**：DETECT_ROI参数排除HUD区域误检，避免将UI元素识别为敌人
