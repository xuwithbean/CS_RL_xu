#!/usr/bin/env python3
"""白底点流实时 reward 瞄准训练器。

这个脚本直接读取 `trainimg.py` 写出的共享状态 JSON：
- `centers`: 当前检测到的点坐标列表
- `centers_ref_w` / `centers_ref_h`: 参考坐标系大小

训练目标：
- 通过 reward 机制让模型学会把点尽量移到屏幕中心。
- 点越接近中心，reward 越高。
- 靠近中心时开火给额外 reward；远离中心时开火给惩罚。

它是一个自包含脚本，内部同时包含：
- 小型神经网络
- 在线增量训练
- 实时鼠标控制

用法示例：
  python point_aim_trainer.py --shared-state /tmp/cs_rl_runtime_state.json
  python point_aim_trainer.py --train-only --save-path point_aim_net.pt
"""
from __future__ import annotations

import argparse
import json
import math
import random
import signal
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
except Exception as exc:  # pragma: no cover
    raise RuntimeError("需要安装 torch 才能运行 point_aim_trainer.py") from exc

from actions import m_actions


ACTION_DIM = 2  # [move_x, move_y]


class PointAimNet(nn.Module):
    def __init__(self, input_dim: int = 3, hidden_dim: int = 64):
        super().__init__()
        # 线性策略：直接把点到中心的状态映射为鼠标动作，便于学习“距离 -> 操作”的关系。
        self.net = nn.Linear(input_dim, ACTION_DIM)
        nn.init.zeros_(self.net.weight)
        nn.init.zeros_(self.net.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        return torch.tanh(out[:, :2])


class CriticNet(nn.Module):
    def __init__(self, state_dim: int = 3, action_dim: int = ACTION_DIM, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([state, action], dim=1))


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = max(1, int(capacity))
        self.data: Deque[tuple[np.ndarray, np.ndarray, float, np.ndarray, float]] = deque(maxlen=self.capacity)

    def add(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray, done: float) -> None:
        self.data.append((state.astype(np.float32), action.astype(np.float32), float(reward), next_state.astype(np.float32), float(done)))

    def sample(self, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        batch = random.sample(self.data, k=min(len(self.data), int(batch_size)))
        states = np.stack([item[0] for item in batch], axis=0)
        actions = np.stack([item[1] for item in batch], axis=0)
        rewards = np.asarray([item[2] for item in batch], dtype=np.float32)
        next_states = np.stack([item[3] for item in batch], axis=0)
        dones = np.asarray([item[4] for item in batch], dtype=np.float32)
        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        return len(self.data)


def read_payload(path: str) -> dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def read_centers(payload: dict[str, Any]) -> tuple[list[tuple[str, int, int, float]], Optional[tuple[int, int]]]:
    centers: list[tuple[str, int, int, float]] = []
    for item in list((payload or {}).get("centers") or []):
        if not isinstance(item, dict):
            continue
        try:
            centers.append((str(item.get("name", "")), int(item.get("cx", 0)), int(item.get("cy", 0)), float(item.get("conf", 0.0))))
        except Exception:
            continue
    ref_w = int((payload or {}).get("centers_ref_w") or 0)
    ref_h = int((payload or {}).get("centers_ref_h") or 0)
    ref_size = (ref_w, ref_h) if ref_w > 0 and ref_h > 0 else None
    return centers, ref_size


def select_target(centers: list[tuple[str, int, int, float]], ref_size: Optional[tuple[int, int]]) -> Optional[tuple[str, int, int, float]]:
    if not centers or ref_size is None:
        return None
    ref_w, ref_h = ref_size
    cx0 = ref_w / 2.0
    cy0 = ref_h / 2.0
    return min(
        centers,
        key=lambda item: ((float(item[1]) - cx0) ** 2 + (float(item[2]) - cy0) ** 2) ** 0.5,
    )


def build_state(target: tuple[str, int, int, float], ref_size: tuple[int, int]) -> tuple[np.ndarray, float]:
    _, tx, ty, conf = target
    ref_w, ref_h = ref_size
    cx0 = ref_w / 2.0
    cy0 = ref_h / 2.0
    ndx = (float(tx) - cx0) / max(1.0, cx0)
    ndy = (float(ty) - cy0) / max(1.0, cy0)
    dist = math.sqrt(ndx * ndx + ndy * ndy)
    aim_error = min(1.0, dist)
    state = np.asarray([ndx, ndy, aim_error], dtype=np.float32)
    return state, aim_error


@dataclass
class TrainStats:
    steps: int = 0
    seen: int = 0
    loss: float = 0.0
    reward: float = 0.0
    avg_reward: float = 0.0


def select_action(model: PointAimNet, state: np.ndarray, device: torch.device, noise_std: float) -> np.ndarray:
    with torch.no_grad():
        state_t = torch.from_numpy(state.astype(np.float32)).unsqueeze(0).to(device)
        action = model(state_t).squeeze(0).cpu().numpy()
    noise = np.random.normal(loc=0.0, scale=max(0.0, float(noise_std)), size=(ACTION_DIM,)).astype(np.float32)
    action = action + noise
    action[0] = float(np.clip(action[0], -1.0, 1.0))
    action[1] = float(np.clip(action[1], -1.0, 1.0))
    return action.astype(np.float32)


def action_to_command(action: np.ndarray, info: dict[str, Any], args: argparse.Namespace) -> tuple[str, int, int, bool]:
    aim_error = float(info.get("aim_error", 1.0))
    max_move_x = int(max(1, int(args.max_move_x)))
    max_move_y = int(max(1, int(args.max_move_y)))
    move_scale_x = float(args.move_gain_x)
    move_scale_y = float(args.move_gain_y)
    # 使用 float 计算并保留方向；避免 int 截断导致 0，如果缩放后非零但绝对值<1，则至少移动 1 像素
    float_move_x = float(action[0]) * move_scale_x
    float_move_y = float(action[1]) * move_scale_y
    clipped_x = float(np.clip(float_move_x, -max_move_x, max_move_x))
    clipped_y = float(np.clip(float_move_y, -max_move_y, max_move_y))
    raw_move_x = int(clipped_x) if abs(clipped_x) >= 1.0 else (int(np.sign(clipped_x)) if abs(clipped_x) > 0.0 else 0)
    raw_move_y = int(clipped_y) if abs(clipped_y) >= 1.0 else (int(np.sign(clipped_y)) if abs(clipped_y) > 0.0 else 0)
    # 瞄准时位移完全由模型输出控制，不再强行按 ndx/ndy 修正方向。
    move_x = int(np.clip(raw_move_x, -max_move_x, max_move_x))
    move_y = int(np.clip(raw_move_y, -max_move_y, max_move_y))

    center_error = float(args.shoot_center_error)
    if aim_error <= center_error:
        # 到中心范围就不再移动，标记为不需要射击（训练里以到达中心作为回合成功）
        move_x = 0
        move_y = 0
        do_shoot = False
    else:
        do_shoot = False
    return "continuous", move_x, move_y, do_shoot


def wait_for_step_transition(
    shared_state: str,
    base_state: np.ndarray,
    base_aim_error: float,
    target_name: str,
    settle_sec: float,
    shift_threshold: float,
    consecutive_frames: int,
    poll_sec: float,
    timeout_sec: float,
) -> tuple[np.ndarray, float, bool, float]:
    """等待点在连续若干帧中都超过位移阈值，认为动作真正生效。"""
    deadline = time.monotonic() + float(timeout_sec)
    last_state = base_state
    last_aim_error = float(base_aim_error)
    stable_frames = 0
    if settle_sec > 0:
        time.sleep(min(float(settle_sec), max(0.0, float(timeout_sec))))
    while time.monotonic() < deadline:
        payload = read_payload(shared_state)
        centers, ref_size = read_centers(payload)
        if not centers or ref_size is None:
            time.sleep(max(0.005, float(poll_sec)))
            continue
        if target_name:
            matched = next((c for c in centers if c[0] == target_name), None)
            target = matched or select_target(centers, ref_size)
        else:
            target = select_target(centers, ref_size)
        if target is None:
            time.sleep(max(0.005, float(poll_sec)))
            continue
        curr_state, curr_aim_error = build_state(target, ref_size)
        last_state = curr_state
        last_aim_error = curr_aim_error
        shift = float(np.linalg.norm(curr_state[:2] - base_state[:2]))
        if shift >= float(shift_threshold):
            stable_frames += 1
            if stable_frames >= max(1, int(consecutive_frames)):
                elapsed = float(timeout_sec) - max(0.0, deadline - time.monotonic())
                return curr_state, curr_aim_error, True, max(0.0, elapsed)
        else:
            stable_frames = 0
        time.sleep(max(0.005, float(poll_sec)))
    return last_state, last_aim_error, False, float(timeout_sec)


def read_current_state(shared_state: str, target_name: str = "") -> tuple[Optional[np.ndarray], Optional[float]]:
    payload = read_payload(shared_state)
    centers, ref_size = read_centers(payload)
    if not centers or ref_size is None:
        return None, None
    if target_name:
        matched = next((c for c in centers if c[0] == target_name), None)
        target = matched or select_target(centers, ref_size)
    else:
        target = select_target(centers, ref_size)
    if target is None:
        return None, None
    state, aim_error = build_state(target, ref_size)
    return state, aim_error


def make_policy_target(state: np.ndarray, shoot_center_error: float) -> np.ndarray:
    ndx = float(state[0])
    ndy = float(state[1])
    move_target = np.asarray([ndx, ndy], dtype=np.float32)
    move_target = np.clip(move_target, -1.0, 1.0)
    return np.asarray([move_target[0], move_target[1]], dtype=np.float32)


def compute_reward(
    round_start_state: np.ndarray,
    round_end_state: np.ndarray,
    reward_scale: float,
    shoot_center_error: float,
) -> tuple[float, dict[str, float]]:
    start_x = float(round_start_state[0])
    start_y = float(round_start_state[1])
    end_x = float(round_end_state[0])
    end_y = float(round_end_state[1])
    eps = max(1e-6, float(shoot_center_error))

    def axis_score(start_v: float, end_v: float) -> float:
        if abs(start_v) <= eps:
            return float(-(abs(end_v) / eps)) if abs(end_v) > eps else 0.0
        return float((-np.sign(start_v) * (end_v - start_v)) / max(abs(start_v), eps))

    score_x = axis_score(start_x, end_x)
    score_y = axis_score(start_y, end_y)
    start_dist = float(math.sqrt(start_x * start_x + start_y * start_y))
    center_dist = float(math.sqrt(end_x * end_x + end_y * end_y))

    progress = 1.0 - (center_dist / max(1e-6, start_dist))
    directional = 0.5 * (score_x + score_y)
    reward = float(reward_scale) * (0.6 * directional + 0.4 * progress)
    if center_dist <= float(shoot_center_error):
        center_ratio = 1.0 - (center_dist / max(eps, float(shoot_center_error)))
        reward += float(reward_scale) * (4.0 + 6.0 * center_ratio)
    elif center_dist > start_dist:
        reward -= float(reward_scale) * ((center_dist / max(1e-6, start_dist)) - 1.0)

    details = {
        "start_x": start_x,
        "start_y": start_y,
        "end_x": end_x,
        "end_y": end_y,
        "start_dist": start_dist,
        "score_x": score_x,
        "score_y": score_y,
        "progress": progress,
        "center_dist": center_dist,
    }
    return float(reward), details


def policy_update(
    model: PointAimNet,
    optimizer: torch.optim.Optimizer,
    states: list[np.ndarray],
    targets: list[np.ndarray],
    weights: list[float],
    device: torch.device,
) -> float:
    if not states:
        return 0.0
    states_t = torch.from_numpy(np.asarray(states, dtype=np.float32)).to(device)
    targets_t = torch.from_numpy(np.asarray(targets, dtype=np.float32)).to(device)
    weights_t = torch.from_numpy(np.asarray(weights, dtype=np.float32)).to(device).unsqueeze(1)
    pred = model(states_t)
    loss = ((pred - targets_t) ** 2 * weights_t).mean()
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
    optimizer.step()
    return float(loss.item())


def random_reset_action(center_error: float, reset_pixels: int) -> tuple[int, int]:
    # 双轴同时随机：x 为 +/-[500,1200]，y 为 +/-[200,500]
    x = random.randint(500, 1200) * random.choice([-1, 1])
    y = random.randint(200, 500) * random.choice([-1, 1])
    return x, y


def train_step(
    actor: PointAimNet,
    target_actor: PointAimNet,
    critic: CriticNet,
    target_critic: CriticNet,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    replay: ReplayBuffer,
    batch_size: int,
    gamma: float,
    tau: float,
    device: torch.device,
) -> float:
    if len(replay) < max(1, int(batch_size)):
        return 0.0
    states, actions, rewards, next_states, dones = replay.sample(batch_size)
    states_t = torch.from_numpy(states).to(device)
    actions_t = torch.from_numpy(actions).to(device)
    rewards_t = torch.from_numpy(rewards).to(device)
    next_states_t = torch.from_numpy(next_states).to(device)
    dones_t = torch.from_numpy(dones).to(device)

    q_values = critic(states_t, actions_t).squeeze(1)
    with torch.no_grad():
        next_actions = target_actor(next_states_t)
        next_q = target_critic(next_states_t, next_actions).squeeze(1)
        target = rewards_t + float(gamma) * next_q * (1.0 - dones_t)

    critic_loss = nn.functional.mse_loss(q_values, target)
    critic_optimizer.zero_grad()
    critic_loss.backward()
    torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=5.0)
    critic_optimizer.step()

    pred_actions = actor(states_t)
    actor_loss = -critic(states_t, pred_actions).mean()
    actor_optimizer.zero_grad()
    actor_loss.backward()
    torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=5.0)
    actor_optimizer.step()

    with torch.no_grad():
        for target_p, p in zip(target_actor.parameters(), actor.parameters()):
            target_p.data.mul_(1.0 - float(tau)).add_(float(tau) * p.data)
        for target_p, p in zip(target_critic.parameters(), critic.parameters()):
            target_p.data.mul_(1.0 - float(tau)).add_(float(tau) * p.data)

    return float((critic_loss.item() + actor_loss.item()) / 2.0)


def save_model(model: PointAimNet, path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), p)


def default_best_path(save_path: str) -> str:
    p = Path(save_path or "point_aim_net.pt")
    stem = p.stem or "point_aim_net"
    suffix = p.suffix or ".pt"
    return str(p.with_name(f"{stem}_best{suffix}"))


def default_plot_path(save_path: str) -> str:
    p = Path(save_path or "reward_curve.png")
    stem = p.stem or "reward_curve"
    suffix = p.suffix or ".png"
    return str(p.with_name(f"{stem}_best{suffix}"))


def load_model(path: str, device: torch.device, hidden_dim: int = 64) -> PointAimNet:
    model = PointAimNet(hidden_dim=hidden_dim).to(device)
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def epsilon_for_step(step: int, epsilon_start: float, epsilon_end: float, epsilon_decay: float) -> float:
    return max(float(epsilon_end), float(epsilon_start) * (float(epsilon_decay) ** float(step)))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a neural network to center a point and shoot near center")
    p.add_argument("--shared-state", type=str, default=str(Path("/tmp/cs_rl_runtime_state.json")))
    p.add_argument("--save-path", type=str, default="point_aim_net.pt")
    p.add_argument("--best-save-path", type=str, default="", help="最好模型保存路径；为空则自动生成 *_best.pt")
    p.add_argument("--reward-plot-path", type=str, default="reward_curve.png", help="当前模型保存时绘制的 reward 曲线图路径")
    p.add_argument("--best-reward-plot-path", type=str, default="", help="最好模型保存时绘制的 reward 曲线图路径；为空则自动生成 *_best.png")
    p.add_argument("--load-path", type=str, default="")
    p.add_argument("--train-only", action="store_true", help="只训练不发鼠标动作")
    p.add_argument("--move-gain-x", type=float, default=2500.0)
    p.add_argument("--move-gain-y", type=float, default=500.0)
    p.add_argument("--max-move-x", type=int, default=1000, help="单次鼠标 x 方向位移上限")
    p.add_argument("--max-move-y", type=int, default=500, help="单次鼠标 y 方向位移上限")
    p.add_argument("--max-step", type=int, default=400)
    p.add_argument("--poll-sec", type=float, default=0.03)
    p.add_argument("--search-step", type=int, default=500, help="无点时左右搜索的单次鼠标位移")
    p.add_argument("--search-interval-sec", type=float, default=0.20, help="无点时左右搜索的间隔秒数")
    p.add_argument("--step-shift-threshold", type=float, default=0.008, help="动作被认为已生效的最小归一化位移变化")
    p.add_argument("--step-consecutive-frames", type=int, default=1, help="连续多少帧超过阈值才认为一步真正生效")
    p.add_argument("--action-settle-sec", type=float, default=0.06, help="鼠标动作后额外等待画面反馈的最短时间")
    p.add_argument("--step-wait-timeout-sec", type=float, default=0.22, help="等待一次动作生效的最长时间")
    p.add_argument("--step-wait-poll-sec", type=float, default=0.01, help="等待动作生效时的采样间隔")
    p.add_argument("--shoot-center-error", type=float, default=0.1, help="到达此归一化误差视为到达中心并结束回合")
    p.add_argument("--episode-reset-pixels", type=int, default=40, help="每轮成功后随机把点移离中心（x: +/-500-1200, y: +/-200-500；参数保持兼容但当前使用内置范围）")
    p.add_argument("--round-wait-sec", type=float, default=2.0, help="每轮发出一次移动后等待的秒数")
    p.add_argument("--round-reward-scale", type=float, default=10.0, help="单轮距离比例奖励系数")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--buffer-size", type=int, default=1024)
    p.add_argument("--gamma", type=float, default=0.98)
    p.add_argument("--noise-start", type=float, default=0.08, help="连续动作探索噪声初始值")
    p.add_argument("--noise-end", type=float, default=0.01, help="连续动作探索噪声最小值")
    p.add_argument("--noise-decay", type=float, default=0.999, help="连续动作探索噪声衰减")
    p.add_argument("--tau", type=float, default=0.01, help="目标网络软更新系数")
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--reward-window", type=int, default=200, help="用于判断最好模型的滑动窗口步数")
    p.add_argument("--actor-lr", type=float, default=1e-3)
    p.add_argument("--critic-lr", type=float, default=1e-3)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--invert-x", action="store_true")
    p.add_argument("--invert-y", action="store_true")
    p.add_argument("--target-name", type=str, default="", help="可选：只跟踪指定目标名")
    return p.parse_args()


def plot_reward_curve(reward_history: list[float], plot_path: str, title: str, best_score: float | None = None) -> None:
    if not reward_history:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[point-aim] 无法绘制 reward 曲线: {exc}")
        return

    p = Path(plot_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    xs = list(range(1, len(reward_history) + 1))
    window = max(1, min(100, len(reward_history)))
    smooth = [sum(reward_history[max(0, i - window + 1): i + 1]) / len(reward_history[max(0, i - window + 1): i + 1]) for i in range(len(reward_history))]

    fig = plt.figure(figsize=(10, 4.5), dpi=120)
    ax = fig.add_subplot(111)
    ax.plot(xs, reward_history, color="#7f8c8d", linewidth=1.0, alpha=0.35, label="reward")
    ax.plot(xs, smooth, color="#2980b9", linewidth=1.8, label=f"smoothed({window})")
    ax.set_title(title)
    ax.set_xlabel("Step")
    ax.set_ylabel("Reward")
    ax.grid(True, alpha=0.25)
    if best_score is not None:
        ax.axhline(best_score, color="#c0392b", linestyle="--", linewidth=1.2, label=f"best={best_score:.3f}")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(p, dpi=120)
    plt.close(fig)


def maybe_save_current(model: PointAimNet, reward_history: list[float], args: argparse.Namespace) -> None:
    try:
        save_model(model, str(args.save_path))
        print(f"[point-aim] current model saved to {args.save_path}")
    except Exception as exc:
        print(f"[point-aim] current model save failed: {exc}")
        return
    try:
        plot_reward_curve(reward_history, str(args.reward_plot_path), "Point Aim Reward Curve")
        print(f"[point-aim] reward curve saved to {args.reward_plot_path}")
    except Exception as exc:
        print(f"[point-aim] reward curve save failed: {exc}")


def maybe_save_best(model: PointAimNet, reward_history: list[float], args: argparse.Namespace, best_score: float) -> None:
    best_plot_path = str(args.best_reward_plot_path or "").strip() or default_plot_path(str(args.reward_plot_path))
    try:
        save_model(model, str(args.best_save_path or default_best_path(str(args.save_path))))
        print(f"[point-aim] best model saved")
    except Exception as exc:
        print(f"[point-aim] best model save failed: {exc}")
        return
    try:
        plot_reward_curve(reward_history, best_plot_path, "Point Aim Best Reward Curve", best_score=best_score)
        print(f"[point-aim] best reward curve saved to {best_plot_path}")
    except Exception as exc:
        print(f"[point-aim] best reward curve save failed: {exc}")


def main() -> int:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    best_save_path = str(args.best_save_path or "").strip() or default_best_path(str(args.save_path))
    if args.load_path:
        model = load_model(args.load_path, device, hidden_dim=int(args.hidden_dim))
        print(f"[point-aim] loaded model from {args.load_path}")
    else:
        model = PointAimNet(hidden_dim=args.hidden_dim).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.actor_lr))
    controller = m_actions()
    stats = TrainStats()
    stop_requested = False
    best_score = float("-inf")
    # track best single-round reward (max), save model when exceeded
    best_single_score = float("-inf")
    reward_history: list[float] = []
    total_steps = 0
    last_search_move_t = 0.0
    search_direction = random.choice([-1, 1])
    next_action_after = 0.0
    episode_start = time.monotonic()
    episode_states: list[np.ndarray] = []
    episode_targets: list[np.ndarray] = []
    episode_weights: list[float] = []

    def _request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)

    def _finish_episode(success: bool, last_reward: float | None = None) -> None:
        nonlocal best_score, best_single_score, episode_start, episode_states, episode_targets, episode_weights, next_action_after
        if episode_states:
            loss = policy_update(model, optimizer, episode_states, episode_targets, episode_weights, device)
            stats.loss = loss
        episode_states = []
        episode_targets = []
        episode_weights = []
        episode_start = time.monotonic()
        if success and not args.train_only:
            reset_dx, reset_dy = random_reset_action(float(args.shoot_center_error), int(args.episode_reset_pixels))
            print(f"[point-aim] episode finished: success; reset move=({reset_dx},{reset_dy}) -> waiting 2.0s")
            controller.mouse_move(reset_dx, reset_dy)
            # 阻止在短时间内重复发送随机重置，等待 2 秒再允许下一次动作
            next_action_after = time.monotonic() + 2.0
            time.sleep(2.0)

        if reward_history:
            window = max(1, int(args.reward_window))
            if len(reward_history) >= window:
                stats.avg_reward = float(sum(reward_history[-window:]) / float(window))
            else:
                stats.avg_reward = float(sum(reward_history) / max(1, len(reward_history)))
            if stats.avg_reward > best_score:
                best_score = float(stats.avg_reward)
                maybe_save_best(model, reward_history, args, best_score)
        # 保存单轮最高 reward 对应模型（在 policy_update 之后，这样保存的是学习后模型）
        try:
            if last_reward is not None and float(last_reward) > float(best_single_score):
                best_single_score = float(last_reward)
                print(f"[point-aim] new best single-round reward={best_single_score:.4f} -> saving model")
                maybe_save_best(model, reward_history, args, best_single_score)
        except Exception:
            pass

    try:
        while True:
            if stop_requested:
                break

            now = time.monotonic()
            if now < next_action_after:
                time.sleep(min(max(0.005, float(args.poll_sec)), max(0.0, next_action_after - now)))
                continue

            payload = read_payload(args.shared_state)
            centers, ref_size = read_centers(payload)
            if not centers or ref_size is None:
                if not args.train_only:
                    now = time.monotonic()
                    if last_search_move_t <= 0.0:
                        last_search_move_t = now - float(args.search_interval_sec)
                    if (now - last_search_move_t) >= float(args.search_interval_sec):
                        move_dx = int(max(1, abs(int(args.search_step)))) * search_direction
                        controller.mouse_move(move_dx, 0)
                        last_search_move_t = now
                        next_action_after = time.monotonic() + max(float(args.action_settle_sec), float(args.step_wait_poll_sec))
                time.sleep(max(0.01, float(args.poll_sec)))
                continue

            if args.target_name:
                matched = next((c for c in centers if c[0] == args.target_name), None)
                target = matched or select_target(centers, ref_size)
            else:
                target = select_target(centers, ref_size)
            if target is None:
                if not args.train_only:
                    now = time.monotonic()
                    if last_search_move_t <= 0.0:
                        last_search_move_t = now - float(args.search_interval_sec)
                    if (now - last_search_move_t) >= float(args.search_interval_sec):
                        move_dx = int(max(1, abs(int(args.search_step)))) * search_direction
                        controller.mouse_move(move_dx, 0)
                        last_search_move_t = now
                        next_action_after = time.monotonic() + max(float(args.action_settle_sec), float(args.step_wait_poll_sec))
                time.sleep(max(0.01, float(args.poll_sec)))
                continue

            last_search_move_t = 0.0
            state, aim_error = build_state(target, ref_size)
            stats.seen += 1

            if aim_error <= float(args.shoot_center_error):
                _finish_episode(True, None)
                total_steps += 1
                stats.steps += 1
                if total_steps % max(1, int(args.save_every)) == 0:
                    maybe_save_current(model, reward_history, args)
                continue

            noise_std = epsilon_for_step(total_steps, float(args.noise_start), float(args.noise_end), float(args.noise_decay))
            action = select_action(model, state, device, noise_std)
            _, move_x, move_y, do_shoot = action_to_command(
                action,
                {"ndx": float(state[0]), "ndy": float(state[1]), "aim_error": aim_error},
                args,
            )

            target_action = make_policy_target(state, float(args.shoot_center_error))

            # 单轮策略：只执行一次移动，等待固定时长后计算奖励并结束该轮。
            if move_x != 0 or move_y != 0:
                if not args.train_only:
                    controller.mouse_move(move_x, move_y)
            round_wait = max(0.0, float(args.round_wait_sec))
            if round_wait > 0.0:
                time.sleep(round_wait)
            next_state, next_aim_error = read_current_state(str(args.shared_state), str(args.target_name or ""))
            if next_state is None or next_aim_error is None:
                next_state, next_aim_error = state, aim_error
            reward, reward_info = compute_reward(
                state,
                next_state,
                float(args.round_reward_scale),
                float(args.shoot_center_error),
            )
            step_weight = 1.0 + max(-0.6, min(2.0, reward / max(1e-6, float(args.round_reward_scale))))
            episode_states.append(state)
            episode_targets.append(target_action)
            episode_weights.append(step_weight)
            reward_history.append(reward)
            stats.reward = reward
            total_steps += 1
            stats.steps += 1
            if len(reward_history) >= max(1, int(args.reward_window)):
                stats.avg_reward = float(sum(reward_history[-int(args.reward_window):]) / float(int(args.reward_window)))
            else:
                stats.avg_reward = float(sum(reward_history) / max(1, len(reward_history)))
            if stats.steps % max(1, int(args.log_every)) == 0:
                print(
                    f"[point-aim] step={stats.steps} seen={stats.seen} loss={stats.loss:.6f} reward={reward:.4f} avg_reward={stats.avg_reward:.4f} noise={noise_std:.4f} action=({move_x},{move_y}) "
                    f"start_xy=({reward_info['start_x']:.4f},{reward_info['start_y']:.4f}) end_xy=({reward_info['end_x']:.4f},{reward_info['end_y']:.4f}) "
                    f"axis=({reward_info['score_x']:.4f},{reward_info['score_y']:.4f}) center_dist={reward_info['center_dist']:.4f}"
                )
            _finish_episode(bool(next_aim_error <= float(args.shoot_center_error)), float(reward))
            if total_steps % max(1, int(args.save_every)) == 0:
                maybe_save_current(model, reward_history, args)
            next_action_after = time.monotonic() + max(float(args.action_settle_sec), float(args.step_wait_poll_sec))
            continue

            time.sleep(max(0.01, float(args.poll_sec)))

            if controller.is_interrupt_x2_pressed():
                controller.stop()
                break
    except KeyboardInterrupt:
        pass
    finally:
        _finish_episode(False, None)
        maybe_save_current(model, reward_history, args)
        if best_score > float("-inf"):
            maybe_save_best(model, reward_history, args, best_score)
        try:
            plot_reward_curve(reward_history, str(args.reward_plot_path), "Point Aim Reward Curve")
        except Exception:
            pass
        try:
            controller.stop()
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
