#!/usr/bin/env python3
"""白底点流实时瞄准训练器。

这个脚本直接读取 `trainimg.py` 写出的共享状态 JSON：
- `centers`: 当前检测到的点坐标列表
- `centers_ref_w` / `centers_ref_h`: 参考坐标系大小

训练目标很简单：
- 点在画面左侧，就输出向左的鼠标移动；右侧就向右。
- 点在上方，就向上；下方就向下。
- 点足够接近中心时，输出开火。

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
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Iterable, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
except Exception as exc:  # pragma: no cover
    raise RuntimeError("需要安装 torch 才能运行 point_aim_trainer.py") from exc

from actions import m_actions


class PointAimNet(nn.Module):
    def __init__(self, input_dim: int = 3, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = self.net(x)
        move = torch.tanh(raw[:, :2])
        shoot = raw[:, 2:3]
        return torch.cat([move, shoot], dim=1)


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


def build_sample(target: tuple[str, int, int, float], ref_size: tuple[int, int], shoot_center_error: float) -> tuple[np.ndarray, np.ndarray, float]:
    _, tx, ty, conf = target
    ref_w, ref_h = ref_size
    cx0 = ref_w / 2.0
    cy0 = ref_h / 2.0
    ndx = (float(tx) - cx0) / max(1.0, cx0)
    ndy = (float(ty) - cy0) / max(1.0, cy0)
    dist = math.sqrt(ndx * ndx + ndy * ndy)
    aim_error = min(1.0, dist)

    # 监督目标：线性比例控制 + 近中心开火
    target_move_x = float(ndx)
    target_move_y = float(ndy)
    shoot = 1.0 if aim_error <= float(shoot_center_error) else 0.0
    shoot_logit = (shoot * 6.0) - 3.0

    x = np.asarray([ndx, ndy, aim_error], dtype=np.float32)
    y = np.asarray([target_move_x, target_move_y, shoot_logit], dtype=np.float32)
    return x, y, aim_error


@dataclass
class TrainStats:
    steps: int = 0
    seen: int = 0
    loss: float = 0.0
    shoot_rate: float = 0.0


def train_step(model: PointAimNet, optimizer: torch.optim.Optimizer, batch_x: np.ndarray, batch_y: np.ndarray, device: torch.device) -> float:
    x = torch.from_numpy(batch_x).to(device)
    y = torch.from_numpy(batch_y).to(device)
    pred = model(x)
    loss_move = nn.functional.mse_loss(pred[:, :2], y[:, :2])
    loss_shoot = nn.functional.mse_loss(pred[:, 2], y[:, 2])
    loss = loss_move + 0.5 * loss_shoot
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return float(loss.item())


def infer_action(model: PointAimNet, x: np.ndarray, move_gain: float, max_step: int, invert_x: bool, invert_y: bool) -> tuple[int, int, float]:
    with torch.no_grad():
        inp = torch.from_numpy(x[None, :]).float()
        out = model(inp).cpu().numpy()[0]
    mvx_norm = float(out[0])
    mvy_norm = float(out[1])
    shoot_logit = float(out[2])
    if invert_x:
        mvx_norm = -mvx_norm
    if invert_y:
        mvy_norm = -mvy_norm
    move_x = int(max(-max_step, min(max_step, round(mvx_norm * move_gain))))
    move_y = int(max(-max_step, min(max_step, round(mvy_norm * move_gain))))
    shoot_prob = 1.0 / (1.0 + math.exp(-shoot_logit))
    return move_x, move_y, shoot_prob


def save_model(model: PointAimNet, path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), p)


def load_model(path: str, device: torch.device) -> PointAimNet:
    model = PointAimNet().to(device)
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a neural network to center a point and shoot near center")
    p.add_argument("--shared-state", type=str, default=str(Path("/tmp/cs_rl_runtime_state.json")))
    p.add_argument("--save-path", type=str, default="point_aim_net.pt")
    p.add_argument("--load-path", type=str, default="")
    p.add_argument("--train-only", action="store_true", help="只训练不发鼠标动作")
    p.add_argument("--move-gain", type=float, default=300.0)
    p.add_argument("--max-step", type=int, default=400)
    p.add_argument("--poll-sec", type=float, default=0.03)
    p.add_argument("--shoot-center-error", type=float, default=0.04)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--buffer-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--invert-x", action="store_true")
    p.add_argument("--invert-y", action="store_true")
    p.add_argument("--target-name", type=str, default="", help="可选：只跟踪指定目标名")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")

    if args.load_path:
        model = load_model(args.load_path, device)
        print(f"[point-aim] loaded model from {args.load_path}")
    else:
        model = PointAimNet(hidden_dim=args.hidden_dim).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr))
    buffer_x: Deque[np.ndarray] = deque(maxlen=int(args.buffer_size))
    buffer_y: Deque[np.ndarray] = deque(maxlen=int(args.buffer_size))

    controller = m_actions()
    stats = TrainStats()

    try:
        while True:
            payload = read_payload(args.shared_state)
            centers, ref_size = read_centers(payload)
            if not centers or ref_size is None:
                time.sleep(max(0.01, float(args.poll_sec)))
                continue

            if args.target_name:
                matched = next((c for c in centers if c[0] == args.target_name), None)
                target = matched or select_target(centers, ref_size)
            else:
                target = select_target(centers, ref_size)
            if target is None:
                time.sleep(max(0.01, float(args.poll_sec)))
                continue

            x, y, aim_error = build_sample(target, ref_size, args.shoot_center_error)
            buffer_x.append(x)
            buffer_y.append(y)
            stats.seen += 1

            if len(buffer_x) >= int(args.batch_size):
                batch_x = np.stack(list(buffer_x)[-int(args.batch_size):], axis=0)
                batch_y = np.stack(list(buffer_y)[-int(args.batch_size):], axis=0)
                loss = train_step(model, optimizer, batch_x, batch_y, device)
                stats.steps += 1
                stats.loss = loss
                if stats.steps % max(1, int(args.log_every)) == 0:
                    print(f"[point-aim] step={stats.steps} seen={stats.seen} loss={loss:.6f}")

            move_x, move_y, shoot_prob = infer_action(model, x, args.move_gain, args.max_step, args.invert_x, args.invert_y)
            if not args.train_only:
                if shoot_prob >= 0.5 and aim_error <= float(args.shoot_center_error):
                    controller.mouse_click(hold_sec=0.03)
                else:
                    if move_x != 0 or move_y != 0:
                        controller.mouse_move(move_x, move_y)

            if stats.steps % max(1, int(args.save_every)) == 0 and stats.steps > 0:
                save_model(model.cpu(), args.save_path)
                model.to(device)
                print(f"[point-aim] saved model to {args.save_path}")

            if controller.is_interrupt_x2_pressed():
                controller.stop()
                break

            time.sleep(max(0.01, float(args.poll_sec)))
    except KeyboardInterrupt:
        pass
    finally:
        try:
            save_model(model.cpu(), args.save_path)
            print(f"[point-aim] final model saved to {args.save_path}")
        except Exception:
            pass
        try:
            controller.stop()
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
