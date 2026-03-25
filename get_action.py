# [x]: 当前应当完成的动作。
"""短期决策动作模块（Q-learning 版本）。

说明：
- 提供离散动作空间与状态离散化。
- 提供 epsilon-greedy 选动作与 Q 表更新函数。
- 提供动作到控制指令的映射，方便未来接入键鼠控制模块。
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any


# 动作空间：可按需要扩展。
ACTIONS: list[str] = [
	"idle",
	"move_forward",
	"move_back",
	"strafe_left",
	"strafe_right",
	"aim_left",
	"aim_right",
	"aim_up",
	"aim_down",
	"shoot",
	"reload",
]


def get_q_table() -> defaultdict[tuple[Any, ...], list[float]]:
	"""返回默认 Q 表结构。"""
	return defaultdict(lambda: [0.0 for _ in ACTIONS])


def _get_bin(value: float, thresholds: list[float]) -> int:
	"""按阈值离散化为桶编号。"""
	for idx, th in enumerate(thresholds):
		if value < th:
			return idx
	return len(thresholds)


def get_state_key(obs: dict[str, Any], manager_goal: str) -> tuple[Any, ...]:
	"""将连续观测离散化为 Q 表键。"""
	enemy_visible = int(bool(obs.get("enemy_visible", False)))
	aim_bin = _get_bin(float(obs.get("aim_error", 1.0)), [0.2, 0.45, 0.7])
	hp_bin = _get_bin(float(obs.get("hp", 100.0)) / 100.0, [0.2, 0.45, 0.7])
	ammo_bin = _get_bin(float(obs.get("ammo", 30.0)) / 30.0, [0.15, 0.35, 0.7])
	danger_bin = _get_bin(float(obs.get("danger_level", 0.0)), [0.25, 0.5, 0.75])

	return (manager_goal, enemy_visible, aim_bin, hp_bin, ammo_bin, danger_bin)


def get_action(
	q_table: defaultdict[tuple[Any, ...], list[float]],
	state_key: tuple[Any, ...],
	epsilon: float,
) -> tuple[int, str]:
	"""使用 epsilon-greedy 选择动作。"""
	if random.random() < float(epsilon):
		action_idx = random.randrange(len(ACTIONS))
	else:
		q_values = q_table[state_key]
		max_q = max(q_values)
		best_idxs = [i for i, q in enumerate(q_values) if q == max_q]
		action_idx = random.choice(best_idxs)
	return action_idx, ACTIONS[action_idx]


def get_q_update(
	q_table: defaultdict[tuple[Any, ...], list[float]],
	state_key: tuple[Any, ...],
	action_idx: int,
	reward: float,
	next_state_key: tuple[Any, ...],
	alpha: float,
	gamma: float,
) -> float:
	"""执行一次 Q-learning 更新并返回更新后的 Q 值。"""
	old_q = q_table[state_key][action_idx]
	next_max_q = max(q_table[next_state_key])
	target = reward + float(gamma) * next_max_q
	new_q = old_q + float(alpha) * (target - old_q)
	q_table[state_key][action_idx] = new_q
	return new_q


def get_action_command(action_name: str) -> dict[str, Any]:
	"""将动作名映射为控制层命令（先给出统一格式，后续接真实控制）。"""
	mapping: dict[str, dict[str, Any]] = {
		"idle": {"keys": [], "mouse": (0, 0), "shoot": False},
		"move_forward": {"keys": ["w"], "mouse": (0, 0), "shoot": False},
		"move_back": {"keys": ["s"], "mouse": (0, 0), "shoot": False},
		"strafe_left": {"keys": ["a"], "mouse": (0, 0), "shoot": False},
		"strafe_right": {"keys": ["d"], "mouse": (0, 0), "shoot": False},
		"aim_left": {"keys": [], "mouse": (-8, 0), "shoot": False},
		"aim_right": {"keys": [], "mouse": (8, 0), "shoot": False},
		"aim_up": {"keys": [], "mouse": (0, -6), "shoot": False},
		"aim_down": {"keys": [], "mouse": (0, 6), "shoot": False},
		"shoot": {"keys": [], "mouse": (0, 0), "shoot": True},
		"reload": {"keys": ["r"], "mouse": (0, 0), "shoot": False},
	}
	return mapping.get(action_name, mapping["idle"]) 