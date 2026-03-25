# [x]:强化学习训练得到分数的计算方式。
"""强化学习奖励函数（短期决策层）。

奖励设计原则：
- 终局事件（击杀/死亡）权重更高。
- 提供密集奖励（瞄准误差变小、减少无效射击）提升学习稳定性。
- 支持按组件返回，便于调参和可解释性分析。
"""

from __future__ import annotations

from typing import Any


def get_reward(
	prev_obs: dict[str, Any],
	curr_obs: dict[str, Any],
	action_name: str,
	manager_goal: str,
) -> tuple[float, dict[str, float]]:
	"""计算一步奖励。

	参数：
	- prev_obs: 上一步观测
	- curr_obs: 当前观测
	- action_name: 当前动作名
	- manager_goal: 长期策略层给出的子目标

	返回：
	- total_reward: 总奖励
	- reward_items: 组件奖励，便于日志分析
	"""
	hit = float(curr_obs.get("hit", 0.0))
	kill = float(curr_obs.get("kill", 0.0))
	death = float(curr_obs.get("death", 0.0))

	prev_aim = float(prev_obs.get("aim_error", 1.0))
	curr_aim = float(curr_obs.get("aim_error", 1.0))
	aim_improve = prev_aim - curr_aim

	ammo_cost = max(0.0, float(prev_obs.get("ammo", 0.0)) - float(curr_obs.get("ammo", 0.0)))
	danger = float(curr_obs.get("danger_level", 0.0))

	reward_items: dict[str, float] = {
		"hit": 3.0 * hit,
		"kill": 8.0 * kill,
		"death": -6.0 * death,
		"aim": 0.8 * aim_improve,
		"waste_fire": -0.2 * ammo_cost,
		"survive": 0.02,
	}

	# 子目标一致性奖励：鼓励短期动作配合长期决策。
	if manager_goal == "take_cover":
		reward_items["goal_align"] = 0.4 * max(0.0, danger - float(curr_obs.get("danger_level", danger)))
		if action_name in {"move_back", "strafe_left", "strafe_right"}:
			reward_items["goal_align"] += 0.15
	elif manager_goal == "fight":
		reward_items["goal_align"] = 0.2 * hit + 0.1 * max(0.0, aim_improve)
		if action_name == "shoot":
			reward_items["goal_align"] += 0.05
	elif manager_goal == "search":
		reward_items["goal_align"] = 0.1 if curr_obs.get("enemy_visible", False) else 0.02
	else:
		reward_items["goal_align"] = 0.0

	total_reward = float(sum(reward_items.values()))
	return total_reward, reward_items