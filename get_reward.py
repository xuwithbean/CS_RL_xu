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
	center_lock = max(0.0, 1.0 - curr_aim)
	center_snap = max(0.0, 0.15 - curr_aim)

	ammo_cost = max(0.0, float(prev_obs.get("ammo", 0.0)) - float(curr_obs.get("ammo", 0.0)))
	shot_fired = float(curr_obs.get("shot_fired", ammo_cost))
	hit_event = 1.0 if hit > 0.5 else 0.0
	wasted_shot = max(0.0, shot_fired - hit_event)

	# 击杀耗时（秒）：越快越好。
	kill_time_sec = float(curr_obs.get("kill_time_sec", curr_obs.get("fight_time_sec", 0.0)))
	no_target_time_sec = float(curr_obs.get("no_target_time_sec", 0.0))
	kill_confirmed = bool(curr_obs.get("kill_confirmed", False)) or kill > 0.5
	# 4 秒内完成击杀可获得接近满额速度奖励，超过后奖励衰减到 0。
	kill_speed = max(0.0, 1.0 - kill_time_sec / 4.0)
	danger = float(curr_obs.get("danger_level", 0.0))

	reward_items: dict[str, float] = {
		"hit": 3.0 * hit,
		"kill": 10.0 * kill,
		"kill_speed": 3.0 * kill * kill_speed,
		"death": -6.0 * death,
		"aim": 0.8 * aim_improve,
		"center_lock": 0.75 * center_lock,
		"center_snap": 1.5 * center_snap,
		"waste_fire": -0.50 * wasted_shot,
		"ammo_cost": -0.15 * ammo_cost,
		"survive": 0.01,
		"time_penalty": -0.01 * max(0.0, float(curr_obs.get("fight_time_sec", 0.0))),
	}

	# 点瞄准任务里不要鼓励“让目标消失”，否则策略会学会低头/躲开。
	# 只有真实命中或击杀时才给一点确认奖励。
	if kill_confirmed and (hit_event > 0.5 or kill > 0.5):
		reward_items["kill_confirm_bonus"] = 0.5
	else:
		reward_items["kill_confirm_bonus"] = 0.0

	if no_target_time_sec > 0.0 and not kill_confirmed:
		reward_items["hide_penalty"] = -0.2 * min(1.0, no_target_time_sec / 1.5)
	else:
		reward_items["hide_penalty"] = 0.0

	# 子目标一致性奖励：鼓励短期动作配合长期决策。
	if str(manager_goal).startswith("fight"):
		reward_items["goal_align"] = 0.25 * hit + 0.15 * max(0.0, aim_improve) + 0.15 * center_lock
		if action_name == "shoot":
			reward_items["goal_align"] += 0.05
	elif manager_goal == "search":
		reward_items["goal_align"] = 0.1 if curr_obs.get("enemy_visible", False) else 0.02
	else:
		reward_items["goal_align"] = 0.0

	if kill_confirmed:
		reward_items["goal_align"] += 0.5

	total_reward = float(sum(reward_items.values()))
	return total_reward, reward_items