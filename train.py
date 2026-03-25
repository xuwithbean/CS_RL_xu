# [x]: 强化学习的训练代码。
"""分层决策训练脚本（长期策略 + 短期强化学习）。

当前版本目标：
- 先跑通最小闭环，验证算法流程与接口。
- 长期决策层（Manager）使用轻量规则策略，按低频更新子目标。
- 短期决策层（Worker）使用 Q-learning 学习具体动作。

后续接入真实游戏时，只需替换：
1) 环境观测（截图 + 敌人识别）
2) 动作执行（键鼠控制）
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from find_enemy import get_enemy_feedback
from get_action import ACTIONS, get_action, get_q_table, get_q_update, get_state_key
from get_reward import get_reward


GOALS = ["search", "fight", "take_cover"]


@dataclass
class TrainConfig:
	episodes: int = 200
	max_steps: int = 200
	manager_interval: int = 10
	alpha: float = 0.15
	gamma: float = 0.95
	epsilon_start: float = 0.40
	epsilon_end: float = 0.05
	epsilon_decay: float = 0.995
	seed: int = 7
	save_path: str = "q_table.json"


class SimpleCombatEnv:
	"""用于算法联调的简化战斗环境。

	该环境并不追求游戏真实性，而是用于验证分层 RL 代码是否正确连通。
	"""

	def __init__(self, seed: int = 7):
		self.rng = random.Random(seed)
		self.max_ammo = 30
		self.reset()

	def reset(self) -> dict[str, Any]:
		self.hp = 100.0
		self.ammo = 30.0
		self.enemy_visible = self.rng.random() < 0.55
		self.aim_error = self.rng.uniform(0.35, 0.95)
		self.danger_level = self.rng.uniform(0.2, 0.7)
		return self._get_obs(hit=0.0, kill=0.0, death=0.0)

	def _get_obs(self, hit: float, kill: float, death: float) -> dict[str, Any]:
		enemy_distance = 0.3 + 0.7 * self.rng.random() if self.enemy_visible else 1.0
		return {
			"hp": max(0.0, min(100.0, self.hp)),
			"ammo": max(0.0, min(float(self.max_ammo), self.ammo)),
			"enemy_visible": bool(self.enemy_visible),
			"enemy_distance": max(0.0, min(1.0, enemy_distance)),
			"aim_error": max(0.0, min(1.0, self.aim_error)),
			"danger_level": max(0.0, min(1.0, self.danger_level)),
			"hit": hit,
			"kill": kill,
			"death": death,
		}

	def step(self, action_name: str, manager_goal: str) -> tuple[dict[str, Any], bool]:
		hit = 0.0
		kill = 0.0
		death = 0.0

		# 视野变化
		if manager_goal == "search":
			self.enemy_visible = self.rng.random() < 0.75
		else:
			self.enemy_visible = self.rng.random() < 0.55

		# 瞄准动作对误差的影响
		if action_name in {"aim_left", "aim_right", "aim_up", "aim_down"}:
			self.aim_error = max(0.0, self.aim_error - self.rng.uniform(0.02, 0.08))
		else:
			self.aim_error = min(1.0, self.aim_error + self.rng.uniform(-0.01, 0.02))

		# 射击逻辑
		if action_name == "shoot" and self.ammo > 0:
			self.ammo -= 1
			if self.enemy_visible:
				hit_prob = max(0.05, 0.85 - self.aim_error)
				if manager_goal == "fight":
					hit_prob += 0.08
				if self.rng.random() < hit_prob:
					hit = 1.0
					# 命中后有概率击杀
					if self.rng.random() < 0.20 + 0.30 * (1.0 - self.aim_error):
						kill = 1.0

		if action_name == "reload":
			self.ammo = float(self.max_ammo)

		# 危险变化与受伤逻辑
		if manager_goal == "take_cover":
			self.danger_level = max(0.0, self.danger_level - self.rng.uniform(0.04, 0.12))
		else:
			self.danger_level = min(1.0, self.danger_level + self.rng.uniform(-0.02, 0.06))

		if self.enemy_visible:
			hurt_prob = 0.05 + 0.35 * self.danger_level
			if action_name in {"move_back", "strafe_left", "strafe_right"}:
				hurt_prob *= 0.8
			if manager_goal == "take_cover":
				hurt_prob *= 0.75
			if self.rng.random() < hurt_prob:
				self.hp -= self.rng.uniform(4.0, 11.0)

		done = False
		if self.hp <= 0:
			death = 1.0
			done = True
		elif kill > 0.5:
			done = True

		return self._get_obs(hit=hit, kill=kill, death=death), done


def get_manager_goal(obs: dict[str, Any], step_idx: int, manager_interval: int) -> str:
	"""长期决策层（低频）子目标选择。

	这里先用规则版，后续可切换为 LLM / 学习型策略网络。
	"""
	if step_idx % manager_interval != 0:
		return ""

	hp = float(obs.get("hp", 100.0))
	danger = float(obs.get("danger_level", 0.0))
	enemy_visible = bool(obs.get("enemy_visible", False))

	if hp < 35 or danger > 0.72:
		return "take_cover"
	if enemy_visible:
		return "fight"
	return "search"


def _get_serialize_q_table(q_table: dict[tuple[Any, ...], list[float]]) -> dict[str, list[float]]:
	return {"|".join(map(str, key)): values for key, values in q_table.items()}


def get_train_config_from_args() -> TrainConfig:
	parser = argparse.ArgumentParser(description="Hierarchical RL trainer")
	parser.add_argument("--episodes", type=int, default=200)
	parser.add_argument("--max-steps", type=int, default=200)
	parser.add_argument("--manager-interval", type=int, default=10)
	parser.add_argument("--alpha", type=float, default=0.15)
	parser.add_argument("--gamma", type=float, default=0.95)
	parser.add_argument("--epsilon-start", type=float, default=0.40)
	parser.add_argument("--epsilon-end", type=float, default=0.05)
	parser.add_argument("--epsilon-decay", type=float, default=0.995)
	parser.add_argument("--seed", type=int, default=7)
	parser.add_argument("--save-path", type=str, default="q_table.json")
	args = parser.parse_args()
	return TrainConfig(
		episodes=args.episodes,
		max_steps=args.max_steps,
		manager_interval=args.manager_interval,
		alpha=args.alpha,
		gamma=args.gamma,
		epsilon_start=args.epsilon_start,
		epsilon_end=args.epsilon_end,
		epsilon_decay=args.epsilon_decay,
		seed=args.seed,
		save_path=args.save_path,
	)


def train_loop(cfg: TrainConfig) -> None:
	random.seed(cfg.seed)
	env = SimpleCombatEnv(seed=cfg.seed)
	q_table = get_q_table()
	epsilon = cfg.epsilon_start

	for ep in range(1, cfg.episodes + 1):
		obs = env.reset()
		goal = "search"
		ep_reward = 0.0
		ep_hit = 0.0
		ep_kill = 0.0
		ep_death = 0.0

		for step_idx in range(cfg.max_steps):
			maybe_goal = get_manager_goal(obs, step_idx, cfg.manager_interval)
			if maybe_goal:
				goal = maybe_goal

			enemy_feedback = get_enemy_feedback(obs)
			worker_obs = dict(obs)
			worker_obs.update(enemy_feedback)

			state_key = get_state_key(worker_obs, goal)
			action_idx, action_name = get_action(q_table, state_key, epsilon)
			next_obs, done = env.step(action_name, goal)

			reward, _reward_items = get_reward(obs, next_obs, action_name, goal)
			ep_reward += reward
			ep_hit += float(next_obs.get("hit", 0.0))
			ep_kill += float(next_obs.get("kill", 0.0))
			ep_death += float(next_obs.get("death", 0.0))

			next_enemy_feedback = get_enemy_feedback(next_obs)
			next_worker_obs = dict(next_obs)
			next_worker_obs.update(next_enemy_feedback)
			next_state_key = get_state_key(next_worker_obs, goal)

			get_q_update(
				q_table=q_table,
				state_key=state_key,
				action_idx=action_idx,
				reward=reward,
				next_state_key=next_state_key,
				alpha=cfg.alpha,
				gamma=cfg.gamma,
			)

			obs = next_obs
			if done:
				break

		epsilon = max(cfg.epsilon_end, epsilon * cfg.epsilon_decay)

		if ep == 1 or ep % 10 == 0:
			print(
				f"[Episode {ep:03d}] reward={ep_reward:.2f} hit={ep_hit:.0f} "
				f"kill={ep_kill:.0f} death={ep_death:.0f} eps={epsilon:.3f}"
			)

	save_path = Path(cfg.save_path)
	save_path.write_text(
		json.dumps(
			{
				"meta": {
					"episodes": cfg.episodes,
					"max_steps": cfg.max_steps,
					"actions": ACTIONS,
				},
				"q_table": _get_serialize_q_table(q_table),
			},
			ensure_ascii=False,
			indent=2,
		),
		encoding="utf-8",
	)
	print(f"训练完成，Q 表已保存到: {save_path}")


if __name__ == "__main__":
	config = get_train_config_from_args()
	train_loop(config)