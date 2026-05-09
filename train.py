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
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from actions import m_actions
from find_enemy import get_enemy_feedback
from get_action import ACTIONS, get_action, get_action_command, get_q_table, get_q_update, get_state_key
from get_reward import get_reward


GOALS = ["search", "fight", "take_cover"]


@dataclass
class TrainConfig:
	episodes: int = 200
	max_steps: int = 200
	manager_interval: int = 10
	env_mode: str = "auto"
	shared_state_path: str = "/tmp/cs_rl_runtime_state.json"
	shared_frame_path: str = "/tmp/cs_rl_latest_frame.jpg"
	target_disappear_sec: float = 1.5
	step_dt_sec: float = 0.12
	apply_actions: bool = True
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
		self.step_dt_sec = 0.20
		self.reset()

	def reset(self) -> dict[str, Any]:
		self.hp = 100.0
		self.ammo = 30.0
		self.enemy_visible = self.rng.random() < 0.55
		self.aim_error = self.rng.uniform(0.35, 0.95)
		self.danger_level = self.rng.uniform(0.2, 0.7)
		self.fight_time_sec = 0.0
		return self._get_obs(hit=0.0, kill=0.0, death=0.0, shot_fired=0.0, kill_time_sec=0.0)

	def _get_obs(self, hit: float, kill: float, death: float, shot_fired: float, kill_time_sec: float) -> dict[str, Any]:
		enemy_distance = 0.3 + 0.7 * self.rng.random() if self.enemy_visible else 1.0
		return {
			"hp": max(0.0, min(100.0, self.hp)),
			"ammo": max(0.0, min(float(self.max_ammo), self.ammo)),
			"enemy_visible": bool(self.enemy_visible),
			"enemy_distance": max(0.0, min(1.0, enemy_distance)),
			"aim_error": max(0.0, min(1.0, self.aim_error)),
			"danger_level": max(0.0, min(1.0, self.danger_level)),
			"fight_time_sec": max(0.0, float(self.fight_time_sec)),
			"kill_time_sec": max(0.0, float(kill_time_sec)),
			"shot_fired": max(0.0, float(shot_fired)),
			"hit": hit,
			"kill": kill,
			"death": death,
		}

	def step(self, action_name: str, manager_goal: str) -> tuple[dict[str, Any], bool]:
		hit = 0.0
		kill = 0.0
		death = 0.0
		shot_fired = 0.0
		kill_time_sec = 0.0

		# 视野变化
		if manager_goal == "search":
			self.enemy_visible = self.rng.random() < 0.75
		else:
			self.enemy_visible = self.rng.random() < 0.55

		if self.enemy_visible:
			self.fight_time_sec += float(self.step_dt_sec)
		else:
			self.fight_time_sec = 0.0

		# 瞄准动作对误差的影响
		if action_name in {"aim_left", "aim_right", "aim_up", "aim_down"}:
			self.aim_error = max(0.0, self.aim_error - self.rng.uniform(0.02, 0.08))
		else:
			self.aim_error = min(1.0, self.aim_error + self.rng.uniform(-0.01, 0.02))

		# 射击逻辑
		if action_name == "shoot" and self.ammo > 0:
			self.ammo -= 1
			shot_fired = 1.0
			if self.enemy_visible:
				hit_prob = max(0.05, 0.85 - self.aim_error)
				if manager_goal == "fight":
					hit_prob += 0.08
				if self.rng.random() < hit_prob:
					hit = 1.0
					# 命中后有概率击杀
					if self.rng.random() < 0.20 + 0.30 * (1.0 - self.aim_error):
						kill = 1.0
						kill_time_sec = float(self.fight_time_sec)

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
			if kill_time_sec <= 0.0:
				kill_time_sec = float(self.fight_time_sec)

		return self._get_obs(hit=hit, kill=kill, death=death, shot_fired=shot_fired, kill_time_sec=kill_time_sec), done


class SharedPointEnv:
	"""基于 `trainimg.py` 共享点流的训练环境。

	该环境读取共享状态中的中心点坐标，把“点是否存在、离中心有多远、消失持续多久”
	转换成训练观测。当前默认把点消失持续 `target_disappear_sec` 秒视为击杀成功。
	"""

	def __init__(
		self,
		shared_state_path: str,
		shared_frame_path: str,
		target_disappear_sec: float = 1.5,
		step_dt_sec: float = 0.12,
	):
		self.shared_state_path = str(shared_state_path)
		self.shared_frame_path = str(shared_frame_path)
		self.target_disappear_sec = max(0.1, float(target_disappear_sec))
		self.step_dt_sec = max(0.01, float(step_dt_sec))
		self.max_ammo = 30
		self.rng = random.Random(7)
		self._visible_since: float | None = None
		self._lost_since: float | None = None
		self._last_target_signature = ""
		self._last_target_visible = False
		self._kill_confirmed = False
		self._ammo = float(self.max_ammo)
		self._shot_fired = 0.0
		self._death = 0.0
		self._hp = 100.0
		self._last_target_ref: tuple[int, int] | None = None

	@staticmethod
	def _read_payload(path: str) -> dict[str, Any]:
		if not path or not Path(path).exists():
			return {}
		try:
			with open(path, "r", encoding="utf-8") as f:
				payload = json.load(f)
			return payload if isinstance(payload, dict) else {}
		except Exception:
			return {}

	@staticmethod
	def _pick_target(centers: list[tuple[str, int, int, float]], ref_w: int, ref_h: int) -> tuple[str, int, int, float] | None:
		if not centers or ref_w <= 0 or ref_h <= 0:
			return None
		cx0 = ref_w / 2.0
		cy0 = ref_h / 2.0
		head_alias = {"head", "enemy_head", "person_head", "ct_head", "t_head"}
		body_alias = {"person", "enemy", "ct", "t", "body"}

		def _target_type(name: str) -> str:
			lname = str(name or "").strip().lower()
			if lname in head_alias or "head" in lname:
				return "head"
			if lname in body_alias:
				return "body"
			return "other"

		typed: list[tuple[str, int, int, float, str]] = []
		for name, x, y, conf in centers:
			typed.append((str(name), int(x), int(y), float(conf), _target_type(name)))

		def _pick(cands: list[tuple[str, int, int, float, str]]) -> tuple[str, int, int, float, str] | None:
			if not cands:
				return None
			return min(
				cands,
				key=lambda t: (
					((float(t[1]) - cx0) ** 2 + (float(t[2]) - cy0) ** 2) ** 0.5,
					-float(t[3]),
				),
			)

		chosen = _pick([c for c in typed if c[4] == "head"]) or _pick([c for c in typed if c[4] == "body"]) or _pick([c for c in typed if c[4] == "other"])
		if chosen is None:
			return None
		return chosen[:4]

	def _make_obs(
		self,
		visible: bool,
		center_error: float,
		fight_time_sec: float,
		kill_time_sec: float,
		no_target_time_sec: float,
		hit: float,
		kill: float,
		death: float,
		shot_fired: float,
	) -> dict[str, Any]:
		enemy_distance = max(0.0, min(1.0, center_error))
		return {
			"hp": max(0.0, min(100.0, float(self._hp))),
			"ammo": max(0.0, min(float(self.max_ammo), float(self._ammo))),
			"enemy_visible": bool(visible),
			"enemy_distance": enemy_distance,
			"aim_error": max(0.0, min(1.0, float(center_error))),
			"danger_level": max(0.0, min(1.0, center_error)),
			"fight_time_sec": max(0.0, float(fight_time_sec)),
			"kill_time_sec": max(0.0, float(kill_time_sec)),
			"no_target_time_sec": max(0.0, float(no_target_time_sec)),
			"shot_fired": max(0.0, float(shot_fired)),
			"hit": float(hit),
			"kill": float(kill),
			"death": float(death),
			"kill_confirmed": bool(self._kill_confirmed),
		}

	def reset(self) -> dict[str, Any]:
		self._visible_since = None
		self._lost_since = None
		self._last_target_signature = ""
		self._last_target_visible = False
		self._kill_confirmed = False
		self._ammo = float(self.max_ammo)
		self._shot_fired = 0.0
		self._death = 0.0
		self._hp = 100.0
		return self._observe(force_refresh=True)

	def _observe(self, force_refresh: bool = False) -> dict[str, Any]:
		payload = self._read_payload(self.shared_state_path)
		centers = []
		for item in list((payload or {}).get("centers") or []):
			if not isinstance(item, dict):
				continue
			try:
				centers.append((str(item.get("name", "")), int(item.get("cx", 0)), int(item.get("cy", 0)), float(item.get("conf", 0.0))))
			except Exception:
				continue

		ref_w = int((payload or {}).get("centers_ref_w") or 0)
		ref_h = int((payload or {}).get("centers_ref_h") or 0)
		if ref_w <= 0 or ref_h <= 0:
			ref_w = 1280
			ref_h = 720

		now = time.monotonic()
		target = self._pick_target(centers, ref_w, ref_h)
		visible = target is not None
		center_error = 1.0
		if target is not None:
			_, tx, ty, conf = target
			cx0 = ref_w / 2.0
			cy0 = ref_h / 2.0
			dist = ((float(tx) - cx0) ** 2 + (float(ty) - cy0) ** 2) ** 0.5
			center_error = max(0.0, min(1.0, dist / max(1.0, ((ref_w / 2.0) ** 2 + (ref_h / 2.0) ** 2) ** 0.5)))
			signature = f"{target[0]}:{target[1]}:{target[2]}:{int(conf * 1000)}"
			self._last_target_signature = signature
			self._last_target_ref = (ref_w, ref_h)
		else:
			signature = ""

		if visible:
			if not self._last_target_visible:
				self._visible_since = now
			self._lost_since = None
			self._last_target_visible = True
			fight_time_sec = 0.0 if self._visible_since is None else now - self._visible_since
			no_target_time_sec = 0.0
			self._kill_confirmed = False
			return self._make_obs(
				visible=True,
				center_error=center_error,
				fight_time_sec=fight_time_sec,
				kill_time_sec=0.0,
				no_target_time_sec=no_target_time_sec,
				hit=0.0,
				kill=0.0,
				death=0.0,
				shot_fired=self._shot_fired,
			)

		# 目标不可见：记录消失持续时间
		if self._last_target_visible:
			self._lost_since = now if self._lost_since is None else self._lost_since
			self._last_target_visible = False
		elif self._lost_since is None:
			self._lost_since = now

		no_target_time_sec = 0.0 if self._lost_since is None else now - self._lost_since
		kill = 0.0
		kill_time_sec = 0.0
		fight_time_sec = 0.0 if self._visible_since is None else now - self._visible_since
		if self._visible_since is not None and no_target_time_sec >= self.target_disappear_sec:
			self._kill_confirmed = True
			kill = 1.0
			kill_time_sec = fight_time_sec

		return self._make_obs(
			visible=False,
			center_error=center_error,
			fight_time_sec=fight_time_sec,
			kill_time_sec=kill_time_sec,
			no_target_time_sec=no_target_time_sec,
			hit=0.0,
			kill=kill,
			death=0.0,
			shot_fired=self._shot_fired,
		)

	def step(self, action_name: str, manager_goal: str) -> tuple[dict[str, Any], bool]:
		self._shot_fired = 1.0 if action_name == "shoot" and self._ammo > 0 else 0.0
		if action_name == "shoot" and self._ammo > 0:
			self._ammo -= 1
		elif action_name == "reload":
			self._ammo = float(self.max_ammo)

		# 给外部控制与点流更新留出时间。
		time.sleep(self.step_dt_sec)
		obs = self._observe()
		done = bool(obs.get("kill", 0.0) > 0.5 or obs.get("death", 0.0) > 0.5)
		return obs, done


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
	parser.add_argument("--env-mode", type=str, default="auto", choices=["auto", "smoke", "shared"], help="训练环境模式：smoke 为模拟，shared 为读取 trainimg 点流")
	parser.add_argument("--shared-state-path", type=str, default="/tmp/cs_rl_runtime_state.json")
	parser.add_argument("--shared-frame-path", type=str, default="/tmp/cs_rl_latest_frame.jpg")
	parser.add_argument("--target-disappear-sec", type=float, default=1.5, help="目标消失达到该时长视为击杀成功")
	parser.add_argument("--step-dt-sec", type=float, default=0.12, help="共享流模式下每步等待时长")
	parser.add_argument("--apply-actions", action="store_true", help="在 shared 模式下实际执行鼠标/键盘动作")
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
		env_mode=args.env_mode,
		shared_state_path=args.shared_state_path,
		shared_frame_path=args.shared_frame_path,
		target_disappear_sec=args.target_disappear_sec,
		step_dt_sec=args.step_dt_sec,
		apply_actions=bool(args.apply_actions),
		alpha=args.alpha,
		gamma=args.gamma,
		epsilon_start=args.epsilon_start,
		epsilon_end=args.epsilon_end,
		epsilon_decay=args.epsilon_decay,
		seed=args.seed,
		save_path=args.save_path,
	)


def _make_env(cfg: TrainConfig) -> Any:
	if cfg.env_mode == "smoke":
		return SimpleCombatEnv(seed=cfg.seed)
	if cfg.env_mode == "shared":
		return SharedPointEnv(
			shared_state_path=cfg.shared_state_path,
			shared_frame_path=cfg.shared_frame_path,
			target_disappear_sec=cfg.target_disappear_sec,
			step_dt_sec=cfg.step_dt_sec,
		)
	if Path(cfg.shared_state_path).exists() and Path(cfg.shared_frame_path).exists():
		return SharedPointEnv(
			shared_state_path=cfg.shared_state_path,
			shared_frame_path=cfg.shared_frame_path,
			target_disappear_sec=cfg.target_disappear_sec,
			step_dt_sec=cfg.step_dt_sec,
		)
	return SimpleCombatEnv(seed=cfg.seed)


def _execute_action(action_name: str, controller: m_actions) -> None:
	cmd = get_action_command(action_name)
	dx, dy = cmd.get("mouse", (0, 0))
	for key in list(cmd.get("keys", [])):
		if key == "w":
			controller.move_forward(0.02)
		elif key == "s":
			controller.move_back(0.02)
		elif key == "a":
			controller.move_left(0.02)
		elif key == "d":
			controller.move_right(0.02)
		elif key == "r":
			controller.reload(0.02)
	if int(dx) != 0 or int(dy) != 0:
		controller.mouse_move(int(dx), int(dy))
	if bool(cmd.get("shoot", False)):
		controller.mouse_click(hold_sec=0.03)


def train_loop(cfg: TrainConfig) -> None:
	random.seed(cfg.seed)
	env = _make_env(cfg)
	q_table = get_q_table()
	epsilon = cfg.epsilon_start
	controller = m_actions() if (cfg.apply_actions or cfg.env_mode != "smoke") else None
	if cfg.env_mode != "smoke":
		print(
			f"[train] env_mode={cfg.env_mode} apply_actions={bool(controller is not None)} "
			f"target_disappear_sec={cfg.target_disappear_sec:.2f}",
			flush=True,
		)

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
			if controller is not None:
				_execute_action(action_name, controller)
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

		if controller is not None:
			controller.stop()

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