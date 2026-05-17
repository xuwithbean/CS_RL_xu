# [x]: 强化学习的训练代码。
from __future__ import annotations

import argparse
import importlib
import json
import random
import signal
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from actions import m_actions
from decision_advisor import get_query_kill_count_from_frame
from find_enemy import get_enemy_feedback
from get_action import get_action_command
from get_reward import get_reward
from td3_agent import ReplayBuffer, TD3Agent
from visual_recognition.stream_ffplay_pipeline import get_qwen_location_client, get_resolve_qwen_api_key


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
	step_dt_sec: float = 0.03
	apply_actions: bool = True
	target_family: str = "MIXED"
	resume: bool = True
	load_path: str = ""
	gamma: float = 0.99
	seed: int = 7
	save_path: str = "td3_checkpoint.pt"
	best_save_path: str = ""
	reward_plot_path: str = "reward_curve.png"
	best_reward_plot_path: str = ""
	reward_plot_every: int = 100
	reward_kpm_weight: float = 0.05
	move_gain: float = 400.0
	max_step: int = 400
	batch_size: int = 128
	replay_size: int = 50000
	start_steps: int = 400
	updates_per_step: int = 1
	policy_noise: float = 0.20
	noise_clip: float = 0.50
	policy_delay: int = 2
	tau: float = 0.005
	exploration_noise: float = 0.15
	shoot_threshold: float = 0.12
	shoot_center_error: float = 0.04
	use_proportional_control: bool = False
	invert_x: bool = False
	invert_y: bool = False
	checkpoint_every: int = 10
	qwen_api_key: str = ""
	no_target_search_step: int = 16
	no_target_search_interval_sec: float = 1.0
	stream_delay_sec: float = 1.0
	auto_measure_stream_delay: bool = True
	delay_measure_trials: int = 3
	delay_measure_move_px: int = 220
	delay_measure_min_shift: float = 0.06
	delay_measure_timeout_sec: float = 3.0
	delay_measure_poll_sec: float = 0.03


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
			"hp": max(0.0, min( 100.0, self.hp)),
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

	该环境读取共享状态中的中心点坐标，把“点是否存在、离中心有多远、消失持续多久"
	转换成训练观测。点消失仅表示丢失目标，不直接记为击杀。
	"""

	def __init__(
		self,
		shared_state_path: str,
		shared_frame_path: str,
		target_disappear_sec: float = 1.5,
		step_dt_sec: float = 0.03,
		stream_delay_sec: float = 1.0,
		target_family: str = "MIXED",
	):
		self.shared_state_path = str(shared_state_path)
		self.shared_frame_path = str(shared_frame_path)
		self.target_disappear_sec = max(0.1, float(target_disappear_sec))
		self.step_dt_sec = max(0.01, float(step_dt_sec))
		self.stream_delay_sec = max(0.0, float(stream_delay_sec))
		self.target_family = str(target_family or "MIXED").upper()
		self.max_ammo = 30
		self.rng = random.Random(7)
		self._visible_since: float | None = None
		self._lost_since: float | None = None
		self._last_target_signature = ""
		self._last_target_visible = False
		self._tracked_target_name: str | None = None
		self._tracked_target_ref: tuple[int, int] | None = None
		self._kill_confirmed = False
		self._ammo = float(self.max_ammo)
		self._shot_fired = 0.0
		self._death = 0.0
		self._hp = 100.0
		self._last_target_ref: tuple[int, int] | None = None
		self._last_target_name: str | None = None
		self._last_target_dx: float = 0.0
		self._last_target_dy: float = 0.0

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

	def _select_target(
		self,
		centers: list[tuple[str, int, int, float]],
		ref_w: int,
		ref_h: int,
	) -> tuple[str, int, int, float] | None:
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

		if self._tracked_target_name:
			locked = [c for c in typed if c[0] == self._tracked_target_name]
			if locked:
				ref_pos = self._tracked_target_ref
				if ref_pos is None:
					ref_pos = (int(cx0), int(cy0))
				return min(
					locked,
					key=lambda t: (
						((float(t[1]) - float(ref_pos[0])) ** 2 + (float(t[2]) - float(ref_pos[1])) ** 2) ** 0.5,
						-float(t[3]),
					),
				)[:4]

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
		self._tracked_target_name = str(chosen[0])
		self._tracked_target_ref = (int(chosen[1]), int(chosen[2]))
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
			"target_visible": bool(visible),
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
			"target_name": str(self._last_target_name or ""),
			"target_dx": float(self._last_target_dx),
			"target_dy": float(self._last_target_dy),
			"target_x": int(self._last_target_ref[0] / 2 + self._last_target_dx * self._last_target_ref[0] / 2) if (visible and self._last_target_ref) else -1,
			"target_y": int(self._last_target_ref[1] / 2 + self._last_target_dy * self._last_target_ref[1] / 2) if (visible and self._last_target_ref) else -1,
		}

	def reset(self) -> dict[str, Any]:
		self._visible_since = None
		self._lost_since = None
		self._last_target_signature = ""
		self._last_target_visible = False
		self._tracked_target_name = None
		self._tracked_target_ref = None
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
		target = self._select_target(centers, ref_w, ref_h)
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
			self._last_target_name = str(target[0])
			self._last_target_dx = float((float(tx) - cx0) / max(1.0, cx0))
			self._last_target_dy = float((float(ty) - cy0) / max(1.0, cy0))
		else:
			signature = ""
			self._last_target_name = None
			self._last_target_dx = 0.0
			self._last_target_dy = 0.0
			self._tracked_target_name = None
			self._tracked_target_ref = None

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
		fight_time_sec = 0.0 if self._visible_since is None else now - self._visible_since
		# 点流模式下，目标消失只表示视野丢失，不再当作击杀成功。
		self._kill_confirmed = False

		return self._make_obs(
			visible=False,
			center_error=center_error,
			fight_time_sec=fight_time_sec,
			kill_time_sec=0.0,
			no_target_time_sec=no_target_time_sec,
			hit=0.0,
			kill=0.0,
			death=0.0,
			shot_fired=self._shot_fired,
		)

	def step(self, action_name: str, manager_goal: str) -> tuple[dict[str, Any], bool]:
		self._shot_fired = 1.0 if action_name == "shoot" and self._ammo > 0 else 0.0
		if action_name == "shoot" and self._ammo > 0:
			self._ammo -= 1
		elif action_name == "reload":
			self._ammo = float(self.max_ammo)

		# 按基础步频采样；动作-反馈延迟由训练循环中的延迟对齐队列处理。
		time.sleep(self.step_dt_sec)
		obs = self._observe()
		done = bool(obs.get("kill", 0.0) > 0.5 or obs.get("death", 0.0) > 0.5)
		return obs, done

	def get_observation(self) -> dict[str, Any]:
		"""无动作采样当前共享观测，用于延迟测量与调试。"""
		return self._observe(force_refresh=True)


def get_manager_goal(obs: dict[str, Any], step_idx: int, manager_interval: int) -> str:
	"""长期决策层（低频）子目标选择。

	这里先用规则版，后续可切换为 LLM / 学习型策略网络。
	"""
	if step_idx % manager_interval != 0:
		return ""

	hp = float(obs.get("hp", 100.0))
	danger = float(obs.get("danger_level", 0.0))
	enemy_visible = bool(obs.get("enemy_visible", False))

	if enemy_visible:
		return "fight"
	return "search"


def _get_serialize_q_table(q_table: dict[tuple[Any, ...], list[float]]) -> dict[str, list[float]]:
	return {"|".join(map(str, key)): values for key, values in q_table.items()}


def _load_q_table_from_path(load_path: str) -> dict[tuple[Any, ...], list[float]]:
	path = Path(str(load_path or "").strip())
	if not path.exists():
		return get_q_table()
	try:
		payload = json.loads(path.read_text(encoding="utf-8"))
	except Exception:
		return get_q_table()
	loaded = get_q_table()
	raw_table = (payload or {}).get("q_table") if isinstance(payload, dict) else None
	if not isinstance(raw_table, dict):
		return loaded
	for key_text, values in raw_table.items():
		if not isinstance(key_text, str) or not isinstance(values, list):
			continue
		key = tuple(key_text.split("|"))
		loaded[key] = [float(v) for v in values]
	return loaded


def _save_q_table_to_path(save_path: str, q_table: dict[tuple[Any, ...], list[float]], cfg: TrainConfig, episode_idx: int, epsilon: float) -> None:
	path = Path(str(save_path or "").strip())
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(
		json.dumps(
			{
				"meta": {
					"episodes": cfg.episodes,
					"max_steps": cfg.max_steps,
					"actions": ACTIONS,
					"episode_idx": int(episode_idx),
					"epsilon": float(epsilon),
					"target_family": str(cfg.target_family),
				},
				"q_table": _get_serialize_q_table(q_table),
			},
			ensure_ascii=False,
			indent=2,
		),
		encoding="utf-8",
	)


def _get_default_best_model_path(save_path: str) -> str:
	path = Path(str(save_path or "q_table.json"))
	stem = path.stem or "q_table"
	return str(path.with_name(f"{stem}_best{path.suffix or '.json'}"))


def _get_default_best_plot_path(plot_path: str) -> str:
	path = Path(str(plot_path or "reward_curve.png"))
	stem = path.stem or "reward_curve"
	return str(path.with_name(f"{stem}_best{path.suffix or '.png'}"))


def _save_reward_plot(
	reward_history: list[float],
	kpm_history: list[float],
	plot_path: str,
	best_episode_idx: int = 0,
	best_reward: float = 0.0,
	best_kpm: float = 0.0,
) -> bool:
	if not reward_history:
		return False
	try:
		import matplotlib
		matplotlib.use("Agg")
		import matplotlib.pyplot as plt
	except Exception as exc:
		print(f"[train] 无法保存 reward/KPM 图（缺少 matplotlib）: {exc}", flush=True)
		return False

	path = Path(str(plot_path or "reward_curve.png"))
	path.parent.mkdir(parents=True, exist_ok=True)

	xs = list(range(1, len(reward_history) + 1))
	fig = plt.figure(figsize=(10, 4.5), dpi=120)
	ax = fig.add_subplot(111)
	ax.plot(xs, reward_history, color="#2b6cb0", linewidth=1.6, label="episode_reward")
	ax.set_title("Training Reward Curve")
	ax.set_xlabel("Episode")
	ax.set_ylabel("Reward")
	ax.grid(True, alpha=0.25)
	ax2 = ax.twinx()
	ax2.plot(xs, kpm_history, color="#c05621", linewidth=1.2, alpha=0.9, label="episode_kpm")
	ax2.set_ylabel("KPM")

	if best_episode_idx > 0:
		ax.scatter([best_episode_idx], [best_reward], color="#1f4e79", s=28, zorder=3, label="best_reward")
		ax2.scatter([best_episode_idx], [best_kpm], color="#9c4221", s=28, zorder=3, label="best_kpm")
		ax.axvline(best_episode_idx, color="#c53030", alpha=0.20, linewidth=1.0)

	handles1, labels1 = ax.get_legend_handles_labels()
	handles2, labels2 = ax2.get_legend_handles_labels()
	ax.legend(handles1 + handles2, labels1 + labels2, loc="best")
	fig.tight_layout()
	fig.savefig(path, dpi=120)
	plt.close(fig)
	return True


def get_train_config_from_args() -> TrainConfig:
	parser = argparse.ArgumentParser(description="Hierarchical RL trainer")
	parser.add_argument("--episodes", type=int, default=200)
	parser.add_argument("--max-steps", type=int, default=200)
	parser.add_argument("--manager-interval", type=int, default=10)
	parser.add_argument("--env-mode", type=str, default="auto", choices=["auto", "smoke", "shared"], help="训练环境模式：smoke 为模拟，shared 为读取 trainimg 点流")
	parser.add_argument("--shared-state-path", type=str, default="/tmp/cs_rl_runtime_state.json")
	parser.add_argument("--shared-frame-path", type=str, default="/tmp/cs_rl_latest_frame.jpg")
	parser.add_argument("--target-disappear-sec", type=float, default=1.5, help="目标消失达到该时长视为击杀成功")
	parser.add_argument("--step-dt-sec", type=float, default=0.03, help="共享流模式下每步等待时长")
	parser.add_argument("--apply-actions", action="store_true", help="在 shared 模式下实际执行鼠标/键盘动作")
	parser.add_argument("--target-family", type=str, default="MIXED", choices=["MIXED", "CT", "T"], help="训练/应用时的目标家族；MIXED 表示 CT/T 都学习")
	parser.add_argument("--resume", action="store_true", help="从已有模型继续训练")
	parser.add_argument("--load-path", type=str, default="", help="加载已有模型的路径；为空则使用 --save-path")
	parser.add_argument("--gamma", type=float, default=0.99)
	parser.add_argument("--seed", type=int, default=7)
	parser.add_argument("--save-path", type=str, default="td3_checkpoint.pt")
	parser.add_argument("--best-save-path", type=str, default="", help="最佳 TD3 模型保存路径；为空则自动生成 *_best.pt")
	parser.add_argument("--reward-plot-path", type=str, default="reward_curve.png", help="reward 曲线图输出路径")
	parser.add_argument("--best-reward-plot-path", type=str, default="", help="最佳模型对应 reward 图路径；为空则自动生成 *_best.png")
	parser.add_argument("--reward-plot-every", type=int, default=100, help="每多少轮更新一次 reward/KPM 曲线图")
	parser.add_argument("--reward-kpm-weight", type=float, default=0.05, help="将 KPM 作为 episode 级奖励加成的权重")
	parser.add_argument("--move-gain", type=float, default=400.0, help="连续动作映射到鼠标像素位移的缩放倍数")
	parser.add_argument("--max-step", type=int, default=400, help="直接瞄准时单步鼠标移动的最大绝对值")
	parser.add_argument("--batch-size", type=int, default=128, help="TD3 每次更新的 batch size")
	parser.add_argument("--replay-size", type=int, default=50000, help="TD3 回放池容量")
	parser.add_argument("--start-steps", type=int, default=400, help="开始使用 TD3 更新前的最少样本数")
	parser.add_argument("--updates-per-step", type=int, default=1, help="每个环境步执行多少次 TD3 更新")
	parser.add_argument("--policy-noise", type=float, default=0.20, help="TD3 目标策略噪声")
	parser.add_argument("--noise-clip", type=float, default=0.50, help="TD3 目标策略噪声裁剪上限")
	parser.add_argument("--policy-delay", type=int, default=2, help="TD3 actor 更新延迟步数")
	parser.add_argument("--tau", type=float, default=0.005, help="TD3 软更新系数")
	parser.add_argument("--exploration-noise", type=float, default=0.15, help="TD3 动作探索噪声")
	parser.add_argument("--shoot-threshold", type=float, default=0.12, help="动作输出中触发开火的阈值")
	parser.add_argument("--shoot-center-error", type=float, default=0.04, help="只有在准星接近中心时才允许开火")
	parser.add_argument("--proportional-control", action="store_true", help="使用简单的比例控制：直接把 target_dx/target_dy 线性映射为鼠标移动量并在靠近中心时自动开火")
	parser.add_argument("--invert-x", action="store_true", help="反转横轴动作（用于调试方向不一致时启用）")
	parser.add_argument("--invert-y", action="store_true", help="反转纵轴动作（用于调试方向不一致时启用）")
	parser.add_argument("--checkpoint-every", type=int, default=10, help="每多少轮保存一次当前 checkpoint")
	parser.add_argument("--qwen-api-key", type=str, default="", help="可选：显式传入 Qwen API Key（不传则读环境变量）")
	parser.add_argument("--no-target-search-step", type=int, default=16, help="无目标时随机搜索鼠标位移半径")
	parser.add_argument("--no-target-search-interval-sec", type=float, default=1.0, help="无目标时随机搜索的间隔秒数")
	parser.add_argument("--stream-delay-sec", type=float, default=1.0, help="共享视频流的观测延迟，单位秒")
	parser.add_argument("--auto-measure-stream-delay", action="store_true", help="训练开始前自动测量视频流延迟")
	parser.add_argument("--no-auto-measure-stream-delay", dest="auto_measure_stream_delay", action="store_false", help="关闭训练前自动测量视频流延迟")
	parser.set_defaults(auto_measure_stream_delay=True)
	parser.add_argument("--delay-measure-trials", type=int, default=3, help="视频流延迟测量的重复次数")
	parser.add_argument("--delay-measure-move-px", type=int, default=220, help="视频流延迟测量时的鼠标位移像素")
	parser.add_argument("--delay-measure-min-shift", type=float, default=0.06, help="判定点大幅移动的最小归一化位移")
	parser.add_argument("--delay-measure-timeout-sec", type=float, default=3.0, help="单次延迟测量超时秒数")
	parser.add_argument("--delay-measure-poll-sec", type=float, default=0.03, help="延迟测量采样间隔秒数")
	args = parser.parse_args()
	best_save_path = str(args.best_save_path or "").strip() or _get_default_best_model_path(str(args.save_path))
	best_reward_plot_path = str(args.best_reward_plot_path or "").strip() or _get_default_best_plot_path(str(args.reward_plot_path))
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
		gamma=args.gamma,
		seed=args.seed,
		save_path=args.save_path,
		best_save_path=best_save_path,
		reward_plot_path=str(args.reward_plot_path),
		best_reward_plot_path=best_reward_plot_path,
		reward_plot_every=max(1, int(args.reward_plot_every)),
		reward_kpm_weight=float(args.reward_kpm_weight),
		move_gain=float(args.move_gain),
		max_step=int(args.max_step),
		batch_size=max(1, int(args.batch_size)),
		replay_size=max(1, int(args.replay_size)),
		start_steps=max(0, int(args.start_steps)),
		updates_per_step=max(1, int(args.updates_per_step)),
		policy_noise=float(args.policy_noise),
		noise_clip=float(args.noise_clip),
		policy_delay=max(1, int(args.policy_delay)),
		tau=float(args.tau),
		exploration_noise=float(args.exploration_noise),
		shoot_threshold=float(args.shoot_threshold),
		shoot_center_error=float(args.shoot_center_error),
		use_proportional_control=bool(getattr(args, "proportional_control", False)),
		invert_x=bool(getattr(args, "invert_x", False)),
		invert_y=bool(getattr(args, "invert_y", False)),
		checkpoint_every=max(1, int(args.checkpoint_every)),
		qwen_api_key=str(args.qwen_api_key or "").strip(),
		no_target_search_step=max(0, int(args.no_target_search_step)),
		no_target_search_interval_sec=max(0.1, float(args.no_target_search_interval_sec)),
		stream_delay_sec=max(0.0, float(args.stream_delay_sec)),
		auto_measure_stream_delay=bool(args.auto_measure_stream_delay),
		delay_measure_trials=max(1, int(args.delay_measure_trials)),
		delay_measure_move_px=max(30, int(args.delay_measure_move_px)),
		delay_measure_min_shift=max(0.01, float(args.delay_measure_min_shift)),
		delay_measure_timeout_sec=max(0.5, float(args.delay_measure_timeout_sec)),
		delay_measure_poll_sec=max(0.01, float(args.delay_measure_poll_sec)),
		target_family=str(args.target_family or "MIXED").upper(),
		resume=bool(args.resume),
		load_path=str(args.load_path or "").strip(),
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
			stream_delay_sec=cfg.stream_delay_sec,
			target_family=cfg.target_family,
		)
	if Path(cfg.shared_state_path).exists() and Path(cfg.shared_frame_path).exists():
		return SharedPointEnv(
			shared_state_path=cfg.shared_state_path,
			shared_frame_path=cfg.shared_frame_path,
			target_disappear_sec=cfg.target_disappear_sec,
			step_dt_sec=cfg.step_dt_sec,
			stream_delay_sec=cfg.stream_delay_sec,
			target_family=cfg.target_family,
		)
	return SimpleCombatEnv(seed=cfg.seed)


def _execute_action(action_name: str, controller: m_actions) -> None:
	cmd = get_action_command(action_name)
	dx, dy = cmd.get("mouse", (0, 0))
	if int(dx) != 0 or int(dy) != 0:
		controller.mouse_move(int(dx), int(dy))
	if bool(cmd.get("shoot", False)):
		controller.mouse_click(hold_sec=0.03)



def _goal_to_vector(goal: str) -> list[float]:
	if goal == "fight":
		return [0.0, 1.0, 0.0]
	if goal == "take_cover":
		return [0.0, 0.0, 1.0]
	return [1.0, 0.0, 0.0]


def _build_td3_state(obs: dict[str, Any], goal: str) -> np.ndarray:
	return np.asarray(
		[
			float(bool(obs.get("target_visible", obs.get("enemy_visible", False)))),
			float(bool(obs.get("enemy_visible", False))),
			max(-1.0, min(1.0, float(obs.get("target_dx", 0.0)))),
			max(-1.0, min(1.0, float(obs.get("target_dy", 0.0)))),
			max(0.0, min(1.0, float(obs.get("aim_error", 1.0)))),
			max(0.0, min(1.0, float(obs.get("enemy_distance", 1.0)))),
			max(0.0, min(1.0, float(obs.get("danger_level", 0.0)))),
			max(0.0, min(1.0, float(obs.get("hp", 100.0)) / 100.0)),
			max(0.0, min(1.0, float(obs.get("ammo", 30.0)) / 30.0)),
			max(0.0, min(1.0, float(obs.get("fight_time_sec", 0.0)) / 5.0)),
			max(0.0, min(1.0, float(obs.get("kill_time_sec", 0.0)) / 5.0)),
			max(0.0, min(1.0, float(obs.get("no_target_time_sec", 0.0)) / 5.0)),
			max(0.0, min(1.0, float(obs.get("shot_fired", 0.0)))),
			float(obs.get("hit", 0.0)),
			float(obs.get("kill", 0.0)),
			float(obs.get("death", 0.0)),
			*_goal_to_vector(goal),
		],
		dtype=np.float32,
	)


def _continuous_action_to_command(action: np.ndarray, obs: dict[str, Any], cfg: TrainConfig) -> tuple[str, int, int, bool]:
	action = np.asarray(action, dtype=np.float32).reshape(-1)
	if action.size < 3:
		action = np.pad(action, (0, 3 - action.size), mode="constant")
	# 很小的输出视为静止，避免在中心附近抖动时持续把准星往下拉。
	deadzone = 0.01
	if abs(float(action[0])) < deadzone:
		action[0] = 0.0
	if abs(float(action[1])) < deadzone:
		action[1] = 0.0
	# 如果启用比例控制（更简单直接）：忽略 actor 的横纵输出，直接根据观测的 target_dx/target_dy 线性映射到像素位移
	visible = bool(obs.get("target_visible", obs.get("enemy_visible", False)))
	if getattr(cfg, "use_proportional_control", False) and visible:
		tx = float(obs.get("target_dx", 0.0))
		ty = float(obs.get("target_dy", 0.0))
		# 直接线性映射到像素（tx,ty 为 -1..1）
		px = float(tx) * cfg.move_gain
		py = float(ty) * cfg.move_gain
		if getattr(cfg, "invert_x", False):
			px = -px
		if getattr(cfg, "invert_y", False):
			py = -py
		move_x = int(max(-cfg.max_step, min(cfg.max_step, round(px))))
		move_y = int(max(-cfg.max_step, min(cfg.max_step, round(py))))
		center_error = float(obs.get("aim_error", 1.0))
		# 当目标足够接近中心时自动开火
		if center_error <= float(cfg.shoot_center_error):
			return "shoot", 0, 0, True
		if move_x == 0 and move_y == 0:
			return "idle", 0, 0, False
		if abs(move_x) >= abs(move_y):
			return ("aim_right" if move_x > 0 else "aim_left"), move_x, move_y, False
		return ("aim_down" if move_y > 0 else "aim_up"), move_x, move_y, False
	# 压缩动作幅度：对 actor 输出做幂次缩放以减小单步位移（靠近0时更小），再乘以 move_gain
	ax = float(action[0])
	ay = float(action[1])
	# 使用指数 >1 但稍小于之前值以提升响应速度（例如 1.4），在 [-1,1] 范围内平衡抑制与响应
	power = 1.4
	scaled_x = (abs(ax) ** power) * (1.0 if ax >= 0.0 else -1.0)
	scaled_y = (abs(ay) ** power) * (1.0 if ay >= 0.0 else -1.0)
	# 支持运行时反转轴，便于调试方向问题
	if getattr(cfg, "invert_x", False):
		scaled_x = -float(scaled_x)
	if getattr(cfg, "invert_y", False):
		scaled_y = -float(scaled_y)
	move_x = int(max(-cfg.max_step, min(cfg.max_step, round(float(scaled_x) * cfg.move_gain))))
	move_y = int(max(-cfg.max_step, min(cfg.max_step, round(float(scaled_y) * cfg.move_gain))))
	shoot_score = float(action[2])
	center_error = float(obs.get("aim_error", 1.0))
	visible = bool(obs.get("target_visible", obs.get("enemy_visible", False)))

	if visible and shoot_score >= cfg.shoot_threshold and center_error <= cfg.shoot_center_error:
		return "shoot", 0, 0, True
	if move_x == 0 and move_y == 0:
		return "idle", 0, 0, False
	if abs(move_x) >= abs(move_y):
		return ("aim_right" if move_x > 0 else "aim_left"), move_x, move_y, False
	return ("aim_down" if move_y > 0 else "aim_up"), move_x, move_y, False


def _should_update_on_obs(obs: dict[str, Any]) -> bool:
	return bool(obs.get("target_visible", obs.get("enemy_visible", False)))


def _try_measure_stream_delay(env: Any, controller: Any, cfg: TrainConfig) -> float | None:
	"""训练前测量视频流延迟：以鼠标大幅移动到点位大幅变化的耗时作为估计。"""
	if not isinstance(env, SharedPointEnv) or controller is None:
		return None

	delays: list[float] = []
	for trial_idx in range(max(1, int(cfg.delay_measure_trials))):
		visible_obs: dict[str, Any] | None = None
		wait_deadline = time.monotonic() + 6.0
		while time.monotonic() < wait_deadline:
			obs = env.get_observation()
			if _should_update_on_obs(obs):
				visible_obs = obs
				break
			time.sleep(max(0.01, cfg.delay_measure_poll_sec))

		if visible_obs is None:
			continue

		base_dx = float(visible_obs.get("target_dx", 0.0))
		base_dy = float(visible_obs.get("target_dy", 0.0))
		t0 = time.monotonic()
		move_px = int(max(30, cfg.delay_measure_move_px))
		controller.mouse_move(move_px, 0)

		detected_delay: float | None = None
		measure_deadline = t0 + max(0.5, float(cfg.delay_measure_timeout_sec))
		while time.monotonic() < measure_deadline:
			obs = env.get_observation()
			if _should_update_on_obs(obs):
				dx = float(obs.get("target_dx", 0.0))
				dy = float(obs.get("target_dy", 0.0))
				shift = ((dx - base_dx) ** 2 + (dy - base_dy) ** 2) ** 0.5
				if shift >= float(cfg.delay_measure_min_shift):
					detected_delay = time.monotonic() - t0
					break
			time.sleep(max(0.01, cfg.delay_measure_poll_sec))

		# 粗略回中，避免连续测量把视角越拉越偏。
		controller.mouse_move(-move_px, 0)
		time.sleep(0.08)

		if detected_delay is not None:
			delays.append(float(detected_delay))
			print(f"[train] delay_measure trial={trial_idx + 1} delay_sec={detected_delay:.3f}", flush=True)

	if not delays:
		return None

	delays.sort()
	median_delay = float(delays[len(delays) // 2])
	measured = max(0.05, min(3.0, median_delay))
	env.stream_delay_sec = float(measured)
	return measured


def _auto_detect_axis_inversion(env: Any, controller: Any, cfg: TrainConfig) -> tuple[bool, bool] | None:
	"""自动检测鼠标横/纵轴方向是否需要反转。

	方法：对可见目标分别做小幅正向像素移动，检测归一化的 target_dx/target_dy 的变化方向。
	如果移动为正但观测到的 target_dx/dy 反向变化（符号相反），则认为该轴需要反转。
	返回 (invert_x, invert_y) 或 None（检测失败）。
	"""
	if not isinstance(env, SharedPointEnv) or controller is None:
		return None

	try:
		wait_deadline = time.monotonic() + 4.0
		visible_obs = None
		while time.monotonic() < wait_deadline:
			obs = env.get_observation()
			if _should_update_on_obs(obs):
				visible_obs = obs
				break
			time.sleep(max(0.01, cfg.delay_measure_poll_sec))
		if visible_obs is None:
			return None

		base_dx = float(visible_obs.get("target_dx", 0.0))
		base_dy = float(visible_obs.get("target_dy", 0.0))

		move_px = int(max(20, min(200, cfg.delay_measure_move_px // 3)))
		# 横轴检测：向右移动 move_px
		t0 = time.monotonic()
		controller.mouse_move(move_px, 0)
		detected_dx = None
		deadline = t0 + 1.0
		while time.monotonic() < deadline:
			obs2 = env.get_observation()
			if _should_update_on_obs(obs2):
				dx2 = float(obs2.get("target_dx", 0.0))
				if abs(dx2 - base_dx) >= float(cfg.delay_measure_min_shift) * 0.25:
					detected_dx = dx2 - base_dx
					break
			time.sleep(max(0.01, cfg.delay_measure_poll_sec))
		# 回中
		controller.mouse_move(-move_px, 0)
		time.sleep(0.06)

		# 纵轴检测：向下移动 move_px
		t0 = time.monotonic()
		controller.mouse_move(0, move_px)
		detected_dy = None
		deadline = t0 + 1.0
		while time.monotonic() < deadline:
			obs3 = env.get_observation()
			if _should_update_on_obs(obs3):
				dy3 = float(obs3.get("target_dy", 0.0))
				if abs(dy3 - base_dy) >= float(cfg.delay_measure_min_shift) * 0.25:
					detected_dy = dy3 - base_dy
					break
			time.sleep(max(0.01, cfg.delay_measure_poll_sec))
		controller.mouse_move(0, -move_px)
		time.sleep(0.06)

		invert_x = False
		invert_y = False
		if detected_dx is not None:
			# 如果向右移动但 target_dx 变小（负），说明方向相反
			if float(detected_dx) < 0.0:
				invert_x = True
		if detected_dy is not None:
			# 如果向下移动但 target_dy 变小（负），说明方向相反
			if float(detected_dy) < 0.0:
				invert_y = True

		print(f"[train] auto_detect_axis -> invert_x={invert_x} invert_y={invert_y}", flush=True)
		return (invert_x, invert_y)
	except Exception:
		return None


def _build_llm_kill_counter(shared_frame_path: str, explicit_api_key: str = "") -> Any | None:
	qwen_api_key = get_resolve_qwen_api_key(explicit_api_key)
	if not qwen_api_key:
		print("[train] LLM kill counter=disabled (missing API key)", flush=True)
		return None
	try:
		cv2 = importlib.import_module("cv2")
	except Exception:
		return None
	try:
		qwen_client = get_qwen_location_client(api_key=qwen_api_key, model="qwen3.6-plus")
	except Exception:
		print("[train] LLM kill counter=disabled (qwen client init failed)", flush=True)
		return None

	def _query(prev_obs: dict[str, Any], curr_obs: dict[str, Any], action_name: str, manager_goal: str) -> dict[str, Any]:
		frame = None
		path = str(shared_frame_path or "").strip()
		if path:
			frame = cv2.imread(path)
		summary_text = (
			f"prev_visible={bool(prev_obs.get('target_visible', prev_obs.get('enemy_visible', False)))}; "
			f"curr_visible={bool(curr_obs.get('target_visible', curr_obs.get('enemy_visible', False)))}; "
			f"aim_error={float(curr_obs.get('aim_error', 1.0)):.4f}; "
			f"target_dx={float(curr_obs.get('target_dx', 0.0)):.4f}; "
			f"target_dy={float(curr_obs.get('target_dy', 0.0)):.4f}; "
			f"no_target_time_sec={float(curr_obs.get('no_target_time_sec', 0.0)):.4f}; "
			f"action={action_name}; goal={manager_goal}"
		)
		if frame is None:
			return {"kill_count": 0, "reason": "shared_frame_missing"}

		count_payload = get_query_kill_count_from_frame(cv2, frame, qwen_client, (0.0, 0.0, 1.0, 1.0), summary_text)
		try:
			current_count = int(max(0, int(count_payload.get("kill_count", 0))))
		except Exception:
			current_count = 0

		# 返回当前累计击杀数，由 reward 层做差值判断本次是否新增击杀。
		return {"kill_count": int(current_count), "reason": count_payload.get("reason", "")}

	return _query


def train_loop(cfg: TrainConfig) -> None:
	random.seed(cfg.seed)
	np.random.seed(cfg.seed)
	env = _make_env(cfg)
	reward_history: list[float] = []
	reward_kpm_history: list[float] = []
	best_reward = float("-inf")
	best_episode_idx = 0
	best_reward_kpm = 0.0
	controller = m_actions() if (cfg.apply_actions or cfg.env_mode != "smoke") else None
	llm_kill_counter = _build_llm_kill_counter(cfg.shared_frame_path, cfg.qwen_api_key) if cfg.env_mode != "smoke" else None
	# 自动检测轴方向（左右/上下是否需反转）并应用到 cfg
	if isinstance(env, SharedPointEnv) and controller is not None and cfg.env_mode != "smoke":
		detected = _auto_detect_axis_inversion(env, controller, cfg)
		if isinstance(detected, tuple) and len(detected) == 2:
			cfg.invert_x, cfg.invert_y = bool(detected[0]), bool(detected[1])

	if cfg.auto_measure_stream_delay and isinstance(env, SharedPointEnv) and controller is not None and cfg.env_mode != "smoke":
		measured_delay = _try_measure_stream_delay(env, controller, cfg)
		if measured_delay is not None:
			cfg.stream_delay_sec = float(measured_delay)
			print(f"[train] stream_delay_sec auto-measured: {cfg.stream_delay_sec:.3f}", flush=True)
		else:
			print(f"[train] stream_delay_sec auto-measure failed, keep configured value: {cfg.stream_delay_sec:.3f}", flush=True)
	feedback_delay_steps = 1
	if isinstance(env, SharedPointEnv):
		feedback_delay_steps = max(1, int(round(cfg.stream_delay_sec / max(0.001, float(cfg.step_dt_sec)))))
		print(f"[train] feedback_delay_steps={feedback_delay_steps} (stream_delay_sec={cfg.stream_delay_sec:.3f}, step_dt_sec={cfg.step_dt_sec:.3f})", flush=True)
	stop_requested = False
	checkpoint_path = str(cfg.load_path or cfg.save_path)
	loaded_agent: TD3Agent | None = None
	loaded_buffer: ReplayBuffer | None = None
	checkpoint_meta: dict[str, Any] = {}
	agent: TD3Agent | None = None
	replay_buffer: ReplayBuffer | None = None
	train_stats = None
	current_episode = 0
	last_saved_episode = 0
	last_saved_total_reward = 0.0

	def _handle_stop(signum, frame):
		nonlocal stop_requested
		stop_requested = True

	signal.signal(signal.SIGINT, _handle_stop)
	signal.signal(signal.SIGTERM, _handle_stop)

	if cfg.env_mode != "smoke":
		print(
			f"[train] env_mode={cfg.env_mode} apply_actions={bool(controller is not None)} "
			f"target_disappear_sec={cfg.target_disappear_sec:.2f} resume={cfg.resume} "
			f"load_path={checkpoint_path} reward_plot_every={cfg.reward_plot_every} reward_kpm_weight={cfg.reward_kpm_weight}",
			flush=True,
		)

	try:
		obs = env.reset()
		goal = "search"
		state = _build_td3_state(obs, goal)
		if cfg.resume and checkpoint_path:
			loaded = TD3Agent.load(checkpoint_path)
			if loaded is not None:
				loaded_agent, loaded_buffer, checkpoint_meta = loaded
				print(f"[train] 已加载 TD3 checkpoint: {checkpoint_path}", flush=True)
				if loaded_agent.state_dim != int(state.size):
					print(
						f"[train] checkpoint state_dim={loaded_agent.state_dim} 与当前 state_dim={state.size} 不一致，重新初始化",
						flush=True,
					)
					loaded_agent = None
					loaded_buffer = None

		agent = loaded_agent or TD3Agent(
			state_dim=int(state.size),
			action_dim=3,
			action_limit=1.0,
			gamma=cfg.gamma,
			tau=cfg.tau,
			policy_noise=cfg.policy_noise,
			noise_clip=cfg.noise_clip,
			policy_delay=cfg.policy_delay,
		)
		replay_buffer = loaded_buffer or ReplayBuffer(state_dim=int(state.size), action_dim=3, capacity=cfg.replay_size)
		current_episode = int(checkpoint_meta.get("episode_idx", 0))
		best_reward = float(checkpoint_meta.get("best_reward", best_reward))
		best_episode_idx = int(checkpoint_meta.get("best_episode_idx", best_episode_idx))
		print(
			f"[train] TD3 device={getattr(agent, 'device', 'unknown')} batch_size={cfg.batch_size} replay_size={cfg.replay_size} "
			f"start_steps={cfg.start_steps} updates_per_step={cfg.updates_per_step} move_gain={cfg.move_gain} max_step={cfg.max_step}",
			flush=True,
		)
		if llm_kill_counter is not None:
			print("[train] LLM kill counter=enabled", flush=True)

		for ep in range(current_episode + 1, cfg.episodes + 1):
			episode_start = time.monotonic()
			obs = env.reset()
			goal = "search"
			kill_count_state = {"last_kill_count": 0}
			pending_transitions: list[dict[str, Any]] = []
			last_no_target_search_move_t = 0.0
			ep_reward = 0.0
			ep_base_reward = 0.0
			ep_hit = 0.0
			ep_kill = 0.0
			ep_death = 0.0
			ep_updates = 0

			for step_idx in range(cfg.max_steps):
				if stop_requested:
					break

				if not _should_update_on_obs(obs):
					now = time.monotonic()
					if last_no_target_search_move_t <= 0.0:
						last_no_target_search_move_t = now - cfg.no_target_search_interval_sec
					if (now - last_no_target_search_move_t) >= cfg.no_target_search_interval_sec:
						search_radius = int(max(0, cfg.no_target_search_step))
						if search_radius > 0:
							recover_dx = random.randint(-search_radius, search_radius)
							recover_dy = random.randint(-search_radius, search_radius)
							if controller is not None and cfg.env_mode != "smoke":
								controller.mouse_move(recover_dx, recover_dy)
						last_no_target_search_move_t = now
					obs, _ = env.step("aim_up", goal)
					continue
				last_no_target_search_move_t = 0.0

				maybe_goal = get_manager_goal(obs, step_idx, cfg.manager_interval)
				if maybe_goal:
					goal = maybe_goal

				enemy_feedback = get_enemy_feedback(obs)
				worker_obs = dict(obs)
				worker_obs.update(enemy_feedback)

				state = _build_td3_state(worker_obs, goal)
				if replay_buffer.size < max(1, cfg.start_steps):
					action = np.random.uniform(-1.0, 1.0, size=3).astype(np.float32)
				else:
					action = agent.select_action(state, noise_scale=cfg.exploration_noise, deterministic=False)

				action_name, move_dx, move_dy, should_click = _continuous_action_to_command(action, worker_obs, cfg)
				if controller is not None and cfg.env_mode != "smoke":
					if move_dx != 0 or move_dy != 0:
						controller.mouse_move(move_dx, move_dy)
					if should_click:
						controller.mouse_click(hold_sec=0.03)

				next_obs, done = env.step(action_name, goal)

				pending_transitions.append(
					{
						"worker_obs": worker_obs,
						"state": state,
						"action": action,
						"action_name": action_name,
						"goal": goal,
					}
				)

				if len(pending_transitions) >= max(1, feedback_delay_steps):
					ready = pending_transitions.pop(0)
					reward, _reward_items = get_reward(
						ready["worker_obs"],
						next_obs,
						str(ready["action_name"]),
						str(ready["goal"]),
						kill_count_reader=llm_kill_counter,
						kill_count_state=kill_count_state,
					)
					ep_reward += reward
					ep_base_reward += reward
					ep_hit += float(next_obs.get("hit", 0.0))
					ep_kill += float(_reward_items.get("llm_kill_delta", _reward_items.get("kill", 0.0)))
					ep_death += float(next_obs.get("death", 0.0))

					next_enemy_feedback = get_enemy_feedback(next_obs)
					next_worker_obs = dict(next_obs)
					next_worker_obs.update(next_enemy_feedback)
					next_state = _build_td3_state(next_worker_obs, str(ready["goal"]))
					done_float = 1.0 if done else 0.0
					replay_buffer.add(ready["state"], ready["action"], reward, next_state, done_float)

					if replay_buffer.size >= max(1, cfg.start_steps):
						for _ in range(max(1, cfg.updates_per_step)):
							train_stats = agent.train_step(replay_buffer, cfg.batch_size)
							ep_updates += 1

				obs = next_obs
				if done:
					break

			# 用最后观测为未结算动作补齐训练样本，避免动作被丢弃。
			if pending_transitions:
				terminal_done = bool(obs.get("kill", 0.0) > 0.5 or obs.get("death", 0.0) > 0.5)
				for idx, ready in enumerate(pending_transitions):
					reward, _reward_items = get_reward(
						ready["worker_obs"],
						obs,
						str(ready["action_name"]),
						str(ready["goal"]),
						kill_count_reader=llm_kill_counter,
						kill_count_state=kill_count_state,
					)
					ep_reward += reward
					ep_base_reward += reward
					ep_hit += float(obs.get("hit", 0.0))
					ep_kill += float(_reward_items.get("llm_kill_delta", _reward_items.get("kill", 0.0)))
					ep_death += float(obs.get("death", 0.0))

					next_enemy_feedback = get_enemy_feedback(obs)
					next_worker_obs = dict(obs)
					next_worker_obs.update(next_enemy_feedback)
					next_state = _build_td3_state(next_worker_obs, str(ready["goal"]))
					done_float = 1.0 if (terminal_done and idx == len(pending_transitions) - 1) else 0.0
					replay_buffer.add(ready["state"], ready["action"], reward, next_state, done_float)

					if replay_buffer.size >= max(1, cfg.start_steps):
						for _ in range(max(1, cfg.updates_per_step)):
							train_stats = agent.train_step(replay_buffer, cfg.batch_size)
							ep_updates += 1

			if controller is not None:
				controller.stop()

			episode_duration_sec = max(time.monotonic() - episode_start, cfg.step_dt_sec)
			ep_kpm = float(ep_kill) * 60.0 / episode_duration_sec
			ep_kpm_bonus = cfg.reward_kpm_weight * ep_kpm
			ep_reward += ep_kpm_bonus
			reward_kpm_history.append(float(ep_kpm))
			reward_history.append(float(ep_reward))

			if ep_reward > best_reward:
				best_reward = float(ep_reward)
				best_reward_kpm = float(ep_kpm)
				best_episode_idx = int(ep)
				agent.save(cfg.best_save_path, replay_buffer=replay_buffer, extra_meta={"episode_idx": ep, "best_reward": best_reward, "best_episode_idx": best_episode_idx})
				_save_reward_plot(reward_history, reward_kpm_history, cfg.best_reward_plot_path, best_episode_idx=best_episode_idx, best_reward=best_reward, best_kpm=best_reward_kpm)
				print(f"[train] 新最佳模型已保存: episode={ep} reward={ep_reward:.2f} -> {cfg.best_save_path}", flush=True)

			if ep % max(1, cfg.reward_plot_every) == 0:
				_save_reward_plot(reward_history, reward_kpm_history, cfg.reward_plot_path, best_episode_idx=best_episode_idx, best_reward=best_reward if best_episode_idx > 0 else 0.0, best_kpm=best_reward_kpm if best_episode_idx > 0 else 0.0)
				print(f"[train] reward/KPM 曲线已更新: {cfg.reward_plot_path} (episode={ep})", flush=True)

			if ep == 1 or ep % 10 == 0 or stop_requested:
				critic_loss = getattr(train_stats, "critic_loss", 0.0) if train_stats is not None else 0.0
				actor_loss = getattr(train_stats, "actor_loss", 0.0) if train_stats is not None else 0.0
				print(
					f"[Episode {ep:03d}] reward={ep_reward:.2f} base_reward={ep_base_reward:.2f} kpm_bonus={ep_kpm_bonus:.2f} kpm={ep_kpm:.2f} hit={ep_hit:.0f} "
					f"kill={ep_kill:.0f} death={ep_death:.0f} updates={ep_updates} critic_loss={critic_loss:.4f} actor_loss={actor_loss:.4f}",
					flush=True,
				)

			if ep % max(1, cfg.checkpoint_every) == 0:
				agent.save(
					cfg.save_path,
					replay_buffer=replay_buffer,
					extra_meta={"episode_idx": ep, "best_reward": best_reward, "best_episode_idx": best_episode_idx},
				)
				last_saved_episode = ep
				last_saved_total_reward = ep_reward
			current_episode = ep

			if stop_requested:
				break
	finally:
		if controller is not None:
			try:
				controller.stop()
			except Exception:
				pass
		if agent is not None and replay_buffer is not None:
			try:
				agent.save(
					cfg.save_path,
					replay_buffer=replay_buffer,
					extra_meta={"episode_idx": current_episode if stop_requested else (ep if "ep" in locals() else current_episode), "best_reward": best_reward, "best_episode_idx": best_episode_idx, "last_saved_episode": last_saved_episode, "last_saved_total_reward": last_saved_total_reward},
				)
			except Exception as exc:
				print(f"[train] 保存 TD3 checkpoint 失败: {exc}", flush=True)
		_save_reward_plot(reward_history, reward_kpm_history, cfg.reward_plot_path, best_episode_idx=best_episode_idx, best_reward=best_reward if best_episode_idx > 0 else 0.0, best_kpm=best_reward_kpm if best_episode_idx > 0 else 0.0)
		print(f"训练已保存到: {cfg.save_path}")


if __name__ == "__main__":
	config = get_train_config_from_args()
	train_loop(config)