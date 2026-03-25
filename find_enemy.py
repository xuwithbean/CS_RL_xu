# [x]:完成对敌人的识别和位置信息的反馈
"""敌人信息抽取模块（轻量原型）。

说明：
- 当前仓库仍在原型阶段，这里先实现一个统一的数据接口，便于训练代码联调。
- 若接入真实视觉模型（YOLO/模板匹配等），只需要把检测结果整理成本模块的输入格式。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class EnemyFeature:
	"""敌人特征。

	字段约定：
	- enemy_visible: 是否可见敌人
	- enemy_distance: 估计距离，范围 [0, 1]，越小越近
	- aim_error: 准星误差，范围 [0, 1]，越小越准
	- danger_level: 危险程度，范围 [0, 1]
	"""

	enemy_visible: bool
	enemy_distance: float
	aim_error: float
	danger_level: float


def _get_clip01(value: float) -> float:
	return max(0.0, min(1.0, float(value)))


def get_enemy_feature(raw_obs: dict[str, Any] | None) -> EnemyFeature:
	"""将原始观测转换为统一敌人特征。

	输入 raw_obs 示例（可裁剪）：
	{
		"enemy_visible": True,
		"enemy_distance": 0.35,
		"aim_error": 0.42,
		"danger_level": 0.70,
	}
	"""
	if raw_obs is None:
		raw_obs = {}

	enemy_visible = bool(raw_obs.get("enemy_visible", False))
	enemy_distance = _get_clip01(raw_obs.get("enemy_distance", 1.0))
	aim_error = _get_clip01(raw_obs.get("aim_error", 1.0))
	danger_level = _get_clip01(raw_obs.get("danger_level", 0.0))

	return EnemyFeature(
		enemy_visible=enemy_visible,
		enemy_distance=enemy_distance,
		aim_error=aim_error,
		danger_level=danger_level,
	)


def get_enemy_feedback(raw_obs: dict[str, Any] | None) -> dict[str, Any]:
	"""兼容上层训练流程的字典输出接口。"""
	feat = get_enemy_feature(raw_obs)
	return {
		"enemy_visible": feat.enemy_visible,
		"enemy_distance": feat.enemy_distance,
		"aim_error": feat.aim_error,
		"danger_level": feat.danger_level,
	}