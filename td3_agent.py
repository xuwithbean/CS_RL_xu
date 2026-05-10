"""TD3 agent implementation for continuous mouse control.

This module is designed to be importable even when PyTorch is not installed.
The actual TD3 functionality becomes available once `torch` is present.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception as exc:  # pragma: no cover - import guard for environments without torch
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    TORCH_IMPORT_ERROR = exc
else:
    TORCH_IMPORT_ERROR = None


if torch is not None:
    class Actor(nn.Module):
        def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim),
                nn.Tanh(),
            )

        def forward(self, state: torch.Tensor) -> torch.Tensor:
            return self.net(state)


    class Critic(nn.Module):
        def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
            super().__init__()
            input_dim = state_dim + action_dim
            self.q1 = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )
            self.q2 = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )

        def forward(self, state: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            state_action = torch.cat([state, action], dim=-1)
            return self.q1(state_action), self.q2(state_action)

        def q1_value(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
            state_action = torch.cat([state, action], dim=-1)
            return self.q1(state_action)


    def _soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.mul_(1.0 - tau)
            target_param.data.add_(tau * source_param.data)
else:
    Actor = object  # type: ignore[misc,assignment]
    Critic = object  # type: ignore[misc,assignment]


@dataclass
class TD3TrainStats:
    critic_loss: float = 0.0
    actor_loss: float = 0.0
    q1_mean: float = 0.0
    q2_mean: float = 0.0


class ReplayBuffer:
    def __init__(self, state_dim: int, action_dim: int, capacity: int):
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.capacity = max(1, int(capacity))
        self.states = np.zeros((self.capacity, self.state_dim), dtype=np.float32)
        self.actions = np.zeros((self.capacity, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((self.capacity, self.state_dim), dtype=np.float32)
        self.dones = np.zeros((self.capacity, 1), dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def add(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray, done: float) -> None:
        idx = self.ptr
        self.states[idx] = np.asarray(state, dtype=np.float32).reshape(self.state_dim)
        self.actions[idx] = np.asarray(action, dtype=np.float32).reshape(self.action_dim)
        self.rewards[idx, 0] = float(reward)
        self.next_states[idx] = np.asarray(next_state, dtype=np.float32).reshape(self.state_dim)
        self.dones[idx, 0] = float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if torch is None:
            raise RuntimeError(f"PyTorch is required for TD3: {TORCH_IMPORT_ERROR}")
        batch_size = min(int(batch_size), self.size)
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        indices = np.random.randint(0, self.size, size=batch_size)
        states = torch.as_tensor(self.states[indices], dtype=torch.float32)
        actions = torch.as_tensor(self.actions[indices], dtype=torch.float32)
        rewards = torch.as_tensor(self.rewards[indices], dtype=torch.float32)
        next_states = torch.as_tensor(self.next_states[indices], dtype=torch.float32)
        dones = torch.as_tensor(self.dones[indices], dtype=torch.float32)
        return states, actions, rewards, next_states, dones

    def state_dict(self) -> dict[str, Any]:
        return {
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "capacity": self.capacity,
            "states": self.states,
            "actions": self.actions,
            "rewards": self.rewards,
            "next_states": self.next_states,
            "dones": self.dones,
            "ptr": self.ptr,
            "size": self.size,
        }

    def load_state_dict(self, payload: dict[str, Any]) -> None:
        self.state_dim = int(payload.get("state_dim", self.state_dim))
        self.action_dim = int(payload.get("action_dim", self.action_dim))
        self.capacity = int(payload.get("capacity", self.capacity))
        self.states = np.asarray(payload.get("states", self.states), dtype=np.float32)
        self.actions = np.asarray(payload.get("actions", self.actions), dtype=np.float32)
        self.rewards = np.asarray(payload.get("rewards", self.rewards), dtype=np.float32)
        self.next_states = np.asarray(payload.get("next_states", self.next_states), dtype=np.float32)
        self.dones = np.asarray(payload.get("dones", self.dones), dtype=np.float32)
        self.ptr = int(payload.get("ptr", 0))
        self.size = int(payload.get("size", 0))


class TD3Agent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int = 3,
        action_limit: float = 1.0,
        actor_lr: float = 1e-4,
        critic_lr: float = 1e-3,
        gamma: float = 0.99,
        tau: float = 0.005,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        policy_delay: int = 2,
        device: str | None = None,
        hidden_dim: int = 256,
    ):
        if torch is None:
            raise RuntimeError(f"PyTorch is required for TD3: {TORCH_IMPORT_ERROR}")
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.action_limit = float(action_limit)
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.policy_noise = float(policy_noise)
        self.noise_clip = float(noise_clip)
        self.policy_delay = max(1, int(policy_delay))
        self.total_it = 0
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.actor = Actor(self.state_dim, self.action_dim, hidden_dim=hidden_dim).to(self.device)
        self.actor_target = Actor(self.state_dim, self.action_dim, hidden_dim=hidden_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic = Critic(self.state_dim, self.action_dim, hidden_dim=hidden_dim).to(self.device)
        self.critic_target = Critic(self.state_dim, self.action_dim, hidden_dim=hidden_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=float(actor_lr))
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=float(critic_lr))

    def select_action(self, state: np.ndarray, noise_scale: float = 0.0, deterministic: bool = False) -> np.ndarray:
        if torch is None:
            raise RuntimeError(f"PyTorch is required for TD3: {TORCH_IMPORT_ERROR}")
        state_tensor = torch.as_tensor(np.asarray(state, dtype=np.float32), device=self.device).view(1, -1)
        with torch.no_grad():
            action = self.actor(state_tensor).cpu().numpy().reshape(-1)
        if deterministic:
            return np.clip(action, -1.0, 1.0)
        if noise_scale > 0.0:
            noise = np.random.normal(0.0, float(noise_scale), size=self.action_dim).astype(np.float32)
            action = action + noise
        return np.clip(action, -1.0, 1.0)

    def train_step(self, replay_buffer: ReplayBuffer, batch_size: int) -> TD3TrainStats | None:
        if torch is None:
            raise RuntimeError(f"PyTorch is required for TD3: {TORCH_IMPORT_ERROR}")
        if replay_buffer.size < max(2, int(batch_size)):
            return None

        self.total_it += 1
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)
        state = state.to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device)
        next_state = next_state.to(self.device)
        done = done.to(self.device)

        with torch.no_grad():
            noise = torch.randn_like(action) * self.policy_noise
            noise = noise.clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-1.0, 1.0)
            target_q1, target_q2 = self.critic_target(next_state, next_action)
            target_q = reward + (1.0 - done) * self.gamma * torch.min(target_q1, target_q2)

        current_q1, current_q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss_value = 0.0
        if self.total_it % self.policy_delay == 0:
            actor_loss = -self.critic.q1_value(state, self.actor(state)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            _soft_update(self.actor_target, self.actor, self.tau)
            _soft_update(self.critic_target, self.critic, self.tau)
            actor_loss_value = float(actor_loss.item())

        return TD3TrainStats(
            critic_loss=float(critic_loss.item()),
            actor_loss=actor_loss_value,
            q1_mean=float(current_q1.mean().item()),
            q2_mean=float(current_q2.mean().item()),
        )

    def save(self, path: str, replay_buffer: ReplayBuffer | None = None, extra_meta: dict[str, Any] | None = None) -> None:
        if torch is None:
            raise RuntimeError(f"PyTorch is required for TD3: {TORCH_IMPORT_ERROR}")
        payload: dict[str, Any] = {
            "meta": {
                "state_dim": self.state_dim,
                "action_dim": self.action_dim,
                "action_limit": self.action_limit,
                "gamma": self.gamma,
                "tau": self.tau,
                "policy_noise": self.policy_noise,
                "noise_clip": self.noise_clip,
                "policy_delay": self.policy_delay,
                "total_it": self.total_it,
            },
            "actor": self.actor.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
        }
        if extra_meta:
            payload["meta"].update(extra_meta)
        if replay_buffer is not None:
            payload["replay_buffer"] = replay_buffer.state_dict()
        path_obj = Path(str(path or "").strip())
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path_obj)

    @classmethod
    def load(cls, path: str, device: str | None = None) -> tuple["TD3Agent", ReplayBuffer | None, dict[str, Any]] | None:
        if torch is None:
            raise RuntimeError(f"PyTorch is required for TD3: {TORCH_IMPORT_ERROR}")
        path_obj = Path(str(path or "").strip())
        if not path_obj.exists():
            return None
        load_kwargs = {"map_location": device or ("cuda" if torch.cuda.is_available() else "cpu")}
        try:
            payload = torch.load(path_obj, weights_only=False, **load_kwargs)
        except TypeError:
            payload = torch.load(path_obj, **load_kwargs)
        if not isinstance(payload, dict):
            return None
        meta = dict(payload.get("meta") or {})
        state_dim = int(meta.get("state_dim", 0))
        action_dim = int(meta.get("action_dim", 3))
        agent = cls(
            state_dim=state_dim,
            action_dim=action_dim,
            action_limit=float(meta.get("action_limit", 1.0)),
            gamma=float(meta.get("gamma", 0.99)),
            tau=float(meta.get("tau", 0.005)),
            policy_noise=float(meta.get("policy_noise", 0.2)),
            noise_clip=float(meta.get("noise_clip", 0.5)),
            policy_delay=int(meta.get("policy_delay", 2)),
            device=device,
        )
        agent.total_it = int(meta.get("total_it", 0))
        agent.actor.load_state_dict(payload["actor"])
        agent.actor_target.load_state_dict(payload["actor_target"])
        agent.critic.load_state_dict(payload["critic"])
        agent.critic_target.load_state_dict(payload["critic_target"])
        if payload.get("actor_optimizer"):
            agent.actor_optimizer.load_state_dict(payload["actor_optimizer"])
        if payload.get("critic_optimizer"):
            agent.critic_optimizer.load_state_dict(payload["critic_optimizer"])

        replay_buffer = None
        if payload.get("replay_buffer") and state_dim > 0:
            rb_payload = payload["replay_buffer"]
            replay_buffer = ReplayBuffer(state_dim=state_dim, action_dim=action_dim, capacity=int(rb_payload.get("capacity", 1)))
            replay_buffer.load_state_dict(rb_payload)
        return agent, replay_buffer, meta
