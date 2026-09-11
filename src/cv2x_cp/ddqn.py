from __future__ import annotations

from collections import deque
import math
import random

import numpy as np
import torch
from torch import nn


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_sizes: list[int]):
        super().__init__()
        layers: list[nn.Module] = []
        width = state_dim
        for hidden in hidden_sizes:
            layers.extend((nn.Linear(width, int(hidden)), nn.ReLU()))
            width = int(hidden)
        layers.append(nn.Linear(width, action_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

    def predict(self, observation: np.ndarray) -> np.ndarray:
        device = next(self.parameters()).device
        with torch.no_grad():
            x = torch.as_tensor(observation, dtype=torch.float32, device=device)
            if x.ndim == 1:
                x = x.unsqueeze(0)
            return self(x).cpu().numpy()


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.data = deque(maxlen=int(capacity))

    def __len__(self):
        return len(self.data)

    def append(self, state, action, reward, next_state, done):
        self.data.append((np.asarray(state), int(action), float(reward), np.asarray(next_state), bool(done)))

    def sample(self, size: int, rng: random.Random, device: torch.device):
        rows = rng.sample(self.data, int(size))
        states, actions, rewards, next_states, dones = zip(*rows)
        return (
            torch.as_tensor(np.stack(states), dtype=torch.float32, device=device),
            torch.as_tensor(actions, dtype=torch.long, device=device),
            torch.as_tensor(rewards, dtype=torch.float32, device=device),
            torch.as_tensor(np.stack(next_states), dtype=torch.float32, device=device),
            torch.as_tensor(dones, dtype=torch.float32, device=device),
        )


def seed_everything(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic, warn_only=True)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(name)


def epsilon_at(step: int, cfg: dict) -> float:
    start, end = float(cfg["epsilon_start"]), float(cfg["epsilon_end"])
    return end + (start - end) * math.exp(-5 * step / max(int(cfg["epsilon_decay_steps"]), 1))


def optimize(online, target, optimizer, batch, gamma: float, clip: float) -> float:
    states, actions, rewards, next_states, dones = batch
    chosen = online(states).gather(1, actions[:, None]).squeeze(1)
    with torch.no_grad():
        # Double-DQN: online network selects; target network evaluates.
        next_actions = online(next_states).argmax(1, keepdim=True)
        next_values = target(next_states).gather(1, next_actions).squeeze(1)
        targets = rewards + float(gamma) * (1 - dones) * next_values
    loss = nn.functional.smooth_l1_loss(chosen, targets)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(online.parameters(), float(clip))
    optimizer.step()
    return float(loss.detach().cpu())


def save_checkpoint(path, network, seed: int, episode: int, validation_return: float, actions):
    torch.save({"state_dict": network.state_dict(), "seed": seed, "episode": episode,
                "validation_return": validation_return, "actions": list(actions)}, path)


def load_checkpoint(path, cfg: dict, device_name: str):
    device = resolve_device(device_name)
    payload = torch.load(path, map_location=device, weights_only=False)
    actions = (-1, 0, 1)
    if tuple(payload["actions"]) != actions:
        raise ValueError("Checkpoint action space differs from configuration")
    network = QNetwork(8, len(actions), cfg["ddqn"]["hidden_sizes"]).to(device)
    network.load_state_dict(payload["state_dict"])
    network.eval()
    return network, payload
