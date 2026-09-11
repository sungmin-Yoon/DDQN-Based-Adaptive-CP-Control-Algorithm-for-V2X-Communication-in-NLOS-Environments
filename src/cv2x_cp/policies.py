from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .channel import ChannelSnapshot
from .environment import adaptive_delta_samples


@dataclass
class FixedNormalCPPolicy:
    name: str = "Fixed Normal CP"

    def reset(self) -> None:
        pass

    def select(self, snapshot: ChannelSnapshot, observation: np.ndarray) -> int | None:
        return None


@dataclass
class RuleBasedCPPolicy:
    cfg: dict
    rho: float
    name: str = "Rule-based"

    def reset(self) -> None:
        pass

    def select(self, snapshot: ChannelSnapshot, observation: np.ndarray) -> int:
        required = self.rho * snapshot.estimated_ds_s / snapshot.sample_period_s
        return int(np.clip(math.ceil(required), int(self.cfg["cp"]["min_samples"]), int(self.cfg["cp"]["max_samples"])))


class DDQNPolicy:
    name = "DDQN"

    def __init__(self, network, cfg: dict):
        self.network = network
        self.cfg = cfg
        self.action_deltas = (-1, 0, 1)
        self.reset()

    def reset(self) -> None:
        self.current_cp_samples = int(self.cfg["cp"].get("initial_samples", self.cfg["cp"]["fixed_other_samples"]))

    def select(self, snapshot: ChannelSnapshot, observation: np.ndarray) -> int:
        action_index = int(self.network.predict(observation).argmax())
        delta = self.action_deltas[action_index]
        step = adaptive_delta_samples(snapshot, self.current_cp_samples, delta, self.cfg)
        self.current_cp_samples = int(np.clip(
            self.current_cp_samples + delta * step,
            int(self.cfg["cp"]["min_samples"]),
            int(self.cfg["cp"]["max_samples"]),
        ))
        return self.current_cp_samples
