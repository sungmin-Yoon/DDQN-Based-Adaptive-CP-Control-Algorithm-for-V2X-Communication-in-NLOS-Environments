from __future__ import annotations

import numpy as np

from .channel import ChannelSnapshot, UrbanTraceGenerator
from .phy import abstract_link


def observation_vector(
    snapshot: ChannelSnapshot,
    recent_prr: float,
    current_cp_samples: int,
    previous_switch: int,
    cfg: dict,
) -> np.ndarray:
    max_action = int(cfg["cp"]["max_samples"])
    ts = snapshot.sample_period_s
    return np.asarray([
        np.clip(snapshot.estimated_ds_s / (max_action * ts), 0, 2),
        np.clip(snapshot.estimated_max_delay_s / (max_action * ts), 0, 2),
        np.clip((snapshot.snr_db + 20) / 60, 0, 1),
        np.clip((snapshot.rsrp_dbm + 140) / 100, 0, 1),
        np.clip(recent_prr, 0, 1),
        current_cp_samples / max_action,
        abs(previous_switch) / max_action,
        np.clip(snapshot.doppler_hz / 1000, 0, 1),
    ], dtype=np.float32)


class AdaptiveCPEnvironment:
    state_dim = 8

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.action_deltas = (-1, 0, 1)
        self.actions = self.action_deltas
        self.action_dim = len(self.action_deltas)
        self.generator = UrbanTraceGenerator(cfg)

    def reset(self, seed: int) -> np.ndarray:
        self.trace = self.generator.generate(seed)
        self.index = 0
        initial = int(self.cfg["cp"].get("initial_samples", self.cfg["cp"]["fixed_other_samples"]))
        self.current_cp_samples = int(np.clip(initial, self.cfg["cp"]["min_samples"], self.cfg["cp"]["max_samples"]))
        self.previous_switch = 0
        self.recent_prr = 1.0
        return observation_vector(self.trace[0], self.recent_prr, self.current_cp_samples, 0, self.cfg)

    def step(self, action_index: int):
        snapshot = self.trace[self.index]
        delta = self.action_deltas[int(action_index)]
        previous = self.current_cp_samples
        step = adaptive_delta_samples(snapshot, previous, delta, self.cfg)
        action = int(np.clip(previous + delta * step, self.cfg["cp"]["min_samples"], self.cfg["cp"]["max_samples"]))
        metric = abstract_link(snapshot, action, self.cfg)
        # In this PHY model PRR is monotone in protected channel energy; the
        # precheck verifies that assumption across the registered CP sweep.
        feasible = abstract_link(snapshot, int(self.cfg["cp"]["max_samples"]), self.cfg).expected_prr
        reward, terms = reward_value(metric.expected_prr, feasible, metric.cp_overhead,
                                     action, previous, self.cfg,
                                     boundary_hit=(action == previous and delta != 0),
                                     target_cp=target_cp_samples(snapshot, self.cfg),
                                     direction=delta, channel_state=snapshot.state)
        switch = action - previous
        self.current_cp_samples, self.previous_switch = action, switch
        self.recent_prr = 0.8 * self.recent_prr + 0.2 * metric.expected_prr
        self.index += 1
        done = self.index == len(self.trace)
        next_snapshot = self.trace[-1] if done else self.trace[self.index]
        state = observation_vector(next_snapshot, self.recent_prr, action, switch, self.cfg)
        return state, reward, done, {
            "metric": metric, "feasible_prr": feasible, "delta_samples": step,
            "delta_action": delta, "target_cp_samples": target_cp_samples(snapshot, self.cfg),
            **terms,
        }


def reward_value(prr: float, feasible_prr: float, overhead: float,
                 action: int, previous_action: int, cfg: dict,
                 boundary_hit: bool = False, target_cp: int | None = None,
                 direction: int = 0, channel_state: str = "") -> tuple[float, dict]:
    p = cfg["reward"]
    tolerance = float(cfg["phy"]["reliability_tolerance"])
    target = float(cfg["phy"]["target_prr"])
    gap = max(0.0, feasible_prr - tolerance - prr)
    target_gap = max(0.0, min(target, feasible_prr) - prr)
    avoidable_outage = float(prr < target and feasible_prr >= target)
    switch = abs(action - previous_action) / int(cfg["cp"]["max_samples"])
    cp_max = int(cfg["cp"]["max_samples"])
    cp_min = int(cfg["cp"]["min_samples"])
    if target_cp is None:
        overprovision = 0.0
        signed_error_after = 0
    else:
        margin = max(int(cfg["cp"]["delta_min_samples"]), 2)
        overprovision = max(0.0, (action - int(target_cp) - margin) / max(cp_max - cp_min, 1))
        signed_error_before = int(target_cp) - int(previous_action)
        signed_error_after = int(target_cp) - int(action)
    wrong_direction = 0.0
    if direction > 0 and target_cp is not None and previous_action >= int(target_cp):
        wrong_direction = 1.0
    elif direction < 0 and target_cp is not None and previous_action <= int(target_cp):
        wrong_direction = 1.0
    elif target_cp is not None and abs(signed_error_after) > abs(signed_error_before) + 1:
        wrong_direction = 1.0
    state_multiplier = float(p.get("los_overprovision_multiplier", 1.0)) if channel_state == "LOS" else 1.0
    rel_term = -float(p["reliability_weight"]) * gap**2
    target_term = -float(p["target_prr_weight"]) * target_gap**2
    outage_term = -float(p["avoidable_outage_weight"]) * avoidable_outage * target_gap**2
    cp_term = -float(p["cp_overhead_weight"]) * overhead
    switch_term = -float(p["switch_weight"]) * switch
    boundary_term = -float(p["boundary_weight"]) * float(boundary_hit)
    over_term = -float(p["overprovision_weight"]) * state_multiplier * overprovision**2
    wrong_term = -float(p.get("wrong_direction_weight", 0.0)) * wrong_direction
    prr_term = float(p["positive_prr_weight"]) * prr
    useful_term = float(p["useful_efficiency_weight"]) * prr * (1.0 - overhead)
    reward = (float(p["offset"]) + prr_term + useful_term + rel_term + target_term
              + outage_term + cp_term + switch_term + boundary_term + over_term + wrong_term)
    return reward, {"reward_reliability": rel_term, "reward_cp": cp_term,
                    "reward_switch": switch_term, "reward_target": target_term,
                    "reward_avoidable_outage": outage_term, "reward_boundary": boundary_term,
                    "reward_overprovision": over_term, "reward_prr": prr_term,
                    "reward_useful_efficiency": useful_term, "reward_wrong_direction": wrong_term,
                    "reliability_gap": gap, "target_gap": target_gap,
                    "overprovision": overprovision, "wrong_direction": wrong_direction,
                    "avoidable_outage": avoidable_outage,
                    "unavoidable_outage": float(prr < target and feasible_prr < target)}


def target_cp_samples(snapshot: ChannelSnapshot, cfg: dict) -> int:
    cp = cfg["cp"]
    boost = float(cfg["reward"].get("urban_target_boost", 1.0)) if snapshot.state == "UrbanCanyon" else 1.0
    target = int(np.ceil(boost * float(cp["target_rho"]) * snapshot.estimated_ds_s / snapshot.sample_period_s))
    return int(np.clip(target, int(cp["min_samples"]), int(cp["max_samples"])))


def adaptive_delta_samples(snapshot: ChannelSnapshot, current_cp_samples: int, direction: int, cfg: dict) -> int:
    cp = cfg["cp"]
    signed_error = target_cp_samples(snapshot, cfg) - int(current_cp_samples)
    if direction > 0:
        error = max(0, signed_error)
    elif direction < 0:
        error = max(0, -signed_error)
    else:
        return 0
    proposed = int(np.ceil(float(cp["delta_gain"]) * error))
    return int(np.clip(proposed, int(cp["delta_min_samples"]), int(cp["delta_max_samples"])))


def cp_sweep_samples(cfg: dict) -> tuple[int, ...]:
    cp = cfg["cp"]
    values = list(range(int(cp["min_samples"]), int(cp["max_samples"]) + 1, int(cp["sweep_step_samples"])))
    if values[-1] != int(cp["max_samples"]):
        values.append(int(cp["max_samples"]))
    for required in (int(cp["fixed_other_samples"]), int(cp["fixed_first_samples"]), int(cp["initial_samples"])):
        if int(cp["min_samples"]) <= required <= int(cp["max_samples"]) and required not in values:
            values.append(required)
    return tuple(sorted(set(values)))
