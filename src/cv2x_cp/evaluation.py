from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .channel import UrbanTraceGenerator
from .ddqn import load_checkpoint
from .environment import observation_vector, reward_value, target_cp_samples
from .phy import abstract_link
from .policies import DDQNPolicy, FixedNormalCPPolicy, RuleBasedCPPolicy


def evaluate_checkpoint(cfg: dict, checkpoint: str | Path, output_dir: str | Path,
                        device_name: str = "cpu") -> tuple[list[dict], list[dict]]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    network, payload = load_checkpoint(checkpoint, cfg, device_name)
    policies = (
        FixedNormalCPPolicy(),
        RuleBasedCPPolicy(cfg, float(cfg["cp"]["rule_rho"])),
        DDQNPolicy(network, cfg),
    )
    seed_id = int(payload["seed"])
    generator = UrbanTraceGenerator(cfg)
    packets_per_step = int(cfg["experiment"]["control_interval_ms"] / cfg["experiment"]["packet_interval_ms"])
    all_steps, summaries = [], []
    test_base = int(cfg["experiment"]["test_seed"])
    for episode in range(int(cfg["experiment"]["test_episodes"])):
        trace_seed = test_base + 1009 * episode
        trace = generator.generate(trace_seed)
        # Common uniforms produce paired packet outcomes across policies.
        packet_uniforms = np.random.default_rng(trace_seed + 700_000_000).random((len(trace), packets_per_step))
        feasible = [abstract_link(s, int(cfg["cp"]["max_samples"]), cfg).expected_prr for s in trace]
        for policy in policies:
            policy.reset()
            previous_action = int(cfg["cp"]["fixed_other_samples"])
            previous_switch, recent_prr = 0, 1.0
            packet_successes: list[int] = []
            rows = []
            for snapshot in trace:
                observation = observation_vector(snapshot, recent_prr, previous_action, previous_switch, cfg)
                selected = policy.select(snapshot, observation)
                metric = abstract_link(snapshot, selected, cfg)
                successes = (packet_uniforms[snapshot.step] < metric.expected_prr).astype(int)
                empirical_prr = float(successes.mean())
                action_for_state = int(round(metric.cp_mean_samples))
                direction = int(np.sign(action_for_state - previous_action))
                reward, terms = reward_value(metric.expected_prr, feasible[snapshot.step], metric.cp_overhead,
                                             action_for_state, previous_action, cfg,
                                             target_cp=target_cp_samples(snapshot, cfg),
                                             direction=direction, channel_state=snapshot.state)
                is_non_los = snapshot.state != "LOS"
                is_stress = snapshot.state == "UrbanCanyon"
                row = {
                    "training_seed": seed_id, "episode": episode, "step": snapshot.step,
                    "time_s": snapshot.time_s, "policy": policy.name, "channel_state": snapshot.state,
                    "is_nlos": int(is_non_los), "is_stress": int(is_stress),
                    "true_ds_us": snapshot.true_ds_s * 1e6,
                    "estimated_ds_us": snapshot.estimated_ds_s * 1e6, "snr_db": snapshot.snr_db,
                    "cp_mean_us": metric.cp_mean_s * 1e6, "cp_mean_samples": metric.cp_mean_samples,
                    "cp_overhead": metric.cp_overhead, "post_sinr_db": metric.post_sinr_db,
                    "expected_prr": metric.expected_prr, "packet_prr": empirical_prr,
                    "packet_successes": "".join(str(int(v)) for v in successes),
                    "feasible_prr": feasible[snapshot.step],
                    "feasible_gap": terms["reliability_gap"],
                    "raw_spectral_efficiency": metric.raw_spectral_efficiency,
                    "goodput_bps_hz": metric.goodput_bps_hz,
                    "useful_rf_ee_bits_per_joule": metric.useful_rf_ee_bits_per_joule,
                    "residual_isi": metric.residual_isi, "reward": reward, **terms,
                }
                rows.append(row)
                packet_successes.extend(successes.tolist())
                previous_switch = action_for_state - previous_action
                previous_action = action_for_state
                recent_prr = 0.8 * recent_prr + 0.2 * empirical_prr
            all_steps.extend(rows)
            summaries.append(summarize_episode(rows, packet_successes, cfg))
    _write_csv(output / "step_metrics.csv", all_steps)
    _write_csv(output / "episode_summary.csv", summaries)
    return all_steps, summaries


def summarize_episode(rows: list[dict], successes: list[int], cfg: dict) -> dict:
    nlos = [r for r in rows if r["is_nlos"]]
    los = [r for r in rows if not r["is_nlos"]]
    urban = [r for r in rows if r["channel_state"] == "UrbanCanyon"]
    packet_interval = float(cfg["experiment"]["packet_interval_ms"])
    intervals = packet_inter_reception_ms(successes, packet_interval)
    failures = maximum_failure_burst(successes)
    return {
        "training_seed": rows[0]["training_seed"], "episode": rows[0]["episode"],
        "policy": rows[0]["policy"], "prr": float(np.mean(successes)),
        "los_expected_prr": _mean(rows=los, key="expected_prr"),
        "nlos_expected_prr": float(np.mean([r["expected_prr"] for r in nlos])),
        "urban_canyon_expected_prr": _mean(rows=urban, key="expected_prr"),
        "goodput_bps_hz": float(np.mean([r["goodput_bps_hz"] for r in rows])),
        "mean_cp_us": float(np.mean([r["cp_mean_us"] for r in rows])),
        "los_mean_cp_us": _mean(rows=los, key="cp_mean_us"),
        "nlos_mean_cp_us": _mean(rows=nlos, key="cp_mean_us"),
        "cp_overhead": float(np.mean([r["cp_overhead"] for r in rows])),
        "useful_rf_ee_bits_per_joule": float(np.mean([r["useful_rf_ee_bits_per_joule"] for r in rows])),
        "avoidable_outage_rate": float(np.mean([r["avoidable_outage"] for r in rows])),
        "unavoidable_outage_rate": float(np.mean([r["unavoidable_outage"] for r in rows])),
        "pir95_ms": float(np.quantile(intervals, 0.95)) if intervals else float("nan"),
        "max_pir_ms": float(max(intervals)) if intervals else float("nan"),
        "max_consecutive_failures": failures,
        "cp_switch_rate_hz": cp_switch_rate_hz(rows),
        "nlos_feasible_gap": float(np.mean([r["feasible_gap"] for r in nlos])),
        "return": float(sum(r["reward"] for r in rows)),
    }


def _mean(rows: list[dict], key: str) -> float:
    return float(np.mean([float(r[key]) for r in rows])) if rows else float("nan")


def packet_inter_reception_ms(successes: list[int], interval_ms: float) -> list[float]:
    received = np.flatnonzero(successes)
    return (np.diff(received) * interval_ms).astype(float).tolist() if len(received) >= 2 else []


def maximum_failure_burst(successes: list[int]) -> int:
    best = current = 0
    for value in successes:
        current = 0 if value else current + 1
        best = max(best, current)
    return best


def cp_switch_rate_hz(rows: list[dict]) -> float:
    ordered = sorted(rows, key=lambda r: int(r["step"]))
    actions = np.asarray([float(r["cp_mean_samples"]) for r in ordered])
    if len(actions) <= 1:
        return 0.0
    switches = int(np.count_nonzero(np.diff(actions)))
    duration_s = float(ordered[-1]["time_s"]) - float(ordered[0]["time_s"])
    return switches / max(duration_s, 1e-12)


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
