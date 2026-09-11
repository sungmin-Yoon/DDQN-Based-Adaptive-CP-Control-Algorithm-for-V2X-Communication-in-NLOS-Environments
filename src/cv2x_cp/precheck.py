from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .channel import UrbanTraceGenerator
from .environment import cp_sweep_samples
from .phy import abstract_link, waveform_link
from .standards import maximum_doppler_hz, normal_cp_overhead_ratio


def run_prechecks(cfg: dict, output_dir: str | Path) -> dict:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    generator = UrbanTraceGenerator(cfg)
    trace = generator.generate(int(cfg["experiment"]["validation_seed"]) - 1)
    los = np.asarray([s.true_ds_s for s in trace if s.state == "LOS"])
    nlos = np.asarray([s.true_ds_s for s in trace if s.state != "LOS"])
    stress = np.asarray([s.true_ds_s for s in trace if s.state == "UrbanCanyon"])
    ds_ratio = float(nlos.mean() / los.mean())
    stress_ratio = float(stress.mean() / los.mean())
    actions = cp_sweep_samples(cfg)
    # Fixed, pre-registered subset; calibration and validation cover the full
    # 20 s trajectory and do not overlap, so LOS recovery/urban-canyon
    # snapshots all participate in the adaptive-problem gate.
    calibration_indices = np.linspace(0, len(trace) - 2, 10, dtype=int)
    validation_indices = np.clip(calibration_indices + 1, 0, len(trace) - 1)
    calibration = [trace[int(i)] for i in calibration_indices]
    validation = [trace[int(i)] for i in validation_indices]
    calibration_rows = [_compare_snapshot(s, actions, cfg, 600_000 + i) for i, s in enumerate(calibration)]
    validation_rows = [_compare_snapshot(s, actions, cfg, 700_000 + i) for i, s in enumerate(validation)]
    mae = float(np.mean([r["mae"] for r in validation_rows]))
    ranking = float(np.mean([r["ranking_agreement"] for r in validation_rows]))
    sensitive = float(np.mean([r["abstract_prr_range"] >= 0.05 for r in validation_rows]))
    oracle_rows = calibration_rows + validation_rows
    oracle = [r["oracle_action"] for r in oracle_rows]
    dominant_share = max(oracle.count(a) for a in actions) / len(oracle)
    oracle_los = [r["oracle_action"] for r in oracle_rows if r["state"] == "LOS"]
    oracle_stress = [r["oracle_action"] for r in oracle_rows if r["state"] == "UrbanCanyon"]
    stress_minus_los = float(np.mean(oracle_stress) - np.mean(oracle_los)) if oracle_los and oracle_stress else float("-inf")
    gates = cfg["gates"]
    checks = {
        "channel_nlos_ds_larger": ds_ratio >= float(gates["min_nlos_to_los_ds_ratio"]),
        "channel_stress_ds_larger": stress_ratio >= float(gates["min_stress_to_los_ds_ratio"]),
        "phy_prr_mae": mae <= float(gates["max_prr_mae"]),
        "phy_action_ranking": ranking >= float(gates["min_action_ranking_agreement"]),
        "cp_sensitive_fraction": sensitive >= float(gates["min_cp_sensitive_fraction"]),
        "oracle_not_collapsed": dominant_share <= float(gates["max_oracle_action_share"]),
        "oracle_uses_longer_cp_in_stress": stress_minus_los >= float(gates["min_oracle_stress_minus_los_cp_samples"]),
        "prr_monotone_with_cp": all(r["prr_monotone"] for r in calibration_rows + validation_rows),
        "normal_cp_overhead": abs(normal_cp_overhead_ratio(1024) - 1 / 15) < 1e-12,
        "doppler_mapping": maximum_doppler_hz(5.9e9, 60.0) > 300.0,
    }
    result = {
        "passed": all(checks.values()), "checks": checks, "nlos_to_los_mean_ds_ratio": ds_ratio,
        "stress_to_los_mean_ds_ratio": stress_ratio,
        "validation_prr_mae": mae, "ranking_agreement": ranking,
        "cp_sensitive_fraction": sensitive, "oracle_dominant_action_share": dominant_share,
        "oracle_stress_minus_los_cp_samples": stress_minus_los,
        "calibration_trace_indices": [s.step for s in calibration],
        "validation_trace_indices": [s.step for s in validation],
        "parameter_mapping": {"carrier_hz": cfg["radio"]["carrier_hz"],
                              "speed_kmh": cfg["radio"]["speed_kmh"],
                              "normal_cp_native_samples": [80, 72],
                              "nlosv_included": False,
                              "urban_canyon_stress_included": True},
    }
    with (output / "precheck_results.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    if not result["passed"]:
        raise RuntimeError("Precheck failed; inspect precheck_results.json and do not train")
    return result


def require_passed_precheck(path: str | Path) -> None:
    with Path(path).open("r", encoding="utf-8") as handle:
        result = json.load(handle)
    if result.get("passed") is not True:
        raise RuntimeError("A passing, pre-registered precheck is required before training")


def _compare_snapshot(snapshot, actions, cfg, seed: int) -> dict:
    abstract = np.asarray([abstract_link(snapshot, a, cfg).expected_prr for a in actions])
    waveform = np.asarray([waveform_link(snapshot, a, cfg, seed).expected_prr for a in actions])
    abstract_order = np.argsort(np.argsort(abstract))
    waveform_order = np.argsort(np.argsort(waveform))
    agreement = float(np.mean(abstract_order == waveform_order))
    feasible = float(np.max(abstract))
    reliable = np.flatnonzero(abstract >= feasible - float(cfg["phy"]["reliability_tolerance"]))
    oracle_index = int(reliable[0]) if len(reliable) else int(np.argmax(abstract))
    return {"mae": float(np.mean(np.abs(abstract - waveform))),
            "state": snapshot.state,
            "ranking_agreement": agreement, "abstract_prr_range": float(np.ptp(abstract)),
            "oracle_action": actions[oracle_index],
            "prr_monotone": bool(np.all(np.diff(abstract) >= -1e-12))}
