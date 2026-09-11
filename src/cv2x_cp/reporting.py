from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


METRICS = (
    "prr", "los_expected_prr", "nlos_expected_prr",
    "urban_canyon_expected_prr", "goodput_bps_hz", "mean_cp_us",
    "los_mean_cp_us", "nlos_mean_cp_us", "cp_overhead",
    "useful_rf_ee_bits_per_joule", "avoidable_outage_rate",
    "unavoidable_outage_rate", "pir95_ms", "max_consecutive_failures",
    "nlos_feasible_gap", "cp_switch_rate_hz",
)


def read_csv(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def make_reports(summary_files: list[str | Path], output_dir: str | Path, cfg: dict) -> None:
    rows = [row for path in summary_files for row in read_csv(path)]
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(cfg["experiment"]["test_seed"]) + 99)
    confidence = float(cfg["statistics"]["confidence_level"])
    samples = int(cfg["statistics"]["bootstrap_samples"])
    seed_rows, aggregate, differences = [], [], []
    seeds = sorted({int(r["training_seed"]) for r in rows})
    for seed in seeds:
        for policy in ("Fixed Normal CP", "Rule-based", "DDQN"):
            selected = [r for r in rows if int(r["training_seed"]) == seed and r["policy"] == policy]
            item = {"training_seed": seed, "policy": policy}
            item.update({metric: metric_mean(selected, metric) for metric in METRICS})
            seed_rows.append(item)
    for policy in ("Fixed Normal CP", "Rule-based", "DDQN"):
        selected = [r for r in seed_rows if r["policy"] == policy]
        for metric in METRICS:
            values = np.asarray([float(r[metric]) for r in selected], dtype=float)
            values = values[np.isfinite(values)]
            lo, hi = bootstrap_ci(values, rng, samples, confidence)
            aggregate.append({"policy": policy, "metric": metric, "mean": float(values.mean()),
                              "ci_low": lo, "ci_high": hi, "n": len(values)})
    for baseline in ("Fixed Normal CP", "Rule-based"):
        for metric in METRICS:
            ddqn = keyed_seed(seed_rows, "DDQN", metric)
            reference = keyed_seed(seed_rows, baseline, metric)
            keys = sorted(ddqn.keys() & reference.keys())
            delta = np.asarray([ddqn[k] - reference[k] for k in keys])
            lo, hi = bootstrap_ci(delta, rng, samples, confidence)
            differences.append({"comparison": f"DDQN - {baseline}", "metric": metric,
                                "mean_difference": float(delta.mean()), "ci_low": lo,
                                "ci_high": hi, "paired_n": len(delta),
                                "prr_noninferiority_margin": cfg["statistics"]["prr_noninferiority_margin"],
                                "multiple_testing": cfg["statistics"]["multiple_testing"]})
    _write(output / "seed_level_metrics.csv", seed_rows)
    _write(output / "paper_table.csv", aggregate)
    _write(output / "paired_differences.csv", differences)


def keyed(rows: list[dict], policy: str, metric: str) -> dict[tuple[int, int], float]:
    return {(int(r["training_seed"]), int(r["episode"])): float(r[metric])
            for r in rows if r["policy"] == policy and np.isfinite(float(r[metric]))}


def keyed_seed(rows: list[dict], policy: str, metric: str) -> dict[int, float]:
    return {int(r["training_seed"]): float(r[metric]) for r in rows
            if r["policy"] == policy and np.isfinite(float(r[metric]))}


def metric_mean(rows: list[dict], metric: str) -> float:
    values = [float(r[metric]) for r in rows if metric in r and r[metric] != ""]
    return float(np.nanmean(values)) if values else float("nan")


def bootstrap_ci(values: np.ndarray, rng, samples: int, confidence: float) -> tuple[float, float]:
    if len(values) == 0:
        return float("nan"), float("nan")
    means = np.empty(samples)
    for index in range(samples):
        means[index] = rng.choice(values, len(values), replace=True).mean()
    alpha = (1 - confidence) / 2
    return float(np.quantile(means, alpha)), float(np.quantile(means, 1 - alpha))


def _write(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
