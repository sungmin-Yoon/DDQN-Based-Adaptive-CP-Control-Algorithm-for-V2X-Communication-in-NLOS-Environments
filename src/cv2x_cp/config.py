from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        cfg = json.load(handle)
    validate_config(cfg)
    return cfg


def validate_config(cfg: dict[str, Any]) -> None:
    exp, radio, cp = cfg["experiment"], cfg["radio"], cfg["cp"]
    if int(exp["episode_steps"]) != 200:
        raise ValueError("Primary protocol requires exactly 200 steps per episode")
    if float(exp["control_interval_ms"]) != 100.0:
        raise ValueError("Primary protocol requires a 100 ms control interval")
    if float(exp["packet_interval_ms"]) != 10.0:
        raise ValueError("Primary protocol requires a 10 ms packet interval")
    schedule = [
        float(exp["urban_canyon_start_s"]), float(exp["urban_canyon_end_s"]),
    ]
    if schedule != sorted(schedule) or schedule[0] <= 0 or schedule[-1] >= 20.0:
        raise ValueError("Urban-canyon stress schedule must be ordered within the 20 s episode")
    if len(exp["training_seeds"]) != 5 or len(set(exp["training_seeds"])) != 5:
        raise ValueError("Exactly five distinct training seeds are required")
    if int(exp["train_episodes"]) not in (1000, 1200, 2000):
        raise ValueError("Training episodes must be a pre-registered value: 1000, 1200, or 2000")
    if float(radio["carrier_hz"]) != 5.9e9 or float(radio["bandwidth_hz"]) != 10e6:
        raise ValueError("Primary experiment is fixed at 5.9 GHz and 10 MHz")
    if int(radio["fft_size"]) != 1024 or float(radio["sample_rate_hz"]) != 15.36e6:
        raise ValueError("10 MHz native mapping requires NFFT=1024 and Fs=15.36 MHz")
    if float(radio["sample_rate_hz"]) / int(radio["fft_size"]) != float(radio["subcarrier_spacing_hz"]):
        raise ValueError("Sampling rate, FFT size, and subcarrier spacing disagree")
    if int(cp["fixed_first_samples"]) != 80 or int(cp["fixed_other_samples"]) != 72:
        raise ValueError("Normal CP at 15.36 MHz is 80/72 native samples")
    if not (0 < int(cp["min_samples"]) < int(cp["max_samples"])):
        raise ValueError("Adaptive CP sample bounds must be positive and ordered")
    if int(cp["delta_min_samples"]) < 1 or int(cp["delta_max_samples"]) < int(cp["delta_min_samples"]):
        raise ValueError("Adaptive CP delta bounds are invalid")
    if int(cp["sweep_step_samples"]) < 1:
        raise ValueError("Precheck CP sweep step must be positive")
    packets = float(exp["control_interval_ms"]) / float(exp["packet_interval_ms"])
    if not packets.is_integer():
        raise ValueError("Control interval must contain an integer number of packets")


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]
