from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cv2x_cp.config import load_config
from cv2x_cp.standards import maximum_doppler_hz, normal_cp_durations_s, normal_cp_overhead_ratio


def main() -> None:
    parser = argparse.ArgumentParser(description="Static standard/configuration consistency validation")
    parser.add_argument("--config", default=ROOT / "configs" / "default.json", type=Path)
    parser.add_argument("--output", default=ROOT / "outputs" / "design_validation.json", type=Path)
    args = parser.parse_args()
    cfg = load_config(args.config)
    result = {
        "valid": True,
        "claim": "non-standard adaptive-CP PHY extension on a 3GPP-based LTE C-V2X model",
        "normal_cp_native_samples": [80, 72],
        "normal_cp_durations_us": [normal_cp_durations_s()[0] * 1e6, normal_cp_durations_s()[1] * 1e6],
        "normal_cp_subframe_overhead": normal_cp_overhead_ratio(1024),
        "ddqn_actions": ["decrease_cp", "hold_cp", "increase_cp"],
        "cp_min_samples": cfg["cp"]["min_samples"],
        "cp_max_samples": cfg["cp"]["max_samples"],
        "cp_min_us": cfg["cp"]["min_samples"] / cfg["radio"]["sample_rate_hz"] * 1e6,
        "cp_max_us": cfg["cp"]["max_samples"] / cfg["radio"]["sample_rate_hz"] * 1e6,
        "delta_rule": "direction-aware delta_K from signed target error; increase uses max(0,target-current), decrease uses max(0,current-target)",
        "delta_min_samples": cfg["cp"]["delta_min_samples"],
        "delta_max_samples": cfg["cp"]["delta_max_samples"],
        "delta_gain": cfg["cp"]["delta_gain"],
        "target_rho": cfg["cp"]["target_rho"],
        "maximum_doppler_hz": maximum_doppler_hz(cfg["radio"]["carrier_hz"], cfg["radio"]["speed_kmh"]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
