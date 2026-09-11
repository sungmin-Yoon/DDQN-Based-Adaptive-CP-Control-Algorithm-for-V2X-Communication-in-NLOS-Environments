from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cv2x_cp.config import load_config
from cv2x_cp.precheck import run_prechecks


def main() -> None:
    parser = argparse.ArgumentParser(description="Run channel, PHY, and oracle gates before training")
    parser.add_argument("--config", default=ROOT / "configs" / "default.json", type=Path)
    parser.add_argument("--output", default=ROOT / "outputs" / "precheck", type=Path)
    args = parser.parse_args()
    result = run_prechecks(load_config(args.config), args.output)
    print(f"precheck_passed={result['passed']} output={args.output}")


if __name__ == "__main__":
    main()

