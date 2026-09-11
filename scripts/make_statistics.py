from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cv2x_cp.config import load_config
from cv2x_cp.reporting import make_reports


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--config", default=ROOT / "configs" / "default.json", type=Path)
    args = parser.parse_args()
    files = sorted(args.run.glob("seed_*/evaluation/episode_summary.csv"))
    if len(files) != 5:
        raise RuntimeError("Statistics require all five seed evaluation files")
    make_reports(files, args.run / "statistics", load_config(args.config))


if __name__ == "__main__":
    main()

