from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cv2x_cp.config import load_config
from cv2x_cp.evaluation import evaluate_checkpoint
from cv2x_cp.precheck import require_passed_precheck
from cv2x_cp.training import train_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate one pre-registered DDQN seed")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--config", default=ROOT / "configs" / "default.json", type=Path)
    parser.add_argument("--run-name", default="single_seed")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    cfg = load_config(args.config)
    require_passed_precheck(ROOT / "outputs" / "precheck" / "precheck_results.json")
    run = ROOT / "outputs" / args.run_name / f"seed_{args.seed}"
    best = train_seed(cfg, args.seed, run / "training", args.device)
    evaluate_checkpoint(cfg, best, run / "evaluation", args.device)
    print(f"completed seed={args.seed} output={run}")


if __name__ == "__main__":
    main()

