from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cv2x_cp.config import load_config
from cv2x_cp.evaluation import evaluate_checkpoint
from cv2x_cp.plotting import make_paper_figures
from cv2x_cp.precheck import require_passed_precheck
from cv2x_cp.reporting import make_reports
from cv2x_cp.training import train_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Five independent train/evaluate runs with paired test traces")
    parser.add_argument("--config", default=ROOT / "configs" / "default.json", type=Path)
    parser.add_argument("--run-name", default="paper_v5_5seeds")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    require_passed_precheck(ROOT / "outputs" / "precheck" / "precheck_results.json")
    run = ROOT / "outputs" / args.run_name
    summary_files, step_files = [], []
    for seed in cfg["experiment"]["training_seeds"]:
        seed_dir = run / f"seed_{seed}"
        best = seed_dir / "training" / "best_model.pt"
        summary = seed_dir / "evaluation" / "episode_summary.csv"
        steps = seed_dir / "evaluation" / "step_metrics.csv"
        if not (args.resume and training_complete(seed_dir / "training", int(cfg["experiment"]["train_episodes"]))):
            best = train_seed(cfg, int(seed), seed_dir / "training", args.device)
        if not (args.resume and summary.exists() and steps.exists()):
            evaluate_checkpoint(cfg, best, seed_dir / "evaluation", args.device)
        summary_files.append(summary); step_files.append(steps)
    combine_csv(summary_files, run / "combined_episode_summary.csv")
    combine_csv(step_files, run / "combined_step_metrics.csv")
    make_reports(summary_files, run / "statistics", cfg)
    make_paper_figures(run, run / "paper_figures")
    print(f"completed five-seed protocol output={run}")


def combine_csv(paths: list[Path], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    with output.open("w", newline="", encoding="utf-8") as target:
        for path in paths:
            with path.open("r", newline="", encoding="utf-8") as source:
                reader = csv.DictReader(source)
                if writer is None:
                    writer = csv.DictWriter(target, fieldnames=reader.fieldnames)
                    writer.writeheader()
                writer.writerows(reader)


def training_complete(training_dir: Path, required_episodes: int) -> bool:
    metadata = training_dir / "training_metadata.json"
    if not metadata.exists() or not (training_dir / "best_model.pt").exists() or not (training_dir / "final_model.pt").exists():
        return False
    return int(json.loads(metadata.read_text(encoding="utf-8"))["episodes"]) == required_episodes


if __name__ == "__main__":
    main()
