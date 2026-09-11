from __future__ import annotations

from pathlib import Path

import numpy as np

from .reporting import read_csv

POLICIES = ("Fixed Normal CP", "Rule-based", "DDQN")
COLORS = {"Fixed Normal CP": "#0057ff", "Rule-based": "#00b050", "DDQN": "#ff1f1f"}
LINE_WIDTHS = {"Fixed Normal CP": 3.2, "Rule-based": 3.2, "DDQN": 3.2}
ZORDERS = {"Fixed Normal CP": 2, "Rule-based": 3, "DDQN": 4}
TIME_TICKS_S = np.arange(0, 20.1, 2.0)
OUTPUT_FILES = (
    "fig1_train_reward.png",
    "fig2_delay_spread.png",
    "fig3_cp_duration.png",
    "fig4_packet_reception_ratio.png",
    "fig5_useful_rf_energy_efficiency.png",
    "fig6_summary_metrics_table.png",
)


def make_paper_figures(run_dir: str | Path, output_dir: str | Path) -> None:
    import matplotlib.pyplot as plt

    run_dir, output = Path(run_dir), Path(output_dir)
    _prepare_output_dir(output)
    summaries = read_csv(run_dir / "combined_episode_summary.csv")
    steps = read_csv(run_dir / "combined_step_metrics.csv")
    chosen = representative_episode(summaries)
    representative = [r for r in steps if int(r["training_seed"]) == chosen[0] and int(r["episode"]) == chosen[1]]
    _training_figure(plt, run_dir, output / "fig1_train_reward.png")
    _delay_spread_figure(plt, representative, output / "fig2_delay_spread.png")
    _cp_duration_figure(plt, representative, output / "fig3_cp_duration.png")
    _prr_figure(plt, representative, output / "fig4_packet_reception_ratio.png")
    _useful_rf_ee_figure(plt, steps, output / "fig5_useful_rf_energy_efficiency.png")
    _summary_table_figure(plt, summaries, steps, output / "fig6_summary_metrics_table.png")


def representative_episode(rows: list[dict]) -> tuple[int, int]:
    ddqn = [r for r in rows if r["policy"] == "DDQN"]
    nlos_median = float(np.median([float(r["nlos_expected_prr"]) for r in ddqn]))
    goodput_median = float(np.median([float(r["goodput_bps_hz"]) for r in ddqn]))
    chosen = min(ddqn, key=lambda r: (abs(float(r["nlos_expected_prr"]) - nlos_median),
                                      abs(float(r["goodput_bps_hz"]) - goodput_median)))
    return int(chosen["training_seed"]), int(chosen["episode"])


def _training_figure(plt, run_dir: Path, path: Path) -> None:
    histories = [np.load(p)["rewards"] for p in sorted(run_dir.glob("seed_*/training/training_history.npz"))]
    if len(histories) != 5 or len({len(v) for v in histories}) != 1:
        raise ValueError("Figure 1 requires five complete equal-length histories")
    episodes = len(histories[0])
    raw = np.mean(histories, axis=0)
    window = 20
    moving = np.mean([np.convolve(v, np.ones(window) / window, mode="valid") for v in histories], axis=0)
    fig, ax = plt.subplots(figsize=(7.2, 3.8), constrained_layout=True)
    ax.plot(np.arange(1, episodes + 1), raw, color="#86b9ff", lw=.8, alpha=.55, label="Across-seed raw mean")
    ax.plot(np.arange(window, episodes + 1), moving, color="#003f9e", lw=2.4, label=f"Mean {window}-episode moving average")
    ax.set(xlabel="Episode", ylabel="Return")
    _legend(ax, 1)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def _delay_spread_figure(plt, rows: list[dict], path: Path) -> None:
    fixed = _policy(rows, "Fixed Normal CP")
    x = np.asarray([float(r["time_s"]) for r in fixed])
    fig, ax = plt.subplots(figsize=(7.2, 3.8), constrained_layout=True)
    ax.plot(x, [float(r["true_ds_us"]) for r in fixed], color="#111111", lw=2.6, ls="-", label="True RMS DS")
    ax.plot(x, [float(r["estimated_ds_us"]) for r in fixed], color="#7a1fa2", lw=2.2, ls="-", label="Estimated RMS DS")
    _nlos(ax)
    ax.set(xlabel="Time (s)", ylabel="RMS delay spread (us)")
    ax.set_xticks(TIME_TICKS_S)
    _legend(ax, 1)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def _cp_duration_figure(plt, rows: list[dict], path: Path) -> None:
    fixed = _policy(rows, "Fixed Normal CP")
    x = np.asarray([float(r["time_s"]) for r in fixed])
    fig, ax = plt.subplots(figsize=(7.2, 3.8), constrained_layout=True)
    for policy in POLICIES:
        selected = _policy(rows, policy)
        _plot_policy_line(ax, x, [float(r["cp_mean_us"]) for r in selected], policy)
    _nlos(ax)
    ax.set(xlabel="Time (s)", ylabel="CP duration (us)")
    ax.set_xticks(TIME_TICKS_S)
    _legend(ax, 2)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def _useful_rf_ee_figure(plt, rows: list[dict], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.8), constrained_layout=True)
    for policy in POLICIES:
        x, mean = _time_mean(rows, policy, "useful_rf_ee_bits_per_joule", scale=1e-6)
        _plot_policy_line(ax, x, mean, policy)
    _nlos(ax)
    ax.set(xlabel="Time (s)", ylabel="Useful RF energy efficiency (Mbit/J)")
    ax.set_xticks(TIME_TICKS_S)
    _legend(ax, 2)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def _prr_figure(plt, rows: list[dict], path: Path) -> None:
    window = 10
    fig, ax = plt.subplots(figsize=(7.2, 3.8), constrained_layout=True)
    for policy in POLICIES:
        selected = _policy(rows, policy)
        x = np.asarray([float(r["time_s"]) for r in selected])
        series = np.asarray([float(r["packet_prr"]) for r in selected])
        x_valid, rolling = moving_average_valid(x, series, window)
        _plot_policy_line(ax, x_valid, rolling, policy)
    _nlos(ax)
    ax.set(xlabel="Time (s)", ylabel=f"Packet reception ratio, {window}-step valid MA")
    ax.set_xticks(TIME_TICKS_S)
    _legend(ax, 2)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def _summary_table_figure(plt, summaries: list[dict], steps: list[dict], path: Path) -> None:
    table_rows = summary_table_rows(summaries, steps)
    columns = ["Metric", "Fixed Normal CP", "Rule-based", "DDQN"]
    cells = [[row[col] for col in columns] for row in table_rows]
    fig, ax = plt.subplots(figsize=(10.8, 4.8), constrained_layout=True)
    ax.axis("off")
    table = ax.table(cellText=cells, colLabels=columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1.0, 1.35)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#b8b8b8")
        if row == 0:
            cell.set_text_props(weight="bold", color="white")
            cell.set_facecolor("#30343b")
        elif col == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#f2f4f7")
        else:
            cell.set_facecolor("white")
    fig.savefig(path, dpi=300)
    plt.close(fig)


def summary_table_rows(summaries: list[dict], steps: list[dict]) -> list[dict]:
    seed_level = _seed_level_metrics(summaries, steps)
    metrics = (
        ("PRR", "prr", "{:.4f}"),
        ("LOS expected PRR", "los_expected_prr", "{:.4f}"),
        ("NLOS expected PRR", "nlos_expected_prr", "{:.4f}"),
        ("Urban-canyon expected PRR", "urban_canyon_expected_prr", "{:.4f}"),
        ("Goodput (bit/s/Hz)", "goodput_bps_hz", "{:.4f}"),
        ("Useful RF EE (Mbit/J)", "useful_rf_ee_mbit_per_joule", "{:.2f}"),
        ("Mean CP duration (us)", "mean_cp_us", "{:.3f}"),
        ("LOS mean CP duration (us)", "los_mean_cp_us", "{:.3f}"),
        ("NLOS mean CP duration (us)", "nlos_mean_cp_us", "{:.3f}"),
        ("CP overhead", "cp_overhead", "{:.4f}"),
        ("Avoidable outage rate", "avoidable_outage_rate", "{:.4f}"),
        ("Unavoidable outage rate", "unavoidable_outage_rate", "{:.4f}"),
        ("NLOS feasible gap", "nlos_feasible_gap", "{:.4f}"),
        ("PIR95 (ms)", "pir95_ms", "{:.1f}"),
        ("Max consecutive failures", "max_consecutive_failures", "{:.2f}"),
        ("CP switch rate (Hz)", "cp_switch_rate_hz", "{:.3f}"),
    )
    out = []
    for label, key, fmt in metrics:
        row = {"Metric": label}
        for policy in POLICIES:
            values = [r[key] for r in seed_level if r["policy"] == policy and np.isfinite(r[key])]
            row[policy] = fmt.format(float(np.mean(values))) if values else "n/a"
        out.append(row)
    return out


def _seed_level_metrics(summaries: list[dict], steps: list[dict]) -> list[dict]:
    out = []
    seeds = sorted({int(r["training_seed"]) for r in summaries})
    for seed in seeds:
        for policy in POLICIES:
            selected = [r for r in summaries if int(r["training_seed"]) == seed and r["policy"] == policy]
            selected_steps = [r for r in steps if int(r["training_seed"]) == seed and r["policy"] == policy]
            item = {"training_seed": seed, "policy": policy}
            for key in ("prr", "nlos_expected_prr", "goodput_bps_hz", "mean_cp_us",
                        "cp_overhead", "nlos_feasible_gap", "pir95_ms",
                        "max_consecutive_failures", "los_expected_prr",
                        "urban_canyon_expected_prr",
                        "los_mean_cp_us", "nlos_mean_cp_us",
                        "avoidable_outage_rate", "unavoidable_outage_rate"):
                item[key] = float(np.nanmean([float(r[key]) for r in selected]))
            item["useful_rf_ee_mbit_per_joule"] = float(
                np.nanmean([float(r["useful_rf_ee_bits_per_joule"]) for r in selected]) * 1e-6
            )
            item["cp_switch_rate_hz"] = cp_switch_rate_hz(selected_steps)
            out.append(item)
    return out


def cp_switch_rate_hz(rows: list[dict]) -> float:
    if not rows:
        return float("nan")
    by_episode: dict[int, list[dict]] = {}
    for row in rows:
        by_episode.setdefault(int(row["episode"]), []).append(row)
    rates = []
    for episode_rows in by_episode.values():
        ordered = sorted(episode_rows, key=lambda r: int(r["step"]))
        actions = np.asarray([float(r["cp_mean_samples"]) for r in ordered])
        if len(actions) <= 1:
            continue
        switches = int(np.count_nonzero(np.diff(actions)))
        duration_s = float(ordered[-1]["time_s"]) - float(ordered[0]["time_s"])
        rates.append(switches / max(duration_s, 1e-12))
    return float(np.mean(rates)) if rates else float("nan")


def _policy(rows: list[dict], policy: str) -> list[dict]:
    return sorted((r for r in rows if r["policy"] == policy), key=lambda r: int(r["step"]))


def _nlos(ax) -> None:
    ax.axvspan(4, 16, color="#d9d9d9", alpha=.55, label="Urban canyon NLOS", zorder=0)


def _legend(ax, ncol: int) -> None:
    ymin, ymax = ax.get_ylim()
    span = max(ymax - ymin, 1e-12)
    ax.set_ylim(ymin, ymax + 0.25 * span)
    ax.legend(loc="upper right", ncol=ncol, frameon=True, facecolor="white", framealpha=.95)


def _plot_policy_line(ax, x, y, policy: str) -> None:
    ax.plot(
        x, y, color=COLORS[policy], lw=LINE_WIDTHS[policy], ls="-",
        label=policy, zorder=ZORDERS[policy], solid_capstyle="round",
    )


def _time_mean(rows: list[dict], policy: str, metric: str, scale: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    times = sorted({float(r["time_s"]) for r in rows if r["policy"] == policy})
    x, mean = [], []
    for time_s in times:
        values = np.asarray([float(r[metric]) * scale for r in rows
                             if r["policy"] == policy and float(r["time_s"]) == time_s])
        if len(values):
            x.append(time_s)
            mean.append(float(values.mean()))
    return np.asarray(x), np.asarray(mean)


def moving_average_valid(x: np.ndarray, values: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    if window <= 1 or len(values) < window:
        return x, values
    weights = np.ones(window) / window
    averaged = np.convolve(values, weights, mode="valid")
    offset = window // 2
    return x[offset : offset + len(averaged)], averaged


def _prepare_output_dir(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for path in output.glob("*.png"):
        path.unlink()
