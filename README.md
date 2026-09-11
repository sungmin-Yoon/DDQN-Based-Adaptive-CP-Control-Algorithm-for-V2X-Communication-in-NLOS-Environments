# Incremental DDQN adaptive CP for urban LTE C-V2X Mode 4

This is a from-scratch research implementation for the prompt's 5.9 GHz,
10 MHz LTE C-V2X PC5 sidelink experiment. The adaptive policies are a
**non-standard PHY extension**. The fixed baseline is the Release-14 Mode-4
normal CP, including its 80/72 native-sample symbol pattern.

This v5 redesign models DDQN as an incremental CP controller:

```text
action 0 = decrease CP index
action 1 = hold CP index
action 2 = increase CP index
```

The controlled 20 s trace is LOS -> UrbanCanyon -> LOS, with the UrbanCanyon
interval fixed at 4-16 s. UrbanCanyon is an explicit CP-limited stress state
layered on the 3GPP-based urban NLOS mapping; it is not presented as standardized
adaptive-CP operation.

Read [DESIGN.md](DESIGN.md) before execution. It records the standard mapping,
action set, timing convention, MDP, PHY abstraction, gates, statistics, figure
rules, and limitations.

## Environment

The intended environment is the user's existing VS Code interpreter with Python
3.14, PyTorch 2.12, and CUDA 13.0. Do not replace an existing CUDA PyTorch build
with the generic package from `requirements.txt`.

```powershell
cd C:\Users\y0318\Desktop\Paper_Experiment_codex\codex\v1
python -m pip install numpy matplotlib pytest
python -m pip install -e . --no-deps
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

## Required execution order

1. Static standard/configuration consistency:

```powershell
python scripts\validate_design.py
```

Expected output: `outputs/design_validation.json`. A mismatch raises `ValueError`
before any training starts.

2. Channel, PHY, and oracle/action gates:

```powershell
python scripts\run_prechecks.py
```

Expected output: `outputs/precheck/precheck_results.json`. A failed threshold
raises `RuntimeError` and prevents the training scripts from proceeding. Do not
relax a threshold after inspecting test results. The waveform check can take time
because it explicitly runs DFT spreading, IFFT/FFT, CP insertion, multipath, MMSE
equalization, and inverse DFT spreading.

3. One seed, including evaluation:

```powershell
python scripts\run_single_seed.py --seed 20260715 --run-name pilot --device cuda
```

4. Full pre-registered protocol:

```powershell
python scripts\run_multiseed.py --run-name paper_v5_5seeds --device cuda
```

Resume completed checkpoints/evaluations without sharing state between seeds:

```powershell
python scripts\run_multiseed.py --run-name paper_v5_5seeds --device cuda --resume
```

Every seed runs 1,200 episodes in v3. Training, validation, and test trace seed
namespaces are disjoint. The five best-validation checkpoints see the same unseen
test traces. Fixed, Rule-based, and DDQN use common packet uniforms for paired
comparisons.

5. Rebuild reports or figures only:

```powershell
python scripts\make_statistics.py --run outputs\paper_v5_5seeds
python scripts\make_paper_figures.py --run outputs\paper_v5_5seeds
```

## Main outputs

```text
outputs/paper_v5_5seeds/
  seed_<seed>/
    training/
      best_model.pt
      final_model.pt
      training_history.npz
      training_metadata.json
    evaluation/
      step_metrics.csv
      episode_summary.csv
  combined_step_metrics.csv
  combined_episode_summary.csv
  statistics/
    seed_level_metrics.csv
    paper_table.csv
    paired_differences.csv
  paper_figures/
    fig1_train_reward.png
    fig2_delay_spread.png
    fig3_cp_duration.png
    fig4_packet_reception_ratio.png
    fig5_useful_rf_energy_efficiency.png
    fig6_summary_metrics_table.png
```

Reward components (`reward_reliability`, `reward_target`,
`reward_avoidable_outage`, `reward_cp`, `reward_switch`, `reward_boundary`) are
retained in step CSVs. The representative episode is selected algorithmically
from median DDQN NLOS PRR, with median goodput as tie-breaker.

Useful RF energy efficiency is deliberately labelled RF-only. It is used to show
how much RF-efficiency is sacrificed while DDQN raises PRR in the urban-canyon
condition, not to claim a standardized energy-saving mechanism.

## Diagnostics

- `Primary protocol requires ...`: configuration drifted from the pre-registered
  5.9 GHz/10 MHz/200-step protocol.
- `Precheck failed`: inspect each boolean and calibration/validation index in
  `precheck_results.json`; do not train until the physical design is corrected.
- `Checkpoint action space differs`: checkpoint and JSON action sets do not match.
- `CUDA was requested but is unavailable`: select the correct VS Code interpreter,
  confirm the CUDA PyTorch build, or use `--device cpu`.
- Missing five histories or evaluation files: finish/resume the multiseed run before
  generating aggregate statistics or figures.

## Reproducibility note

This delivery includes a passing precheck generated with the bundled Codex
Python. Full training still needs the user's PyTorch/CUDA environment.
