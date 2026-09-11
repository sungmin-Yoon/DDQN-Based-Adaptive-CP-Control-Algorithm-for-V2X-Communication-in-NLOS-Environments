from __future__ import annotations

import json
from pathlib import Path
import random

import numpy as np
import torch

from .ddqn import (QNetwork, ReplayBuffer, epsilon_at, optimize, resolve_device,
                   save_checkpoint, seed_everything)
from .environment import AdaptiveCPEnvironment


def train_seed(cfg: dict, seed: int, output_dir: str | Path, device_name: str = "auto") -> Path:
    if seed not in cfg["experiment"]["training_seeds"]:
        raise ValueError("Seed is not in the pre-registered training seed set")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    deterministic = bool(cfg["experiment"]["deterministic_torch"])
    seed_everything(seed, deterministic)
    device = resolve_device(device_name)
    env = AdaptiveCPEnvironment(cfg)
    params = cfg["ddqn"]
    online = QNetwork(env.state_dim, env.action_dim, params["hidden_sizes"]).to(device)
    target = QNetwork(env.state_dim, env.action_dim, params["hidden_sizes"]).to(device)
    target.load_state_dict(online.state_dict())
    target.eval()
    optimizer = torch.optim.Adam(online.parameters(), lr=float(params["learning_rate"]))
    replay = ReplayBuffer(int(params["replay_capacity"]))
    exploration = random.Random(seed)
    rewards, losses, epsilons = [], [], []
    validation_episodes, validation_returns = [], []
    best = -float("inf")
    global_step = 0
    episodes = int(cfg["experiment"]["train_episodes"])
    for episode in range(episodes):
        state = env.reset(seed + 7919 * episode)
        done, total, episode_losses = False, 0.0, []
        while not done:
            epsilon = epsilon_at(global_step, params)
            if exploration.random() < epsilon:
                action = exploration.randrange(env.action_dim)
            else:
                action = int(online.predict(state).argmax())
            next_state, reward, done, _ = env.step(action)
            replay.append(state, action, reward, next_state, done)
            state, total = next_state, total + reward
            global_step += 1
            if len(replay) >= max(int(params["warmup_steps"]), int(params["batch_size"])):
                episode_losses.append(optimize(
                    online, target, optimizer,
                    replay.sample(int(params["batch_size"]), exploration, device),
                    float(params["gamma"]), float(params["gradient_clip_norm"]),
                ))
            if global_step % int(params["target_update_steps"]) == 0:
                target.load_state_dict(online.state_dict())
        rewards.append(total)
        losses.append(float(np.mean(episode_losses)) if episode_losses else np.nan)
        epsilons.append(epsilon)
        number = episode + 1
        if number % int(cfg["experiment"]["validation_interval"]) == 0:
            score = validate(online, cfg)
            validation_episodes.append(number)
            validation_returns.append(score)
            if score > best:
                best = score
                save_checkpoint(output / "best_model.pt", online, seed, number, score, env.actions)
        if number % 100 == 0:
            print(f"seed={seed} episode={number}/{episodes} reward={total:.6f} epsilon={epsilon:.4f} best_validation={best:.6f}")
    save_checkpoint(output / "final_model.pt", online, seed, episodes, validation_returns[-1], env.actions)
    np.savez_compressed(output / "training_history.npz", rewards=np.asarray(rewards),
                        losses=np.asarray(losses), epsilons=np.asarray(epsilons),
                        validation_episodes=np.asarray(validation_episodes),
                        validation_returns=np.asarray(validation_returns))
    with (output / "training_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump({"algorithm": "Double DQN", "seed": seed, "episodes": episodes,
                   "device": str(device), "best_validation_return": best,
                   "training_trace_namespace": "training",
                   "validation_trace_namespace": "validation"}, handle, indent=2)
    return output / "best_model.pt"


def validate(network: QNetwork, cfg: dict) -> float:
    env = AdaptiveCPEnvironment(cfg)
    scores = []
    base = int(cfg["experiment"]["validation_seed"])
    network.eval()
    for episode in range(int(cfg["experiment"]["validation_episodes"])):
        state, done, score = env.reset(base + episode * 3571), False, 0.0
        while not done:
            state, reward, done, _ = env.step(int(network.predict(state).argmax()))
            score += reward
        scores.append(score)
    return float(np.mean(scores))

