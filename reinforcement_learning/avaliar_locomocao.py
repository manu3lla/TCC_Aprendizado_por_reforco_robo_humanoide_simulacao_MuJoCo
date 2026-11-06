import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "op3_model" / "src"))
import atom  # noqa: E402,F401


def restore_env_args(args, model_path):
    config_path = Path(model_path).resolve().parent / "env_config.json"
    if not config_path.exists():
        return
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if args.velocidade_alvo is None:
        args.velocidade_alvo = config.get("desired_walk_speed")
    if args.periodo_gait is None:
        args.periodo_gait = config.get("gait_period_steps")
    if not getattr(args, "modo_gait_parado", False):
        args.modo_gait_rapido |= bool(config.get("fast_walk_mode"))
        args.modo_gait_andando |= bool(config.get("gait_walk_mode"))


def save_env_config(env, directory):
    base = env.unwrapped
    config = {
        "fast_walk_mode": base.fast_walk_mode,
        "gait_walk_mode": base.gait_walk_mode,
        "desired_walk_speed": base.desired_walk_speed,
        "gait_period_steps": base.gait_period_steps,
    }
    (Path(directory) / "env_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )


def evaluate_locomotion(model, env, n_episodes=5, seed=42):
    if n_episodes < 1:
        raise ValueError("A avaliacao precisa de pelo menos um episodio.")
    episodes = []
    for episode in range(n_episodes):
        obs, initial = env.reset(seed=seed + episode)
        initial_distance = float(initial["distance_to_target"])
        total_reward = 0.0
        max_drift = 0.0
        min_upright = 1.0
        for step in range(env.unwrapped.max_episode_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            max_drift = max(max_drift, float(info["path_drift"]))
            min_upright = min(min_upright, float(info["torso_upright"]))
            if terminated or truncated:
                break
        seconds = float(info["elapsed_simulation_seconds"])
        progress = initial_distance - float(info["distance_to_target"])
        episodes.append({
            "seed": seed + episode,
            "success": bool(info["is_success"]),
            "fallen": bool(info["fallen"]),
            "seconds": seconds,
            "progress_m": progress,
            "progress_speed_m_s": progress / seconds,
            "max_path_drift_m": max_drift,
            "min_upright": min_upright,
            "reward": total_reward,
            "length": step + 1,
        })
    return {
        "success_rate": float(np.mean([ep["success"] for ep in episodes])),
        "fall_rate": float(np.mean([ep["fallen"] for ep in episodes])),
        "mean_progress_speed_m_s": float(np.mean([
            ep["progress_speed_m_s"] for ep in episodes
        ])),
        "mean_seconds": float(np.mean([ep["seconds"] for ep in episodes])),
        "max_path_drift_m": max(ep["max_path_drift_m"] for ep in episodes),
        "mean_reward": float(np.mean([ep["reward"] for ep in episodes])),
        "episodes": episodes,
    }


class SafeSpeedEvalCallback(BaseCallback):
    """Keep the fastest checkpoint that reaches the target without falling."""

    def __init__(self, eval_env, best_dir, log_dir, eval_freq, n_episodes, seed):
        super().__init__()
        if eval_freq < 1 or n_episodes < 1:
            raise ValueError("Frequencia e numero de avaliacoes devem ser positivos.")
        self.eval_env = eval_env
        self.best_dir = Path(best_dir)
        self.log_dir = Path(log_dir)
        self.eval_freq = eval_freq
        self.n_episodes = n_episodes
        self.seed = seed
        self.best_speed = -np.inf
        self.evaluations = []

    def _on_training_start(self):
        # Evaluate the starting checkpoint so fine-tuning retains a safe fallback.
        self._evaluate()

    def _on_step(self):
        if self.n_calls % self.eval_freq == 0:
            self._evaluate()
        return True

    def _on_training_end(self):
        if self.evaluations[-1]["timesteps"] != self.num_timesteps:
            self._evaluate()

    def _evaluate(self):
        result = evaluate_locomotion(
            self.model, self.eval_env, self.n_episodes, self.seed
        )
        result["timesteps"] = self.num_timesteps
        eligible = result["success_rate"] == 1.0 and result["fall_rate"] == 0.0
        result["selected"] = bool(
            eligible and result["mean_progress_speed_m_s"] > self.best_speed
        )
        if result["selected"]:
            self.best_dir.mkdir(parents=True, exist_ok=True)
            self.model.save(self.best_dir / "best_model")
            if self.eval_env is not None:
                save_env_config(self.eval_env, self.best_dir)
            self.best_speed = result["mean_progress_speed_m_s"]
        self.evaluations.append(result)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        (self.log_dir / "speed_evaluations.json").write_text(
            json.dumps(self.evaluations, indent=2), encoding="utf-8"
        )
        np.savez(
            self.log_dir / "evaluations.npz",
            timesteps=[ev["timesteps"] for ev in self.evaluations],
            results=[[ep["reward"] for ep in ev["episodes"]] for ev in self.evaluations],
            ep_lengths=[[ep["length"] for ep in ev["episodes"]] for ev in self.evaluations],
            successes=[[ep["success"] for ep in ev["episodes"]] for ev in self.evaluations],
        )
        for key in ("success_rate", "fall_rate", "mean_progress_speed_m_s", "mean_reward"):
            self.logger.record(f"eval/{key}", result[key])
        print(
            f"[Velocidade] steps={self.num_timesteps} "
            f"alvo={result['success_rate']:.0%} quedas={result['fall_rate']:.0%} "
            f"progresso={result['mean_progress_speed_m_s']:.3f} m/s "
            f"tempo={result['mean_seconds']:.2f}s melhor={result['selected']}",
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser(description="Mede velocidade ate o alvo e quedas sem janela.")
    parser.add_argument("--modelo", required=True)
    parser.add_argument("--modo-gait-andando", action="store_true")
    parser.add_argument("--modo-gait-rapido", action="store_true")
    parser.add_argument("--velocidade-alvo", type=float, default=None)
    parser.add_argument("--periodo-gait", type=int, default=None)
    parser.add_argument("--n-episodios", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--saida", type=Path)
    args = parser.parse_args()
    restore_env_args(args, args.modelo)
    import torch
    torch.set_num_threads(1)
    with contextlib.redirect_stdout(io.StringIO()):
        env = gym.make(
            "Atom-v1", gait_walk_mode=args.modo_gait_andando,
            fast_walk_mode=args.modo_gait_rapido,
            desired_walk_speed=args.velocidade_alvo,
            gait_period_steps=args.periodo_gait,
        )
    try:
        model = PPO.load(args.modelo, env=env, device="cpu")
        result = evaluate_locomotion(model, env, args.n_episodios, args.seed)
        result["model"] = str(Path(args.modelo).resolve())
        result["environment"] = "current env.py; deterministic policy and reset"
        result["fast_walk_mode"] = args.modo_gait_rapido
        result["desired_walk_speed"] = env.unwrapped.desired_walk_speed
        result["gait_period_steps"] = env.unwrapped.gait_period_steps
        output = json.dumps(result, indent=2)
        if args.saida:
            args.saida.parent.mkdir(parents=True, exist_ok=True)
            args.saida.write_text(output + "\n", encoding="utf-8")
        print(output)
    finally:
        env.close()


if __name__ == "__main__":
    main()
