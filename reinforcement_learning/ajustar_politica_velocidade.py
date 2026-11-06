import argparse
import contextlib
import io
import itertools
import json
import math
from pathlib import Path

import gymnasium as gym
import torch
from stable_baselines3 import PPO

from avaliar_locomocao import evaluate_locomotion, restore_env_args, save_env_config
from model_registry import sha256sum


def apply_action_adjustment(model, weight, bias, actuator_index, gain, hip, knee, ankle):
    with torch.no_grad():
        model.policy.action_net.weight.copy_(weight * gain)
        model.policy.action_net.bias.copy_(bias * gain)
        for joint, offset in (("leg_pitch", hip), ("knee", knee), ("foot_pitch", ankle)):
            for side, sign in (("left", 1.0), ("right", -1.0)):
                model.policy.action_net.bias[actuator_index[f"{side}_{joint}"]] += sign * offset


def main():
    parser = argparse.ArgumentParser(description="Busca pequenos ajustes na politica PPO por velocidade segura.")
    parser.add_argument("--modelo", required=True, type=Path)
    parser.add_argument("--saida", required=True, type=Path)
    parser.add_argument("--velocidade-alvo", type=float, default=0.35)
    parser.add_argument("--gains", type=float, nargs="+", default=[0.85, 1.0, 1.15])
    parser.add_argument("--hip-offsets", type=float, nargs="+", default=[-0.2, -0.1, 0.0, 0.1, 0.2])
    parser.add_argument("--knee-offsets", type=float, nargs="+", default=[0.0])
    parser.add_argument("--ankle-offsets", type=float, nargs="+", default=[0.0])
    parser.add_argument("--periodos-gait", type=int, nargs="+", default=None)
    args = parser.parse_args()
    args.periodo_gait = None
    args.modo_gait_rapido = True
    args.modo_gait_andando = True
    restore_env_args(args, args.modelo)
    args.periodos_gait = args.periodos_gait or [args.periodo_gait or 76]
    if any(period < 2 for period in args.periodos_gait):
        parser.error("Os periodos da marcha devem ser inteiros >= 2.")
    if any(not math.isfinite(gain) or gain <= 0 for gain in args.gains):
        parser.error("Os ganhos devem ser positivos e finitos.")
    if any(not math.isfinite(value) for value in (
        *args.hip_offsets, *args.knee_offsets, *args.ankle_offsets
    )):
        parser.error("Os ajustes devem ser finitos.")
    torch.set_num_threads(1)
    if args.saida.exists():
        raise SystemExit("Escolha um diretorio de saida novo.")
    args.saida.mkdir(parents=True)
    with contextlib.redirect_stdout(io.StringIO()):
        env = gym.make(
            "Atom-v1", fast_walk_mode=True, desired_walk_speed=args.velocidade_alvo,
            gait_period_steps=args.periodo_gait,
        )
        model = PPO.load(args.modelo, env=env, device="cpu")
    weight = model.policy.action_net.weight.detach().clone()
    bias = model.policy.action_net.bias.detach().clone()
    original_hash = sha256sum(args.modelo)
    best_speed = float("-inf")
    history = []
    candidates = [(1.0, 0.0, 0.0, 0.0, env.unwrapped.gait_period_steps), *itertools.product(
        args.gains, args.hip_offsets, args.knee_offsets, args.ankle_offsets, args.periodos_gait
    )]
    try:
        for index, (gain, hip, knee, ankle, period) in enumerate(dict.fromkeys(candidates)):
            env.unwrapped.gait_period_steps = period
            apply_action_adjustment(
                model, weight, bias, env.unwrapped.actuator_index, gain, hip, knee, ankle
            )
            result = evaluate_locomotion(model, env, n_episodes=1)
            result["adjustment"] = dict(gain=gain, hip=hip, knee=knee, ankle=ankle, period=period)
            result["index"] = index
            eligible = result["success_rate"] == 1.0 and result["fall_rate"] == 0.0
            result["selected"] = eligible and result["mean_progress_speed_m_s"] > best_speed
            if result["selected"]:
                best_speed = result["mean_progress_speed_m_s"]
                model.save(args.saida / "best_model")
                save_env_config(env, args.saida)
                (args.saida / "best_measurement.json").write_text(
                    json.dumps(result, indent=2), encoding="utf-8"
                )
            history.append(result)
            (args.saida / "search.json").write_text(json.dumps({
                "method": "grid search of symmetric actor output adjustments",
                "source": str(args.modelo.resolve()), "source_sha256": original_hash,
                "desired_walk_speed": args.velocidade_alvo,
                "deterministic_reset": True, "evaluations": history,
            }, indent=2), encoding="utf-8")
            print(
                f"[Busca] {index} ajuste={result['adjustment']} "
                f"alvo={result['success_rate']:.0%} quedas={result['fall_rate']:.0%} "
                f"v={result['mean_progress_speed_m_s']:.4f}m/s "
                f"tempo={result['mean_seconds']:.2f}s melhor={result['selected']}",
                flush=True,
            )
    finally:
        env.close()


if __name__ == "__main__":
    main()
