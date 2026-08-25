import time
import argparse
from pathlib import Path

import gymnasium as gym

from stable_baselines3 import PPO

from atom_paths import default_model_path, register_atom_env


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualiza um modelo PPO treinado no ambiente Atom-v1."
    )
    parser.add_argument(
        "--modelo",
        "--model",
        "-m",
        default=default_model_path(),
        type=Path,
        dest="modelo",
        help="Caminho do arquivo .zip do modelo PPO.",
    )
    parser.add_argument(
        "--max-steps",
        "--steps",
        "-s",
        type=int,
        default=10_000,
        dest="max_steps",
        help="Numero maximo de passos de simulacao.",
    )
    parser.add_argument(
        "--sleep",
        "--delay",
        "-d",
        type=float,
        default=0.01,
        dest="sleep",
        help="Pausa entre passos para facilitar a visualizacao.",
    )
    parser.add_argument(
        "--estocastico",
        action="store_true",
        help="Usa a politica estocastica em vez da acao deterministica.",
    )
    parser.add_argument(
        "--debug-reward",
        action="store_true",
        help="Mostra detalhes da recompensa a cada 50 passos.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    register_atom_env()

    model_path = args.modelo.expanduser().resolve()

    if not model_path.exists():
        raise FileNotFoundError(f"Modelo nao encontrado: {model_path}")

    print(f"Carregando modelo: {model_path}")

    env = gym.make(
        "Atom-v1",
        render_mode="human",
        debug_reward=args.debug_reward,
    )

    model = PPO.load(str(model_path), env=env, device="cpu")

    obs, info = env.reset()
    total_reward = 0.0
    episode = 1

    for step in range(args.max_steps):
        action, _ = model.predict(obs, deterministic=not args.estocastico)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)

        print(
            f"ep={episode} | "
            f"step={step} | "
            f"x={info.get('x_position', 0):.4f} | "
            f"v={info.get('forward_velocity', 0):.4f} | "
            f"h={info.get('torso_height', 0):.3f} | "
            f"upright={info.get('upright', 0):.3f} | "
            f"left_contact={info.get('left_foot_contact', False)} | "
            f"right_contact={info.get('right_foot_contact', False)} | "
            f"reward={reward:.4f} | "
            f"total={total_reward:.2f}"
        )

        if args.sleep > 0:
            time.sleep(args.sleep)

        if terminated or truncated:
            print(
                f"Episodio {episode} terminou com reward total: "
                f"{total_reward:.2f}. Resetando..."
            )
            obs, info = env.reset()
            total_reward = 0.0
            episode += 1

    env.close()


if __name__ == "__main__":
    main()
