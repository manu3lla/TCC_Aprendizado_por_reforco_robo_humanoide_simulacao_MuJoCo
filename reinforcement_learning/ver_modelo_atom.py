import os
import sys
import time
import argparse
import gymnasium as gym

from stable_baselines3 import PPO


# ============================================================
# Caminhos do projeto
# ============================================================

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_PATH = os.path.join(PROJECT_ROOT, "op3_model", "src")

sys.path.insert(0, SRC_PATH)

# Importa o ambiente Atom-v1
# Esse import registra o ambiente no Gymnasium
import atom  # noqa: F401


def default_model_path():
    candidates = [
        os.path.join(
            PROJECT_ROOT,
            "saida_optuna_atom",
            "treino_final_melhor",
            "melhor_modelo",
            "best_model.zip",
        ),
        os.path.join(
            PROJECT_ROOT,
            "saida_optuna_atom",
            "treino_final_melhor",
            "ppo_atom_optuna_final.zip",
        ),
        os.path.join(PROJECT_ROOT, "saida_treino_visual_atom", "ppo_atom_visual.zip"),
    ]

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    return candidates[0]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualiza um modelo PPO treinado no ambiente Atom-v1."
    )
    parser.add_argument(
        "--modelo",
        default=default_model_path(),
        help="Caminho do arquivo .zip do modelo PPO.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=10_000,
        help="Numero maximo de passos de simulacao.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.01,
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
    model_path = os.path.abspath(args.modelo)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modelo nao encontrado: {model_path}")

    print(f"Carregando modelo: {model_path}")

    env = gym.make(
        "Atom-v1",
        render_mode="human",
        debug_reward=args.debug_reward,
    )

    model = PPO.load(
        model_path,
        env=env,
        device="cpu",
    )

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
