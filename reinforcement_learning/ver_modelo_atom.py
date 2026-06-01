import os
import sys
import time
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


# ============================================================
# Caminho do modelo treinado
# ============================================================

MODEL_PATH = "saida_treino_visual_atom/ppo_atom_visual.zip"


# ============================================================
# Cria ambiente com visualização
# ============================================================

env = gym.make("Atom-v1", render_mode="human")


# ============================================================
# Carrega modelo
# ============================================================

model = PPO.load(MODEL_PATH)


# ============================================================
# Roda o modelo treinado
# ============================================================

obs, info = env.reset()

for step in range(10_000):
    action, _ = model.predict(obs, deterministic=True)

    obs, reward, terminated, truncated, info = env.step(action)

    print(
        f"step={step} | "
        f"x={info.get('x_position', 0):.4f} | "
        f"dx={info.get('delta_x', 0):.6f} | "
        f"h={info.get('torso_height', 0):.3f} | "
        f"upright={info.get('upright', 0):.3f} | "
        f"reward={reward:.4f}"
    )

    # Deixa mais fácil de enxergar.
    # Se ficar lento demais, diminui ou remove.
    time.sleep(0.01)

    if terminated or truncated:
        print("Episódio terminou. Resetando ambiente...")
        obs, info = env.reset()

env.close()