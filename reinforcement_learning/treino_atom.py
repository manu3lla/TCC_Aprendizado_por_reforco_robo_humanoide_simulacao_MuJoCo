import os
import sys
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_PATH = os.path.join(PROJECT_ROOT, "op3_model", "src")
sys.path.insert(0, SRC_PATH)

import atom  # noqa: F401


SAVE_DIR = "saida_treino_visual_atom"
os.makedirs(SAVE_DIR, exist_ok=True)


# Aqui tentamos abrir a janela do MuJoCo
env = gym.make("Atom-v1", render_mode="none")
env = Monitor(env)


model = PPO(
    policy="MlpPolicy",
    env=env,
    verbose=1,
    device="cuda",
    learning_rate=3e-4,
    n_steps=1024,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.0,
)


model.learn(total_timesteps=2_500_000)


model_path = os.path.join(SAVE_DIR, "ppo_atom_visual")
model.save(model_path)

print(f"Modelo salvo em: {model_path}.zip")

env.close()