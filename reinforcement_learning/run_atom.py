import os
import sys
import time
import gymnasium as gym
import mujoco.viewer

# faz o Python achar op3_model/src
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_PATH = os.path.join(PROJECT_ROOT, "op3_model", "src")
sys.path.insert(0, SRC_PATH)

import atom

from stable_baselines3 import PPO

MODEL_PATH = "saida_treino_atom/ppo_atom_teste.zip"

env = gym.make(
    "Atom-v1",
    render_mode=None
)

model = PPO.load(MODEL_PATH, device="cpu")

obs, info = env.reset()
base_env = env.unwrapped

with mujoco.viewer.launch_passive(base_env.model, base_env.data) as viewer:
    for step in range(3000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        viewer.sync()

        if terminated or truncated:
            obs, info = env.reset()

        time.sleep(0.01)

env.close()