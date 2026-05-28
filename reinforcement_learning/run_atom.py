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

import argparse

DEFAULT_BEST = "saida_treino_atom/melhor_model/best_model.zip"
DEFAULT_FINAL = "saida_treino_atom/ppo_atom_final.zip"

parser = argparse.ArgumentParser(description="Run a trained Atom model in MuJoCo viewer")
parser.add_argument("--model", "-m", default=None, help="Path to the model zip to load")
parser.add_argument("--steps", "-s", type=int, default=3000, help="Number of simulation steps to run")
parser.add_argument("--delay", "-d", type=float, default=0.01, help="Delay between steps (seconds)")
args = parser.parse_args()

MODEL_PATH = args.model
if MODEL_PATH is None:
    if os.path.exists(DEFAULT_BEST):
        MODEL_PATH = DEFAULT_BEST
    elif os.path.exists(DEFAULT_FINAL):
        MODEL_PATH = DEFAULT_FINAL
    else:
        MODEL_PATH = DEFAULT_BEST  # fallback; user can override with --model

env = gym.make("Atom-v1", render_mode=None)
model = PPO.load(MODEL_PATH, device="cpu")

obs, info = env.reset()
base_env = env.unwrapped

with mujoco.viewer.launch_passive(base_env.model, base_env.data) as viewer:
    for step in range(args.steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        viewer.sync()

        if terminated or truncated:
            obs, info = env.reset()

        time.sleep(args.delay)

env.close()