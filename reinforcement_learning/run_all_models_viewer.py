import os
import time
from pathlib import Path
import argparse

import sys
import gymnasium as gym
from stable_baselines3 import PPO
import mujoco.viewer

# make local op3_model/src importable so `Atom-v1` is registered
PROJECT_ROOT = Path(__file__).parent
SRC_PATH = PROJECT_ROOT / "op3_model" / "src"
sys.path.insert(0, str(SRC_PATH))
import atom  # register env

ROOT = Path(__file__).parent
DEFAULT_DIR = ROOT / "saida_treino_atom" / "conjunto_melhores"

parser = argparse.ArgumentParser(description="Run all best models sequentially in MuJoCo viewer")
parser.add_argument("--models-dir", "-m", default=str(DEFAULT_DIR), help="Directory with best_model zip files")
parser.add_argument("--steps", "-s", type=int, default=1000, help="Steps to run per model")
parser.add_argument("--delay", "-d", type=float, default=0.01, help="Delay between steps (s)")
parser.add_argument("--pause", "-p", type=float, default=1.0, help="Pause between models (s)")
args = parser.parse_args()

models_dir = Path(args.models_dir)
if not models_dir.exists():
    print(f"Models dir not found: {models_dir}")
    raise SystemExit(1)

zips = sorted([p for p in models_dir.iterdir() if p.is_file() and p.suffix == ".zip"])
if not zips:
    print(f"No .zip model files found in {models_dir}")
    raise SystemExit(1)

print(f"Found {len(zips)} models. Running each for {args.steps} steps.")

for idx, z in enumerate(zips, start=1):
    print(f"\n=== [{idx}/{len(zips)}] Running model: {z.name} ===")
    model = PPO.load(str(z), device="cpu")
    env = gym.make("Atom-v1", render_mode=None)
    obs, info = env.reset()
    base_env = env.unwrapped

    try:
        with mujoco.viewer.launch_passive(base_env.model, base_env.data) as viewer:
            for step in range(args.steps):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)

                viewer.sync()

                if terminated or truncated:
                    obs, info = env.reset()

                time.sleep(args.delay)
    except Exception as e:
        print(f"Viewer error for {z.name}: {e}")
    finally:
        env.close()

    print(f"Finished model {z.name}. Pausing {args.pause}s before next.")
    time.sleep(args.pause)

print("All models executed.")
