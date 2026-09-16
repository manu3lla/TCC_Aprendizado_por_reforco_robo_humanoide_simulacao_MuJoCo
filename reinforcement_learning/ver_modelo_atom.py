import os
import sys
import time
import argparse
import hashlib
import importlib.util
import shutil
import tempfile
from pathlib import Path

MPL_CACHE_DIR = os.path.join(tempfile.gettempdir(), "matplotlib-codex-cache")
os.makedirs(MPL_CACHE_DIR, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", MPL_CACHE_DIR)

import gymnasium as gym
import mujoco
import numpy as np

from stable_baselines3 import PPO


# ============================================================
# Caminhos do projeto
# ============================================================

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_PATH = os.path.join(PROJECT_ROOT, "op3_model", "src")
PROJECT_ROOT_PATH = Path(PROJECT_ROOT)
CURRENT_MODEL_DIR = PROJECT_ROOT_PATH / "op3_model" / "src" / "model"

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
    parser.add_argument(
        "--usar-gait-inicial",
        action="store_true",
        help="Visualiza usando a mesma marcha base leve do treino residual.",
    )
    parser.add_argument(
        "--modo-gait-parado",
        action="store_true",
        help="Visualiza no modo de treino que imita o gait parado.",
    )
    parser.add_argument(
        "--modo-gait-andando",
        action="store_true",
        help="Visualiza no modo que mantém o gait e anda para frente.",
    )
    parser.add_argument(
        "--somente-gait",
        action="store_true",
        help="Mostra apenas a marcha base, sem carregar o modelo PPO.",
    )
    parser.add_argument(
        "--usar-env-snapshot",
        action="store_true",
        help="Obriga o uso do env_snapshot salvo junto do modelo.",
    )
    parser.add_argument(
        "--sem-env-snapshot",
        action="store_true",
        help="Ignora o env_snapshot e usa o env.py atual do projeto.",
    )
    parser.add_argument(
        "--env-snapshot",
        default=None,
        help="Caminho manual para um diretorio env_snapshot especifico.",
    )
    return parser.parse_args()


def find_snapshot_dir(model_path, args):
    if args.sem_env_snapshot:
        return None

    if args.env_snapshot:
        snapshot_dir = Path(args.env_snapshot).expanduser().resolve()
        if not snapshot_dir.exists():
            raise FileNotFoundError(f"env_snapshot nao encontrado: {snapshot_dir}")
        return snapshot_dir

    if model_path is None:
        if args.usar_env_snapshot:
            raise FileNotFoundError(
                "Use --env-snapshot para visualizar somente o gait com snapshot."
            )
        return None

    for parent in [model_path.parent, *model_path.parents]:
        snapshot_dir = parent / "env_snapshot"
        if snapshot_dir.exists():
            return snapshot_dir

    if args.usar_env_snapshot:
        raise FileNotFoundError(
            "Este modelo nao tem env_snapshot salvo junto dele. "
            "Sem isso, nao da para garantir a visualizacao com o env exato."
        )

    return None


def snapshot_needs_runtime_copy(snapshot_dir):
    snapshot_dir = Path(snapshot_dir)
    return not (
        (snapshot_dir / "atom" / "env.py").exists()
        and (snapshot_dir / "model" / "scene.xml").exists()
        and (snapshot_dir / "model" / "assets").exists()
    )


def prepare_snapshot_runtime(snapshot_dir):
    snapshot_dir = Path(snapshot_dir).resolve()
    digest = hashlib.sha256(str(snapshot_dir).encode("utf-8")).hexdigest()[:16]
    runtime_root = Path(tempfile.gettempdir()) / "atom_env_snapshots" / digest
    runtime_atom_dir = runtime_root / "atom"
    runtime_model_dir = runtime_root / "model"
    runtime_atom_dir.mkdir(parents=True, exist_ok=True)
    runtime_model_dir.mkdir(parents=True, exist_ok=True)

    env_source = snapshot_dir / "atom" / "env.py"
    if not env_source.exists():
        env_source = snapshot_dir / "env_usado_no_treino.py"
    if not env_source.exists():
        raise FileNotFoundError(
            f"Snapshot sem env.py/env_usado_no_treino.py: {snapshot_dir}"
        )

    shutil.copy2(env_source, runtime_atom_dir / "env.py")
    (runtime_atom_dir / "__init__.py").write_text(
        "from .env import AtomEnv\n",
        encoding="utf-8",
    )

    snapshot_model_dir = snapshot_dir / "model"
    if snapshot_model_dir.exists():
        shutil.copytree(snapshot_model_dir, runtime_model_dir, dirs_exist_ok=True)
    else:
        shutil.copytree(CURRENT_MODEL_DIR, runtime_model_dir, dirs_exist_ok=True)

    assets_dir = runtime_model_dir / "assets"
    if not assets_dir.exists():
        try:
            assets_dir.symlink_to(CURRENT_MODEL_DIR / "assets", target_is_directory=True)
        except FileExistsError:
            pass

    return runtime_atom_dir / "env.py"


def load_snapshot_env_class(snapshot_dir):
    snapshot_dir = Path(snapshot_dir).resolve()
    if snapshot_needs_runtime_copy(snapshot_dir):
        env_file = prepare_snapshot_runtime(snapshot_dir)
    else:
        env_file = snapshot_dir / "atom" / "env.py"

    module_digest = hashlib.sha256(str(env_file).encode("utf-8")).hexdigest()[:16]
    module_name = f"atom_snapshot_env_{module_digest}"
    spec = importlib.util.spec_from_file_location(module_name, env_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Nao foi possivel carregar snapshot: {env_file}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.AtomEnv


def make_visualization_env(args, snapshot_dir):
    env_kwargs = {
        "render_mode": "human",
        "debug_reward": args.debug_reward,
        "use_reference_gait": args.usar_gait_inicial,
        "gait_stand_mode": args.modo_gait_parado,
        "gait_walk_mode": args.modo_gait_andando,
    }

    if snapshot_dir:
        print(f"Usando env_snapshot: {snapshot_dir}")
        env_class = load_snapshot_env_class(snapshot_dir)
        return env_class(**env_kwargs)

    print("Aviso: sem env_snapshot para este modelo; usando env.py atual do projeto.")
    return gym.make("Atom-v1", **env_kwargs)


def smoothstep(value):
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def set_joint_offset(atom_env, joint_name, offset):
    qpos_adr = atom_env.joint_qposadr.get(joint_name)

    if qpos_adr is None:
        return

    atom_env.data.qpos[qpos_adr] = atom_env.initial_qpos[qpos_adr] + offset


def apply_kinematic_gait_preview(atom_env, step):
    cycle = (step % 84) / 84.0
    left_swing = cycle < 0.5
    step_phase = (cycle * 2.0) % 1.0
    double_support = 0.14
    swing_raw = (step_phase - double_support) / (1.0 - 2.0 * double_support)
    swing = float(np.sin(np.pi * np.clip(swing_raw, 0.0, 1.0)))
    swing_smooth = smoothstep(swing_raw)

    atom_env.data.qpos[:] = atom_env.initial_qpos
    atom_env.data.qvel[:] = 0.0

    side = 1.0 if left_swing else -1.0
    atom_env.data.qpos[1] = atom_env.initial_qpos[1] + 0.014 * side * swing_smooth
    atom_env.data.qpos[2] = atom_env.initial_qpos[2] - 0.075 + 0.012 * swing

    knee_crouch = 0.18
    ankle_crouch = -0.20
    hip_lift = -0.55
    knee_lift = 0.95
    foot_lift = -0.35
    roll_shift = 0.06 * side * swing_smooth

    set_joint_offset(atom_env, "left_knee", knee_crouch)
    set_joint_offset(atom_env, "right_knee", -knee_crouch)
    set_joint_offset(atom_env, "left_foot_pitch", ankle_crouch)
    set_joint_offset(atom_env, "right_foot_pitch", -ankle_crouch)
    set_joint_offset(atom_env, "left_leg_roll", roll_shift)
    set_joint_offset(atom_env, "right_leg_roll", roll_shift)

    if left_swing:
        set_joint_offset(atom_env, "left_leg_pitch", hip_lift * swing)
        set_joint_offset(atom_env, "left_knee", knee_crouch + knee_lift * swing)
        set_joint_offset(atom_env, "left_foot_pitch", ankle_crouch + foot_lift * swing)
    else:
        set_joint_offset(atom_env, "right_leg_pitch", -hip_lift * swing)
        set_joint_offset(atom_env, "right_knee", -(knee_crouch + knee_lift * swing))
        set_joint_offset(
            atom_env,
            "right_foot_pitch",
            -(ankle_crouch + foot_lift * swing),
        )

    mujoco.mj_forward(atom_env.model, atom_env.data)
    left_clearance = (
        atom_env.data.xpos[atom_env.left_foot_body_id][2]
        - atom_env.initial_left_foot_z
    )
    right_clearance = (
        atom_env.data.xpos[atom_env.right_foot_body_id][2]
        - atom_env.initial_right_foot_z
    )
    min_clearance = min(float(left_clearance), float(right_clearance))
    ground_margin = 0.001

    if min_clearance < ground_margin:
        atom_env.data.qpos[2] += ground_margin - min_clearance
        mujoco.mj_forward(atom_env.model, atom_env.data)


def run_gait_preview(env, args):
    atom_env = env.unwrapped
    env.reset()

    print("Previa cinemática do gait: agachado, alternando os pes no lugar.")

    for step in range(args.max_steps):
        apply_kinematic_gait_preview(atom_env, step)
        atom_env.render()

        if step % 30 == 0:
            left_clearance = (
                atom_env.data.xpos[atom_env.left_foot_body_id][2]
                - atom_env.initial_left_foot_z
            )
            right_clearance = (
                atom_env.data.xpos[atom_env.right_foot_body_id][2]
                - atom_env.initial_right_foot_z
            )
            print(
                f"gait step={step} | "
                f"left_clear={left_clearance:.3f} | "
                f"right_clear={right_clearance:.3f} | "
                f"h={atom_env.data.qpos[2]:.3f}"
            )

        if args.sleep > 0:
            time.sleep(args.sleep)


def main():
    args = parse_args()
    if args.usar_env_snapshot and args.sem_env_snapshot:
        raise SystemExit("Use apenas um: --usar-env-snapshot ou --sem-env-snapshot.")
    if args.somente_gait:
        args.usar_gait_inicial = True
    if args.modo_gait_parado:
        args.usar_gait_inicial = True
    if args.modo_gait_andando:
        args.usar_gait_inicial = True
    if args.modo_gait_parado and args.modo_gait_andando:
        raise SystemExit(
            "Use apenas um modo: --modo-gait-parado ou --modo-gait-andando."
        )

    model_path = Path(args.modelo).expanduser().resolve()
    snapshot_dir = find_snapshot_dir(None if args.somente_gait else model_path, args)

    if not args.somente_gait and not model_path.exists():
        raise FileNotFoundError(f"Modelo nao encontrado: {model_path}")

    if args.somente_gait:
        print("Visualizando somente a marcha base.")
    else:
        print(f"Carregando modelo: {model_path}")

    env = make_visualization_env(args, snapshot_dir)

    if args.somente_gait:
        try:
            run_gait_preview(env, args)
            return
        finally:
            env.close()

    model = None
    if not args.somente_gait:
        model = PPO.load(
            str(model_path),
            env=env,
            device="cpu",
        )

    obs, info = env.reset()
    total_reward = 0.0
    episode = 1

    for step in range(args.max_steps):
        if model is None:
            action = env.action_space.sample() * 0.0
        else:
            action, _ = model.predict(obs, deterministic=not args.estocastico)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)

        print(
            f"ep={episode} | "
            f"step={step} | "
            f"x={info.get('x_position', 0):.4f} | "
            f"y={info.get('y_position', 0):.4f} | "
            f"v={info.get('forward_velocity', 0):.4f} | "
            f"h={info.get('torso_height', 0):.3f} | "
            f"upright={info.get('torso_upright', 0):.3f} | "
            f"lean={info.get('forward_lean', 0):.3f} | "
            f"lean_move={info.get('moving_forward_lean_penalty', 0):.3f} | "
            f"chest_gap={info.get('chest_forward_gap', 0):.3f} | "
            f"proj={info.get('moving_body_projection_penalty', 0):.3f} | "
            f"v_target={info.get('velocity_to_target', 0):.3f} | "
            f"safe_v={info.get('safe_speed_score', 0):.2f} | "
            f"v_lat={info.get('lateral_velocity', 0):.3f} | "
            f"path={info.get('path_drift', 0):.3f} | "
            f"align={info.get('target_alignment_score', 0):.2f} | "
            f"crooked={info.get('crooked_velocity_penalty', 0):.2f} | "
            f"step_gap={info.get('step_forward_distance', 0):.3f} | "
            f"one_foot={info.get('one_foot_walk_score', 0):.2f} | "
            f"human_stride={info.get('human_stride_score', 0):.2f} | "
            f"stride_prog={info.get('stride_progress_score', 0):.2f} | "
            f"stride_up={info.get('upright_stride_factor', 0):.2f} | "
            f"support={info.get('support_stride_factor', 0):.2f} | "
            f"lean_stride={info.get('stride_forward_lean_penalty', 0):.2f} | "
            f"short_step={info.get('short_step_penalty', 0):.2f} | "
            f"lift={info.get('swing_lift_ratio', 0):.2f} | "
            f"miss_lift={info.get('missing_swing_lift_penalty', 0):.2f} | "
            f"air={info.get('double_foot_lift_penalty', 0):.2f} | "
            f"transition={info.get('step_transition_score', 0):.2f} | "
            f"fluid={info.get('fluid_step_score', 0):.2f} | "
            f"stance={info.get('stance_stability_score', 0):.2f} | "
            f"crouch={info.get('soft_crouch_score', 0):.2f} | "
            f"commit={info.get('step_commit_score', 0):.2f} | "
            f"slip={info.get('foot_slip_penalty', 0):.2f} | "
            f"cross={info.get('foot_crossing_penalty', 0):.2f} | "
            f"smooth={info.get('smooth_ctrl_cost', 0):.2f} | "
            f"gait={info.get('gait_pose_score', 0):.2f} | "
            f"gait_fwd={info.get('gait_forward_score', 0):.2f} | "
            f"gait_err={info.get('gait_pose_error', 0):.2f} | "
            f"flat={info.get('stance_foot_flat_score', 0):.2f} | "
            f"stat_xy={info.get('stationary_xy_error', 0):.3f} | "
            f"stat_v={info.get('stationary_velocity_penalty', 0):.3f} | "
            f"arm={info.get('arm_swing_score', 0):.2f} | "
            f"arm_bal={info.get('arm_balance_penalty', 0):.2f} | "
            f"arm_flail={info.get('arm_flail_penalty', 0):.2f} | "
            f"move_factor={info.get('movement_reward_factor', 0):.2f} | "
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
