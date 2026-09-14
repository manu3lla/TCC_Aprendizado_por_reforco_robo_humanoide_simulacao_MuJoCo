import argparse
import contextlib
import json
import os
import sys
from pathlib import Path

try:
    import gymnasium as gym
except ImportError as exc:
    raise SystemExit(
        "Gymnasium nao esta instalado neste Python. Ative o ambiente do projeto "
        "ou instale as dependencias de treino."
    ) from exc

try:
    import optuna
except ImportError:
    optuna = None

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import EvalCallback
    from stable_baselines3.common.evaluation import evaluate_policy
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.utils import set_random_seed
except ImportError as exc:
    raise SystemExit(
        "stable-baselines3 nao esta instalado neste Python. Ative o ambiente "
        "do projeto ou instale as dependencias de treino."
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "op3_model" / "src"
sys.path.insert(0, str(SRC_PATH))

import atom  # noqa: E402,F401

from model_registry import DEFAULT_REGISTRY_DIR, register_training_run


DEFAULT_SAVE_DIR = PROJECT_ROOT / "saida_optuna_atom"
ROLLOUT_BATCH_CHOICES = [
    "256_32",
    "256_64",
    "512_32",
    "512_64",
    "512_128",
    "1024_64",
    "1024_128",
    "1024_256",
    "2048_64",
    "2048_128",
    "2048_256",
]


class QuietEnvWrapper(gym.Wrapper):
    """Oculta prints de debug do ambiente durante a busca do Optuna."""

    def __init__(self, env):
        super().__init__(env)
        self._devnull = open(os.devnull, "w", encoding="utf-8")

    def reset(self, **kwargs):
        with contextlib.redirect_stdout(self._devnull):
            return self.env.reset(**kwargs)

    def step(self, action):
        with contextlib.redirect_stdout(self._devnull):
            return self.env.step(action)

    def close(self):
        try:
            return super().close()
        finally:
            self._devnull.close()


class TrialEvalCallback(EvalCallback):
    """Envia recompensas intermediarias para o Optuna e permite pruning."""

    def __init__(self, eval_env, trial, n_eval_episodes, eval_freq):
        super().__init__(
            eval_env=eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            deterministic=True,
            render=False,
            verbose=0,
        )
        self.trial = trial
        self.eval_idx = 0
        self.is_pruned = False

    def _on_step(self):
        continue_training = super()._on_step()

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            self.eval_idx += 1
            print(
                "[Optuna]",
                f"trial={self.trial.number}",
                f"avaliacao={self.eval_idx}",
                f"timesteps={self.num_timesteps}",
                f"mean_reward={self.last_mean_reward:.4f}",
                flush=True,
            )
            self.trial.report(self.last_mean_reward, self.eval_idx)

            if self.trial.should_prune():
                self.is_pruned = True
                print(
                    "[Optuna]",
                    f"trial={self.trial.number}",
                    "interrompido por pruning",
                    flush=True,
                )
                return False

        return continue_training


def parse_rollout_batch(value):
    n_steps, batch_size = value.split("_")
    return int(n_steps), int(batch_size)


def make_env(seed, monitor_prefix=None, quiet=True):
    if quiet:
        with open(os.devnull, "w", encoding="utf-8") as devnull:
            with contextlib.redirect_stdout(devnull):
                env = gym.make("Atom-v1", render_mode=None)
    else:
        env = gym.make("Atom-v1", render_mode=None)

    if quiet:
        env = QuietEnvWrapper(env)

    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    filename = str(monitor_prefix) if monitor_prefix is not None else None
    return Monitor(env, filename=filename)


def sample_ppo_params(trial):
    rollout_batch = trial.suggest_categorical(
        "rollout_batch",
        ROLLOUT_BATCH_CHOICES,
    )
    n_steps, batch_size = parse_rollout_batch(rollout_batch)

    return {
        "learning_rate": trial.suggest_float(
            "learning_rate",
            1e-5,
            1e-3,
            log=True,
        ),
        "n_steps": n_steps,
        "batch_size": batch_size,
        "n_epochs": trial.suggest_categorical("n_epochs", [5, 10, 15, 20]),
        "gamma": trial.suggest_categorical(
            "gamma",
            [0.97, 0.98, 0.99, 0.995, 0.999],
        ),
        "gae_lambda": trial.suggest_categorical(
            "gae_lambda",
            [0.90, 0.92, 0.95, 0.98, 1.00],
        ),
        "clip_range": trial.suggest_float("clip_range", 0.10, 0.30),
        "ent_coef": trial.suggest_categorical(
            "ent_coef",
            [0.0, 1e-4, 1e-3, 5e-3, 1e-2],
        ),
        "vf_coef": trial.suggest_float("vf_coef", 0.30, 1.00),
        "max_grad_norm": trial.suggest_float("max_grad_norm", 0.30, 1.00),
    }


def params_from_best_trial(best_params):
    n_steps, batch_size = parse_rollout_batch(best_params["rollout_batch"])

    return {
        "learning_rate": best_params["learning_rate"],
        "n_steps": n_steps,
        "batch_size": batch_size,
        "n_epochs": best_params["n_epochs"],
        "gamma": best_params["gamma"],
        "gae_lambda": best_params["gae_lambda"],
        "clip_range": best_params["clip_range"],
        "ent_coef": best_params["ent_coef"],
        "vf_coef": best_params["vf_coef"],
        "max_grad_norm": best_params["max_grad_norm"],
    }


def objective(trial, args, save_dir):
    trial_dir = save_dir / f"trial_{trial.number:04d}"
    log_dir = trial_dir / "logs"
    model_dir = trial_dir / "models"
    log_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    seed = args.seed + trial.number
    set_random_seed(seed)

    ppo_params = sample_ppo_params(trial)
    trial.set_user_attr("ppo_params", ppo_params)
    print(
        "\n[Optuna]",
        f"iniciando trial={trial.number}",
        f"seed={seed}",
        f"params={ppo_params}",
        flush=True,
    )

    train_env = make_env(
        seed=seed,
        monitor_prefix=log_dir / "train",
        quiet=not args.mostrar_debug_env,
    )
    eval_env = make_env(
        seed=seed + 10_000,
        monitor_prefix=log_dir / "eval",
        quiet=not args.mostrar_debug_env,
    )

    try:
        model = PPO(
            policy="MlpPolicy",
            env=train_env,
            verbose=args.verbose_ppo,
            device=args.device,
            seed=seed,
            tensorboard_log=str(log_dir),
            **ppo_params,
        )

        eval_callback = TrialEvalCallback(
            eval_env=eval_env,
            trial=trial,
            n_eval_episodes=args.n_eval_episodes,
            eval_freq=args.eval_freq,
        )

        model.learn(
            total_timesteps=args.trial_timesteps,
            callback=eval_callback,
            tb_log_name=f"trial_{trial.number:04d}",
            progress_bar=args.progress_bar,
        )

        if eval_callback.is_pruned:
            raise optuna.TrialPruned()

        mean_reward, std_reward = evaluate_policy(
            model,
            eval_env,
            n_eval_episodes=args.n_eval_episodes,
            deterministic=True,
        )

        trial.set_user_attr("mean_reward", float(mean_reward))
        trial.set_user_attr("std_reward", float(std_reward))

        model.save(model_dir / "final_model")
        print(
            "[Optuna]",
            f"trial={trial.number}",
            f"finalizado mean_reward={mean_reward:.4f}",
            f"std_reward={std_reward:.4f}",
            flush=True,
        )
        return float(mean_reward)

    finally:
        train_env.close()
        eval_env.close()


def save_best_params(study, save_dir):
    best_ppo_params = params_from_best_trial(study.best_params)
    output = {
        "best_trial": study.best_trial.number,
        "best_value_mean_reward": study.best_value,
        "optuna_params": study.best_params,
        "ppo_params": best_ppo_params,
    }

    output_path = save_dir / "best_params.json"
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(output, file, indent=2)

    return output_path, best_ppo_params


def load_best_ppo_params(save_dir):
    best_params_path = save_dir / "best_params.json"
    if not best_params_path.exists():
        raise SystemExit(
            f"Arquivo nao encontrado: {best_params_path}. Rode a busca primeiro."
        )

    with best_params_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    return data["ppo_params"]


def train_final_model(args, save_dir, ppo_params):
    final_dir = save_dir / args.final_run_name
    log_dir = final_dir / "logs"
    best_dir = final_dir / "melhor_modelo"
    final_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    best_dir.mkdir(parents=True, exist_ok=True)

    train_env = make_env(
        seed=args.seed + 99_000,
        monitor_prefix=log_dir / "train",
        quiet=not args.mostrar_debug_env,
    )
    eval_env = make_env(
        seed=args.seed + 100_000,
        monitor_prefix=log_dir / "eval",
        quiet=not args.mostrar_debug_env,
    )

    try:
        continue_from = None
        if args.continuar_de:
            continue_from = Path(args.continuar_de).expanduser().resolve()
            if not continue_from.exists():
                raise FileNotFoundError(
                    f"Modelo para continuar treino nao encontrado: {continue_from}"
                )

            print(f"Continuando treino a partir de: {continue_from}")
            model = PPO.load(
                str(continue_from),
                env=train_env,
                device=args.device,
                tensorboard_log=str(log_dir),
            )
            model.verbose = 1
        else:
            model = PPO(
                policy="MlpPolicy",
                env=train_env,
                verbose=1,
                device=args.device,
                seed=args.seed + 99_000,
                tensorboard_log=str(log_dir),
                **ppo_params,
            )

        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=str(best_dir),
            log_path=str(log_dir),
            eval_freq=args.eval_freq,
            n_eval_episodes=args.n_eval_episodes,
            deterministic=True,
            render=False,
        )

        model.learn(
            total_timesteps=args.final_timesteps,
            callback=eval_callback,
            tb_log_name=args.final_run_name,
            progress_bar=args.progress_bar,
            reset_num_timesteps=continue_from is None,
        )

        final_model_path = final_dir / "ppo_atom_optuna_final"
        model.save(final_model_path)
        print(f"Modelo final salvo em: {final_model_path}.zip")
        print(f"Melhor modelo da avaliacao salvo em: {best_dir / 'best_model.zip'}")

        if not args.sem_registrar_modelo:
            result = register_training_run(
                run_name=args.final_run_name,
                source_run_dir=final_dir,
                best_model_path=best_dir / "best_model.zip",
                final_model_path=Path(f"{final_model_path}.zip"),
                log_dir=log_dir,
                ppo_params=ppo_params,
                training_args={
                    "device": args.device,
                    "seed": args.seed,
                    "final_timesteps": args.final_timesteps,
                    "eval_freq": args.eval_freq,
                    "n_eval_episodes": args.n_eval_episodes,
                    "study_name": args.study_name,
                    "save_dir": save_dir,
                    "final_run_name": args.final_run_name,
                    "continuar_de": str(continue_from) if continue_from else None,
                },
                registry_dir=args.model_registry_dir,
                best_params_path=save_dir / "best_params.json",
            )
            print(f"Modelo registrado em: {result['entry_dir']}")
            print(f"Documentacao: {result['readme_path']}")
            view_command = result["metadata"]["commands"].get("view_best")
            if view_command:
                print(f"Ver no MuJoCo: {view_command}")

    finally:
        train_env.close()
        eval_env.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Busca hiperparametros do PPO para Atom-v1 usando Optuna."
    )
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--trial-timesteps", type=int, default=100_000)
    parser.add_argument("--eval-freq", type=int, default=10_000)
    parser.add_argument("--n-eval-episodes", type=int, default=5)
    parser.add_argument("--final-timesteps", type=int, default=2_500_000)
    parser.add_argument("--final-run-name", default="treino_final_melhor")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--study-name", default="ppo_atom_optuna")
    parser.add_argument("--save-dir", default=str(DEFAULT_SAVE_DIR))
    parser.add_argument("--storage", default=None)
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--verbose-ppo", type=int, default=0)
    parser.add_argument("--sem-pruning", action="store_true")
    parser.add_argument("--progress-bar", action="store_true")
    parser.add_argument("--mostrar-debug-env", action="store_true")
    parser.add_argument("--treinar-melhor", action="store_true")
    parser.add_argument("--somente-treinar-melhor", action="store_true")
    parser.add_argument("--model-registry-dir", default=str(DEFAULT_REGISTRY_DIR))
    parser.add_argument("--sem-registrar-modelo", action="store_true")
    parser.add_argument(
        "--continuar-de",
        default=None,
        help="Caminho de um modelo .zip existente para continuar o treino.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    save_dir = Path(args.save_dir).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    if args.somente_treinar_melhor:
        best_ppo_params = load_best_ppo_params(save_dir)
        train_final_model(args, save_dir, best_ppo_params)
        return

    if optuna is None:
        raise SystemExit(
            "Optuna nao esta instalado. Instale com: pip install optuna"
        )

    storage = args.storage or f"sqlite:///{save_dir / 'optuna.db'}"
    pruner = (
        optuna.pruners.NopPruner()
        if args.sem_pruning
        else optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=2,
        )
    )

    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=args.seed),
        pruner=pruner,
    )

    study.optimize(
        lambda trial: objective(trial, args, save_dir),
        n_trials=args.n_trials,
        timeout=args.timeout,
        n_jobs=args.n_jobs,
        gc_after_trial=True,
    )

    best_params_path, best_ppo_params = save_best_params(study, save_dir)

    print("\nBusca finalizada.")
    print(f"Melhor trial: {study.best_trial.number}")
    print(f"Melhor recompensa media: {study.best_value:.4f}")
    print(f"Melhores parametros salvos em: {best_params_path}")
    print(json.dumps(best_ppo_params, indent=2))

    if args.treinar_melhor:
        train_final_model(args, save_dir, best_ppo_params)


if __name__ == "__main__":
    main()
