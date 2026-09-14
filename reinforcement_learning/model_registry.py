import csv
import hashlib
import json
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path


RL_ROOT = Path(__file__).resolve().parent
REPO_ROOT = RL_ROOT.parent
DEFAULT_REGISTRY_DIR = RL_ROOT / "melhores_modelos"


def slugify(value):
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9_.-]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "modelo"


def to_repo_path(path):
    path = Path(path).resolve()
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def from_repo_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def json_safe(value):
    if isinstance(value, Path):
        return to_repo_path(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def load_json(path):
    path = Path(path)
    if not path.exists():
        return None

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def sha256sum(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_model(src, dst):
    src = Path(src)
    if not src.exists():
        return None

    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)

    return {
        "path": to_repo_path(dst),
        "source_path": to_repo_path(src),
        "size_bytes": dst.stat().st_size,
        "sha256": sha256sum(dst),
    }


def read_monitor_summary(path):
    path = Path(path)
    if not path.exists():
        return None

    rows = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.startswith("#") or not line.strip():
                continue
            rows.append(line)

    if len(rows) <= 1:
        return None

    reader = csv.DictReader(rows)
    rewards = []
    lengths = []
    times = []
    for row in reader:
        try:
            rewards.append(float(row["r"]))
            lengths.append(int(float(row["l"])))
            times.append(float(row["t"]))
        except (KeyError, TypeError, ValueError):
            continue

    if not rewards:
        return None

    tail = rewards[-10:]
    return {
        "episodes": len(rewards),
        "last_reward": rewards[-1],
        "best_episode_reward": max(rewards),
        "mean_last_10_reward": sum(tail) / len(tail),
        "last_episode_length": lengths[-1] if lengths else None,
        "elapsed_seconds": times[-1] if times else None,
    }


def read_monitor_series(path):
    path = Path(path)
    if not path.exists():
        return None

    rows = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.startswith("#") or not line.strip():
                continue
            rows.append(line)

    if len(rows) <= 1:
        return None

    reader = csv.DictReader(rows)
    rewards = []
    lengths = []
    times = []
    for row in reader:
        try:
            rewards.append(float(row["r"]))
            lengths.append(float(row["l"]))
            times.append(float(row["t"]))
        except (KeyError, TypeError, ValueError):
            continue

    if not rewards:
        return None

    return {
        "episodes": list(range(1, len(rewards) + 1)),
        "rewards": rewards,
        "lengths": lengths,
        "times": times,
    }


def moving_average(values, window):
    if not values:
        return []

    window = max(1, min(window, len(values)))
    averaged = []
    for idx in range(len(values)):
        start = max(0, idx - window + 1)
        averaged.append(sum(values[start : idx + 1]) / (idx - start + 1))
    return averaged


def read_eval_summary(log_dir):
    log_dir = Path(log_dir)
    npz_path = log_dir / "evaluations.npz"
    summary = {
        "train_monitor": read_monitor_summary(log_dir / "train.monitor.csv"),
        "eval_monitor": read_monitor_summary(log_dir / "eval.monitor.csv"),
    }

    if not npz_path.exists():
        return summary

    try:
        import numpy as np

        data = np.load(npz_path)
        results = data["results"]
        timesteps = data["timesteps"]
        means = results.mean(axis=1)
        best_idx = int(means.argmax())
        summary["evaluations_npz"] = {
            "path": to_repo_path(npz_path),
            "num_evaluations": int(len(means)),
            "best_mean_reward": float(means[best_idx]),
            "best_timestep": int(timesteps[best_idx]),
            "last_mean_reward": float(means[-1]),
            "last_timestep": int(timesteps[-1]),
        }
    except Exception as exc:  # noqa: BLE001
        summary["evaluations_npz_error"] = str(exc)

    return summary


def setup_matplotlib():
    cache_dir = Path("/tmp") / "matplotlib-codex-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def save_training_reward_plot(log_dir, plots_dir):
    series = read_monitor_series(Path(log_dir) / "train.monitor.csv")
    if not series:
        return None

    plt = setup_matplotlib()
    output_path = Path(plots_dir) / "recompensa_treino.png"

    rewards = series["rewards"]
    episodes = series["episodes"]
    window = 100 if len(rewards) >= 100 else max(1, len(rewards) // 5)
    avg_rewards = moving_average(rewards, window)

    plt.figure(figsize=(11, 5))
    plt.plot(episodes, rewards, alpha=0.25, linewidth=0.8, label="Recompensa por episodio")
    plt.plot(episodes, avg_rewards, linewidth=2.0, label=f"Media movel ({window})")
    plt.xlabel("Episodio")
    plt.ylabel("Recompensa")
    plt.title("Recompensa durante o treinamento")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=140)
    plt.close()

    return output_path


def save_evaluation_plot(log_dir, plots_dir):
    npz_path = Path(log_dir) / "evaluations.npz"
    if not npz_path.exists():
        return None

    try:
        import numpy as np

        data = np.load(npz_path)
        timesteps = data["timesteps"]
        results = data["results"]
        means = results.mean(axis=1)
        stds = results.std(axis=1)
    except Exception:  # noqa: BLE001
        return None

    plt = setup_matplotlib()
    output_path = Path(plots_dir) / "convergencia_avaliacao.png"

    plt.figure(figsize=(11, 5))
    plt.plot(timesteps, means, marker="o", linewidth=2.0, label="Recompensa media")
    plt.fill_between(timesteps, means - stds, means + stds, alpha=0.2, label="Desvio padrao")
    best_idx = int(means.argmax())
    plt.scatter([timesteps[best_idx]], [means[best_idx]], color="tab:red", zorder=3, label="Melhor avaliacao")
    plt.xlabel("Timesteps")
    plt.ylabel("Recompensa media")
    plt.title("Convergencia nas avaliacoes")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=140)
    plt.close()

    return output_path


def save_eval_monitor_plot(log_dir, plots_dir):
    series = read_monitor_series(Path(log_dir) / "eval.monitor.csv")
    if not series:
        return None

    plt = setup_matplotlib()
    output_path = Path(plots_dir) / "recompensa_episodios_avaliacao.png"

    rewards = series["rewards"]
    episodes = series["episodes"]
    window = 10 if len(rewards) >= 10 else max(1, len(rewards))
    avg_rewards = moving_average(rewards, window)

    plt.figure(figsize=(11, 5))
    plt.plot(episodes, rewards, alpha=0.35, linewidth=0.9, label="Episodios de avaliacao")
    plt.plot(episodes, avg_rewards, linewidth=2.0, label=f"Media movel ({window})")
    plt.xlabel("Episodio de avaliacao")
    plt.ylabel("Recompensa")
    plt.title("Recompensa nos episodios de avaliacao")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=140)
    plt.close()

    return output_path


def generate_plots(entry_dir, log_dir):
    plots_dir = Path(entry_dir) / "graficos"
    plots_dir.mkdir(parents=True, exist_ok=True)

    plot_paths = [
        save_training_reward_plot(log_dir, plots_dir),
        save_evaluation_plot(log_dir, plots_dir),
        save_eval_monitor_plot(log_dir, plots_dir),
    ]

    return [
        {
            "name": path.stem,
            "path": to_repo_path(path),
        }
        for path in plot_paths
        if path is not None
    ]


def command_for_model(model_path, max_steps=None, sleep=None):
    command = [
        "python",
        "reinforcement_learning/ver_modelo_atom.py",
        "--modelo",
        to_repo_path(model_path),
    ]
    if max_steps is not None:
        command += ["--max-steps", str(max_steps)]
    if sleep is not None:
        command += ["--sleep", str(sleep)]
    return " ".join(command)


def tensorboard_command(log_dir):
    return f"tensorboard --logdir {to_repo_path(log_dir)}"


def write_readme(entry_dir, metadata):
    readme_path = Path(entry_dir) / "README.md"
    best_model = metadata["models"].get("best")
    final_model = metadata["models"].get("final")

    lines = [
        f"# {metadata['run_name']}",
        "",
        f"- Registrado em: `{metadata['registered_at']}`",
        f"- Origem: `{metadata['source_run_dir']}`",
        f"- Device: `{metadata['training'].get('device')}`",
        f"- Timesteps finais: `{metadata['training'].get('final_timesteps')}`",
        f"- Frequencia de avaliacao: `{metadata['training'].get('eval_freq')}`",
        f"- Episodios por avaliacao: `{metadata['training'].get('n_eval_episodes')}`",
        "",
        "## Modelos",
        "",
    ]

    if best_model:
        lines.append(f"- Melhor avaliacao: `{best_model['path']}`")
    if final_model:
        lines.append(f"- Modelo final: `{final_model['path']}`")

    plots = metadata.get("plots") or []
    if plots:
        lines += [
            "",
            "## Graficos",
            "",
        ]
        for plot in plots:
            plot_path = from_repo_path(plot["path"])
            try:
                relative_plot_path = plot_path.relative_to(Path(entry_dir).resolve())
                relative_plot_path = relative_plot_path.as_posix()
            except ValueError:
                relative_plot_path = plot["path"]
            lines.append(f"![{plot['name']}]({relative_plot_path})")

    lines += [
        "",
        "## Ver no MuJoCo",
        "",
        "```bash",
    ]

    if best_model:
        lines.append(metadata["commands"]["view_best"])
    elif final_model:
        lines.append(metadata["commands"]["view_final"])
    else:
        lines.append("# Nenhum modelo copiado para esta entrada.")

    lines += [
        "```",
        "",
        "## TensorBoard",
        "",
        "```bash",
        metadata["commands"]["tensorboard"],
        "```",
        "",
        "## Parametros PPO",
        "",
        "```json",
        json.dumps(metadata["ppo_params"], indent=2, ensure_ascii=False),
        "```",
        "",
    ]

    evaluation = metadata.get("evaluation") or {}
    if evaluation:
        lines += [
            "## Avaliacao",
            "",
            "```json",
            json.dumps(evaluation, indent=2, ensure_ascii=False),
            "```",
            "",
        ]

    readme_path.write_text("\n".join(lines), encoding="utf-8")
    return readme_path


def load_model_records(registry_dir=DEFAULT_REGISTRY_DIR):
    registry_dir = Path(registry_dir)
    records = []
    if not registry_dir.exists():
        return records

    for metadata_path in sorted(registry_dir.glob("*/metadata.json")):
        data = load_json(metadata_path)
        if data:
            records.append(data)
    return records


def rebuild_index(registry_dir=DEFAULT_REGISTRY_DIR):
    registry_dir = Path(registry_dir)
    registry_dir.mkdir(parents=True, exist_ok=True)
    records = load_model_records(registry_dir)
    records.sort(key=lambda item: item.get("registered_at", ""))

    json_path = registry_dir / "index.json"
    json_path.write_text(
        json.dumps(records, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    csv_path = registry_dir / "index.csv"
    fieldnames = [
        "run_name",
        "registered_at",
        "device",
        "final_timesteps",
        "eval_freq",
        "n_eval_episodes",
        "best_mean_reward",
        "last_mean_reward",
        "best_model",
        "final_model",
        "view_best_command",
    ]

    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            eval_npz = (record.get("evaluation") or {}).get("evaluations_npz") or {}
            models = record.get("models") or {}
            writer.writerow(
                {
                    "run_name": record.get("run_name"),
                    "registered_at": record.get("registered_at"),
                    "device": (record.get("training") or {}).get("device"),
                    "final_timesteps": (record.get("training") or {}).get(
                        "final_timesteps"
                    ),
                    "eval_freq": (record.get("training") or {}).get("eval_freq"),
                    "n_eval_episodes": (record.get("training") or {}).get(
                        "n_eval_episodes"
                    ),
                    "best_mean_reward": eval_npz.get("best_mean_reward"),
                    "last_mean_reward": eval_npz.get("last_mean_reward"),
                    "best_model": (models.get("best") or {}).get("path"),
                    "final_model": (models.get("final") or {}).get("path"),
                    "view_best_command": (record.get("commands") or {}).get(
                        "view_best"
                    ),
                }
            )

    return {"json": json_path, "csv": csv_path, "records": records}


def register_training_run(
    run_name,
    source_run_dir,
    best_model_path,
    final_model_path,
    log_dir,
    ppo_params,
    training_args,
    registry_dir=DEFAULT_REGISTRY_DIR,
    best_params_path=None,
):
    registry_dir = Path(registry_dir)
    entry_name = slugify(run_name)
    entry_dir = registry_dir / entry_name
    entry_dir.mkdir(parents=True, exist_ok=True)

    best_model = copy_model(best_model_path, entry_dir / "best_model.zip")
    final_model = copy_model(final_model_path, entry_dir / "final_model.zip")
    best_params = load_json(best_params_path) if best_params_path else None

    metadata = {
        "run_name": run_name,
        "registry_name": entry_name,
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "source_run_dir": to_repo_path(source_run_dir),
        "log_dir": to_repo_path(log_dir),
        "models": {
            "best": best_model,
            "final": final_model,
        },
        "training": json_safe(training_args),
        "ppo_params": json_safe(ppo_params),
        "best_params_file": to_repo_path(best_params_path)
        if best_params_path and Path(best_params_path).exists()
        else None,
        "best_params": best_params,
        "evaluation": read_eval_summary(log_dir),
    }
    metadata["plots"] = generate_plots(entry_dir, log_dir)

    commands = {"tensorboard": tensorboard_command(log_dir)}
    if best_model:
        commands["view_best"] = command_for_model(from_repo_path(best_model["path"]))
    if final_model:
        commands["view_final"] = command_for_model(from_repo_path(final_model["path"]))
    metadata["commands"] = commands

    metadata_path = entry_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    readme_path = write_readme(entry_dir, metadata)
    index = rebuild_index(registry_dir)

    return {
        "entry_dir": entry_dir,
        "metadata_path": metadata_path,
        "readme_path": readme_path,
        "index_json": index["json"],
        "index_csv": index["csv"],
        "metadata": metadata,
    }
