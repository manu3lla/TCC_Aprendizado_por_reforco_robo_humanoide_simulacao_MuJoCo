import argparse
import json
import subprocess
import sys
from pathlib import Path

from model_registry import (
    DEFAULT_REGISTRY_DIR,
    REPO_ROOT,
    from_repo_path,
    load_model_records,
    rebuild_index,
    register_training_run,
)


def find_record(name, registry_dir):
    records = load_model_records(registry_dir)
    for record in records:
        if name in {record.get("run_name"), record.get("registry_name")}:
            return record
    raise SystemExit(f"Modelo nao encontrado na biblioteca: {name}")


def cmd_list(args):
    index = rebuild_index(args.registry_dir)
    records = index["records"]
    if not records:
        print(f"Nenhum modelo registrado em: {args.registry_dir}")
        return

    print(f"Biblioteca: {args.registry_dir}")
    print()
    for record in records:
        evaluation = (record.get("evaluation") or {}).get("evaluations_npz") or {}
        training = record.get("training") or {}
        print(
            f"- {record.get('registry_name')} | "
            f"device={training.get('device')} | "
            f"steps={training.get('final_timesteps')} | "
            f"best_mean={evaluation.get('best_mean_reward')}"
        )

    print()
    print(f"Indice CSV: {index['csv']}")


def cmd_info(args):
    record = find_record(args.nome, args.registry_dir)
    if args.json:
        print(json.dumps(record, indent=2, ensure_ascii=False))
        return

    entry_dir = Path(args.registry_dir) / record["registry_name"]
    print(entry_dir / "README.md")


def cmd_ver(args):
    record = find_record(args.nome, args.registry_dir)
    kind = "final" if args.final else "best"
    model_info = (record.get("models") or {}).get(kind)
    if not model_info:
        raise SystemExit(f"Modelo '{kind}' nao existe para: {args.nome}")

    model_path = from_repo_path(model_info["path"])
    command = [
        sys.executable,
        str(REPO_ROOT / "reinforcement_learning" / "ver_modelo_atom.py"),
        "--modelo",
        str(model_path),
        "--max-steps",
        str(args.max_steps),
        "--sleep",
        str(args.sleep),
    ]
    if args.estocastico:
        command.append("--estocastico")

    subprocess.run(command, check=True)


def cmd_registrar(args):
    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        raise SystemExit(f"Pasta nao encontrada: {run_dir}")

    run_name = args.nome or run_dir.name
    save_dir = run_dir.parent
    best_params_path = save_dir / "best_params.json"
    best_params = None
    ppo_params = {}

    if best_params_path.exists():
        with best_params_path.open("r", encoding="utf-8") as file:
            best_params = json.load(file)
        ppo_params = best_params.get("ppo_params", {})

    result = register_training_run(
        run_name=run_name,
        source_run_dir=run_dir,
        best_model_path=run_dir / "melhor_modelo" / "best_model.zip",
        final_model_path=run_dir / "ppo_atom_optuna_final.zip",
        log_dir=run_dir / "logs",
        ppo_params=ppo_params,
        training_args={
            "registered_manually": True,
            "inferred_from": str(run_dir),
            "device": args.device,
            "final_timesteps": args.final_timesteps,
            "eval_freq": args.eval_freq,
            "n_eval_episodes": args.n_eval_episodes,
        },
        registry_dir=args.registry_dir,
        best_params_path=best_params_path if best_params else None,
    )

    print(f"Modelo registrado em: {result['entry_dir']}")
    print(f"Documentacao: {result['readme_path']}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Gerencia a biblioteca de melhores modelos treinados."
    )
    parser.add_argument(
        "--registry-dir",
        default=str(DEFAULT_REGISTRY_DIR),
        type=Path,
        help="Pasta da biblioteca de modelos.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("listar", help="Lista modelos registrados.")
    list_parser.set_defaults(func=cmd_list)

    info_parser = subparsers.add_parser("info", help="Mostra documentacao do modelo.")
    info_parser.add_argument("nome")
    info_parser.add_argument("--json", action="store_true")
    info_parser.set_defaults(func=cmd_info)

    view_parser = subparsers.add_parser("ver", help="Abre modelo registrado no MuJoCo.")
    view_parser.add_argument("nome")
    view_parser.add_argument("--final", action="store_true")
    view_parser.add_argument("--estocastico", action="store_true")
    view_parser.add_argument("--max-steps", type=int, default=10_000)
    view_parser.add_argument("--sleep", type=float, default=0.01)
    view_parser.set_defaults(func=cmd_ver)

    register_parser = subparsers.add_parser(
        "registrar",
        help="Registra uma pasta de treino ja existente.",
    )
    register_parser.add_argument("--run-dir", required=True, type=Path)
    register_parser.add_argument("--nome")
    register_parser.add_argument("--device")
    register_parser.add_argument("--final-timesteps", type=int)
    register_parser.add_argument("--eval-freq", type=int)
    register_parser.add_argument("--n-eval-episodes", type=int)
    register_parser.set_defaults(func=cmd_registrar)

    return parser.parse_args()


def main():
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
