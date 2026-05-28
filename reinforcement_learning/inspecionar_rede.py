import argparse
import os
import sys

from stable_baselines3 import PPO

from rede_utils import imprimir_especificacoes_rede


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_PATH = os.path.join(PROJECT_ROOT, "op3_model", "src")

sys.path.insert(0, SRC_PATH)

# Registra o ambiente Atom-v1 caso o modelo precise dessas definicoes ao carregar.
import atom  # noqa: F401,E402


DEFAULT_MODEL_PATH = os.path.join(PROJECT_ROOT, "saida_treino_atom", "ppo_atom_final.zip")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inspeciona a arquitetura de uma rede PPO ja treinada."
    )
    parser.add_argument(
        "modelo",
        nargs="?",
        default=DEFAULT_MODEL_PATH,
        help="Caminho para o arquivo .zip do modelo salvo.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Dispositivo usado ao carregar o modelo. Padrao: cpu.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Opcional: salva as especificacoes tambem em um arquivo .txt.",
    )
    return parser.parse_args()


def resolver_modelo(modelo):
    if os.path.isabs(modelo):
        return modelo

    caminho_cwd = os.path.abspath(modelo)
    if os.path.exists(caminho_cwd):
        return caminho_cwd

    return os.path.join(PROJECT_ROOT, modelo)


def main():
    args = parse_args()
    model_path = resolver_modelo(args.modelo)

    model = PPO.load(model_path, device=args.device)
    imprimir_especificacoes_rede(model, output_path=args.output)


if __name__ == "__main__":
    main()
