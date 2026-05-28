import os
import sys
import time
import argparse

import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback

from rede_utils import imprimir_especificacoes_rede


# ============================================================
# CONFIGURAÇÃO DE CAMINHOS
# ============================================================

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_PATH = os.path.join(PROJECT_ROOT, "op3_model", "src")

sys.path.insert(0, SRC_PATH)

# Importante: registra o ambiente Atom-v1
import atom  # noqa: F401


BASE_SAVE_DIR = "saida_experimentos_tcc1"

os.makedirs(BASE_SAVE_DIR, exist_ok=True)


# ============================================================
# LISTA DE EXPERIMENTOS
# ============================================================

EXPERIMENTOS = [
    {
        "nome": "exp_01",
        "total_timesteps": 100_000,
        "n_epochs": 5,
        "n_steps": 512,
        "observacao": "Treino curto com poucas épocas",
    },
    {
        "nome": "exp_02",
        "total_timesteps": 100_000,
        "n_epochs": 10,
        "n_steps": 512,
        "observacao": "Treino curto com mais épocas",
    },
    {
        "nome": "exp_03",
        "total_timesteps": 200_000,
        "n_epochs": 5,
        "n_steps": 1024,
        "observacao": "Treino intermediário inicial",
    },
    {
        "nome": "exp_04",
        "total_timesteps": 200_000,
        "n_epochs": 10,
        "n_steps": 1024,
        "observacao": "Treino intermediário com mais épocas",
    },
    {
        "nome": "exp_05",
        "total_timesteps": 500_000,
        "n_epochs": 5,
        "n_steps": 1024,
        "observacao": "Treino médio",
    },
    {
        "nome": "exp_06",
        "total_timesteps": 500_000,
        "n_epochs": 10,
        "n_steps": 1024,
        "observacao": "Treino médio com configuração base",
    },
    {
        "nome": "exp_07",
        "total_timesteps": 1_000_000,
        "n_epochs": 5,
        "n_steps": 1024,
        "observacao": "Treino longo com menos épocas",
    },
    {
        "nome": "exp_08",
        "total_timesteps": 1_000_000,
        "n_epochs": 10,
        "n_steps": 1024,
        "observacao": "Treino longo com configuração base",
    },
    {
        "nome": "exp_09",
        "total_timesteps": 2_000_000,
        "n_epochs": 10,
        "n_steps": 1024,
        "observacao": "Configuração principal do projeto",
    },
    {
        "nome": "exp_10",
        "total_timesteps": 2_000_000,
        "n_epochs": 15,
        "n_steps": 1024,
        "observacao": "Treino longo com mais épocas",
    },
]


# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def moving_average(values, window=50):
    values = np.array(values, dtype=float)

    if len(values) == 0:
        return values

    if len(values) < window:
        return values

    weights = np.ones(window) / window
    return np.convolve(values, weights, mode="valid")


def find_monitor_file(prefix):
    possible_files = [
        prefix,
        prefix + ".monitor.csv",
        prefix + ".csv",
        prefix + ".csv.monitor.csv",
    ]

    for file_path in possible_files:
        if os.path.exists(file_path):
            return file_path

    return None


def load_monitor_csv(prefix):
    monitor_file = find_monitor_file(prefix)

    if monitor_file is None:
        print(f"Arquivo Monitor não encontrado para prefixo: {prefix}")
        return None

    try:
        return pd.read_csv(monitor_file, skiprows=1)
    except Exception as error:
        print(f"Erro ao ler o arquivo Monitor: {error}")
        return None


def save_plot(path):
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Gráfico salvo: {path}")


# ============================================================
# GRÁFICOS DE CADA EXPERIMENTO
# ============================================================

def plot_training_rewards(monitor_prefix, output_dir, nome_experimento):
    df = load_monitor_csv(monitor_prefix)

    if df is None or df.empty:
        print(f"Sem dados de treino para plotar em {nome_experimento}.")
        return None

    if "r" not in df.columns or "l" not in df.columns:
        print("Arquivo Monitor não contém as colunas esperadas: r e l.")
        print("Colunas encontradas:", list(df.columns))
        return None

    rewards = df["r"].values
    episode_lengths = df["l"].values
    episodes = np.arange(1, len(rewards) + 1)

    window = 50

    # ------------------------------------------------------------
    # 1. Recompensa por episódio
    # ------------------------------------------------------------
    plt.figure(figsize=(12, 6))
    plt.plot(episodes, rewards, linewidth=1)
    plt.xlabel("Episódio")
    plt.ylabel("Recompensa total")
    plt.title(f"{nome_experimento} - Recompensa por episódio")
    plt.grid(True, alpha=0.3)
    save_plot(os.path.join(output_dir, "01_recompensa_por_episodio.png"))

    # ------------------------------------------------------------
    # 2. Recompensa média móvel
    # ------------------------------------------------------------
    rewards_ma = moving_average(rewards, window=window)

    if len(rewards) >= window:
        ma_episodes = np.arange(window, window + len(rewards_ma))
    else:
        ma_episodes = episodes

    plt.figure(figsize=(12, 6))
    plt.plot(ma_episodes, rewards_ma, linewidth=2)
    plt.xlabel("Episódio")
    plt.ylabel(f"Recompensa média móvel ({window} episódios)")
    plt.title(f"{nome_experimento} - Recompensa suavizada")
    plt.grid(True, alpha=0.3)
    save_plot(os.path.join(output_dir, "02_recompensa_media_movel.png"))

    # ------------------------------------------------------------
    # 3. Duração dos episódios
    # ------------------------------------------------------------
    plt.figure(figsize=(12, 6))
    plt.plot(episodes, episode_lengths, linewidth=1)
    plt.xlabel("Episódio")
    plt.ylabel("Duração do episódio em timesteps")
    plt.title(f"{nome_experimento} - Duração dos episódios")
    plt.grid(True, alpha=0.3)
    save_plot(os.path.join(output_dir, "03_duracao_episodios.png"))

    # ------------------------------------------------------------
    # 4. Duração média móvel
    # ------------------------------------------------------------
    lengths_ma = moving_average(episode_lengths, window=window)

    if len(episode_lengths) >= window:
        ma_length_episodes = np.arange(window, window + len(lengths_ma))
    else:
        ma_length_episodes = episodes

    plt.figure(figsize=(12, 6))
    plt.plot(ma_length_episodes, lengths_ma, linewidth=2)
    plt.xlabel("Episódio")
    plt.ylabel(f"Duração média móvel ({window} episódios)")
    plt.title(f"{nome_experimento} - Duração média dos episódios")
    plt.grid(True, alpha=0.3)
    save_plot(os.path.join(output_dir, "04_duracao_media_movel.png"))

    # ------------------------------------------------------------
    # 5. Distribuição das recompensas
    # ------------------------------------------------------------
    plt.figure(figsize=(10, 6))
    plt.hist(rewards, bins=40)
    plt.xlabel("Recompensa total do episódio")
    plt.ylabel("Frequência")
    plt.title(f"{nome_experimento} - Distribuição das recompensas")
    plt.grid(True, alpha=0.3)
    save_plot(os.path.join(output_dir, "05_distribuicao_recompensas.png"))

    # ------------------------------------------------------------
    # 6. Comparação início vs final
    # ------------------------------------------------------------
    n_compare = min(100, len(rewards))

    initial_rewards = rewards[:n_compare]
    final_rewards = rewards[-n_compare:]

    labels = [
        f"Primeiros {n_compare} episódios",
        f"Últimos {n_compare} episódios",
    ]

    values = [
        np.mean(initial_rewards),
        np.mean(final_rewards),
    ]

    plt.figure(figsize=(9, 6))
    plt.bar(labels, values)
    plt.ylabel("Recompensa média")
    plt.title(f"{nome_experimento} - Início vs final do treino")
    plt.grid(True, axis="y", alpha=0.3)
    save_plot(os.path.join(output_dir, "06_comparacao_inicio_fim_recompensa.png"))

    summary = {
        "total_episodes": int(len(rewards)),
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "min_reward": float(np.min(rewards)),
        "max_reward": float(np.max(rewards)),
        "first_100_mean_reward": float(np.mean(initial_rewards)),
        "last_100_mean_reward": float(np.mean(final_rewards)),
        "mean_episode_length": float(np.mean(episode_lengths)),
        "last_100_mean_episode_length": float(np.mean(episode_lengths[-n_compare:])),
    }

    summary_path = os.path.join(output_dir, "resumo_treinamento.csv")
    pd.DataFrame([summary]).to_csv(summary_path, index=False)

    return summary


def plot_eval_results(log_dir, output_dir, nome_experimento):
    eval_file = os.path.join(log_dir, "evaluations.npz")

    if not os.path.exists(eval_file):
        print(f"Arquivo de avaliação não encontrado: {eval_file}")
        return None

    data = np.load(eval_file)

    if "timesteps" not in data or "results" not in data:
        print("Arquivo evaluations.npz não contém timesteps/results.")
        print("Chaves encontradas:", list(data.keys()))
        return None

    timesteps = data["timesteps"]
    results = data["results"]

    mean_rewards = np.mean(results, axis=1)
    std_rewards = np.std(results, axis=1)
    best_so_far = np.maximum.accumulate(mean_rewards)

    # ------------------------------------------------------------
    # 7. Curva de avaliação
    # ------------------------------------------------------------
    plt.figure(figsize=(12, 6))
    plt.plot(timesteps, mean_rewards, linewidth=2, label="Recompensa média")
    plt.fill_between(
        timesteps,
        mean_rewards - std_rewards,
        mean_rewards + std_rewards,
        alpha=0.2,
        label="Desvio padrão",
    )
    plt.xlabel("Timesteps")
    plt.ylabel("Recompensa média de avaliação")
    plt.title(f"{nome_experimento} - Curva de avaliação")
    plt.legend()
    plt.grid(True, alpha=0.3)
    save_plot(os.path.join(output_dir, "07_avaliacao_recompensa_media.png"))

    # ------------------------------------------------------------
    # 8. Melhor avaliação acumulada
    # ------------------------------------------------------------
    plt.figure(figsize=(12, 6))
    plt.plot(timesteps, best_so_far, linewidth=2)
    plt.xlabel("Timesteps")
    plt.ylabel("Melhor recompensa média até o momento")
    plt.title(f"{nome_experimento} - Melhor desempenho acumulado")
    plt.grid(True, alpha=0.3)
    save_plot(os.path.join(output_dir, "08_melhor_recompensa_avaliacao.png"))

    eval_df = pd.DataFrame(
        {
            "timesteps": timesteps,
            "mean_reward": mean_rewards,
            "std_reward": std_rewards,
            "best_reward_so_far": best_so_far,
        }
    )

    eval_csv_path = os.path.join(output_dir, "avaliacoes_modelo.csv")
    eval_df.to_csv(eval_csv_path, index=False)

    summary = {
        "best_eval_reward": float(np.max(mean_rewards)),
        "last_eval_reward": float(mean_rewards[-1]),
        "mean_eval_reward": float(np.mean(mean_rewards)),
        "std_last_eval_reward": float(std_rewards[-1]),
    }

    return summary


# ============================================================
# GRÁFICOS COMPARATIVOS ENTRE EXPERIMENTOS
# ============================================================

def gerar_graficos_comparativos(resultados_df):
    output_dir = os.path.join(BASE_SAVE_DIR, "graficos_comparativos")
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------
    # Tempo de treino por experimento
    # ------------------------------------------------------------
    plt.figure(figsize=(13, 6))
    plt.bar(resultados_df["experimento"], resultados_df["tempo_total_minutos"])
    plt.xlabel("Experimento")
    plt.ylabel("Tempo de treino em minutos")
    plt.title("Tempo de treinamento por experimento")
    plt.xticks(rotation=45)
    plt.grid(True, axis="y", alpha=0.3)
    save_plot(os.path.join(output_dir, "tempo_treino_por_experimento.png"))

    # ------------------------------------------------------------
    # Recompensa média final dos últimos 100 episódios
    # ------------------------------------------------------------
    plt.figure(figsize=(13, 6))
    plt.bar(resultados_df["experimento"], resultados_df["last_100_mean_reward"])
    plt.xlabel("Experimento")
    plt.ylabel("Recompensa média dos últimos 100 episódios")
    plt.title("Desempenho final por experimento")
    plt.xticks(rotation=45)
    plt.grid(True, axis="y", alpha=0.3)
    save_plot(os.path.join(output_dir, "recompensa_final_por_experimento.png"))

    # ------------------------------------------------------------
    # Melhor recompensa de avaliação
    # ------------------------------------------------------------
    plt.figure(figsize=(13, 6))
    plt.bar(resultados_df["experimento"], resultados_df["best_eval_reward"])
    plt.xlabel("Experimento")
    plt.ylabel("Melhor recompensa média de avaliação")
    plt.title("Melhor desempenho de avaliação por experimento")
    plt.xticks(rotation=45)
    plt.grid(True, axis="y", alpha=0.3)
    save_plot(os.path.join(output_dir, "melhor_avaliacao_por_experimento.png"))

    # ------------------------------------------------------------
    # Timesteps x tempo
    # ------------------------------------------------------------
    plt.figure(figsize=(10, 6))
    plt.scatter(resultados_df["total_timesteps"], resultados_df["tempo_total_minutos"])
    plt.xlabel("Total de timesteps")
    plt.ylabel("Tempo de treino em minutos")
    plt.title("Relação entre timesteps e tempo de treinamento")
    plt.grid(True, alpha=0.3)
    save_plot(os.path.join(output_dir, "timesteps_vs_tempo.png"))

    # ------------------------------------------------------------
    # Timesteps x recompensa final
    # ------------------------------------------------------------
    plt.figure(figsize=(10, 6))
    plt.scatter(resultados_df["total_timesteps"], resultados_df["last_100_mean_reward"])
    plt.xlabel("Total de timesteps")
    plt.ylabel("Recompensa média dos últimos 100 episódios")
    plt.title("Relação entre timesteps e desempenho final")
    plt.grid(True, alpha=0.3)
    save_plot(os.path.join(output_dir, "timesteps_vs_recompensa_final.png"))


# ============================================================
# EXECUÇÃO DE UM EXPERIMENTO
# ============================================================

def rodar_experimento(config, inspecionar_rede=False):
    nome = config["nome"]

    print("\n============================================================")
    print(f"INICIANDO EXPERIMENTO: {nome}")
    print("============================================================")
    print(config)

    save_dir = os.path.join(BASE_SAVE_DIR, nome)
    best_dir = os.path.join(save_dir, "melhor_modelo")
    log_dir = os.path.join(save_dir, "logs")
    plots_dir = os.path.join(save_dir, "graficos")

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(best_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    train_monitor_prefix = os.path.join(log_dir, "train")
    eval_monitor_prefix = os.path.join(log_dir, "eval")

    # ------------------------------------------------------------
    # Ambientes
    # ------------------------------------------------------------
    train_env = gym.make("Atom-v1", render_mode=None)
    if not inspecionar_rede:
        train_env = Monitor(train_env, filename=train_monitor_prefix)

    eval_env = None
    if not inspecionar_rede:
        eval_env = gym.make("Atom-v1", render_mode=None)
        eval_env = Monitor(eval_env, filename=eval_monitor_prefix)

    # ------------------------------------------------------------
    # Modelo PPO
    # A função de recompensa fica a mesma porque está no ambiente.
    # Aqui mudamos apenas hiperparâmetros de treino.
    # ------------------------------------------------------------
    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        verbose=1,
        device="cuda",

        learning_rate=3e-4,
        n_steps=config["n_steps"],
        batch_size=64,
        n_epochs=config["n_epochs"],

        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,

        tensorboard_log=log_dir,
    )

    if inspecionar_rede:
        output_path = os.path.join(save_dir, "especificacoes_rede.txt")
        imprimir_especificacoes_rede(model, env=train_env, output_path=output_path)
        train_env.close()
        return None

    # ------------------------------------------------------------
    # Callback de avaliação
    # ------------------------------------------------------------
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=best_dir,
        log_path=log_dir,
        eval_freq=5000,
        deterministic=True,
        render=False,
    )

    # ------------------------------------------------------------
    # Treinamento com medição de tempo
    # ------------------------------------------------------------
    inicio = time.perf_counter()

    model.learn(
        total_timesteps=config["total_timesteps"],
        callback=eval_callback,
        tb_log_name=nome,
    )

    fim = time.perf_counter()

    tempo_total_segundos = fim - inicio
    tempo_total_minutos = tempo_total_segundos / 60

    # ------------------------------------------------------------
    # Salva modelo final
    # ------------------------------------------------------------
    final_model_path = os.path.join(save_dir, f"ppo_atom_final_{nome}")
    model.save(final_model_path)

    print("\n============================================================")
    print(f"Experimento finalizado: {nome}")
    print(f"Tempo total: {tempo_total_minutos:.2f} minutos")
    print(f"Modelo final salvo em: {final_model_path}.zip")
    print(f"Melhor modelo salvo em: {os.path.join(best_dir, 'best_model.zip')}")
    print("============================================================")

    # ------------------------------------------------------------
    # Fecha ambientes
    # ------------------------------------------------------------
    train_env.close()
    eval_env.close()

    # ------------------------------------------------------------
    # Gera gráficos e resumos
    # ------------------------------------------------------------
    train_summary = plot_training_rewards(
        monitor_prefix=train_monitor_prefix,
        output_dir=plots_dir,
        nome_experimento=nome,
    )

    eval_summary = plot_eval_results(
        log_dir=log_dir,
        output_dir=plots_dir,
        nome_experimento=nome,
    )

    if train_summary is None:
        train_summary = {}

    if eval_summary is None:
        eval_summary = {}

    # ------------------------------------------------------------
    # Resultado final do experimento
    # ------------------------------------------------------------
    resultado = {
        "experimento": nome,
        "total_timesteps": config["total_timesteps"],
        "n_epochs": config["n_epochs"],
        "n_steps": config["n_steps"],
        "learning_rate": 3e-4,
        "batch_size": 64,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "tempo_total_segundos": tempo_total_segundos,
        "tempo_total_minutos": tempo_total_minutos,
        "observacao": config["observacao"],
    }

    resultado.update(train_summary)
    resultado.update(eval_summary)

    resultado_path = os.path.join(save_dir, "resultado_experimento.csv")
    pd.DataFrame([resultado]).to_csv(resultado_path, index=False)

    print(f"Resumo do experimento salvo em: {resultado_path}")

    return resultado


# ============================================================
# MAIN
# ============================================================

def parse_args():
    nomes_experimentos = [config["nome"] for config in EXPERIMENTOS]

    parser = argparse.ArgumentParser(
        description="Executa experimentos PPO ou apenas inspeciona a rede."
    )
    parser.add_argument(
        "--inspecionar-rede",
        action="store_true",
        help="Mostra a arquitetura da rede e encerra antes de treinar.",
    )
    parser.add_argument(
        "--experimento",
        choices=nomes_experimentos,
        default=nomes_experimentos[0],
        help="Experimento usado ao inspecionar a rede. Padrao: exp_01.",
    )
    return parser.parse_args()


def buscar_config_experimento(nome):
    for config in EXPERIMENTOS:
        if config["nome"] == nome:
            return config

    raise ValueError(f"Experimento nao encontrado: {nome}")


def main():
    args = parse_args()

    if args.inspecionar_rede:
        config = buscar_config_experimento(args.experimento)
        rodar_experimento(config, inspecionar_rede=True)
        return

    resultados = []

    for config in EXPERIMENTOS:
        resultado = rodar_experimento(config)
        resultados.append(resultado)

        resultados_df = pd.DataFrame(resultados)

        resultados_parcial_path = os.path.join(
            BASE_SAVE_DIR,
            "resultados_experimentos_parcial.csv",
        )

        resultados_df.to_csv(resultados_parcial_path, index=False)

        print("\nResumo parcial salvo em:")
        print(resultados_parcial_path)

    # ------------------------------------------------------------
    # CSV final com todos os experimentos
    # ------------------------------------------------------------
    resultados_df = pd.DataFrame(resultados)

    resultados_final_path = os.path.join(
        BASE_SAVE_DIR,
        "resultados_experimentos_tcc1.csv",
    )

    resultados_df.to_csv(resultados_final_path, index=False)

    print("\n============================================================")
    print("TODOS OS EXPERIMENTOS FINALIZADOS")
    print("============================================================")
    print(f"Resultados finais salvos em: {resultados_final_path}")

    # ------------------------------------------------------------
    # Gera gráficos comparativos
    # ------------------------------------------------------------
    gerar_graficos_comparativos(resultados_df)

    print("\nGráficos comparativos salvos em:")
    print(os.path.join(BASE_SAVE_DIR, "graficos_comparativos"))

    print("\nTabela final:")
    print(resultados_df)


if __name__ == "__main__":
    main()
