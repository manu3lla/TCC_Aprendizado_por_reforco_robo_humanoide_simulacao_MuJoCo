# import os
# import sys
# import gymnasium as gym

# PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
# SRC_PATH = os.path.join(PROJECT_ROOT, "op3_model", "src")
# sys.path.insert(0, SRC_PATH)

# import atom

# from stable_baselines3 import PPO
# from stable_baselines3.common.monitor import Monitor
# from stable_baselines3.common.callbacks import EvalCallback

# SAVE_DIR = "saida_treino_atom"
# BEST_DIR = os.path.join(SAVE_DIR, "melhor_modelo")
# LOG_DIR = os.path.join(SAVE_DIR, "logs")

# os.makedirs(SAVE_DIR, exist_ok=True)
# os.makedirs(BEST_DIR, exist_ok=True)
# os.makedirs(LOG_DIR, exist_ok=True)

# # ambiente de treino
# train_env = gym.make("Atom-v1", render_mode=None)
# train_env = Monitor(train_env)

# # ambiente de avaliação
# eval_env = gym.make("Atom-v1", render_mode=None)
# eval_env = Monitor(eval_env)

# #ver hiperparametros 
# model = PPO(
#     policy="MlpPolicy",
#     env=train_env,
#     verbose=1,
#     device="cuda",
#     learning_rate=3e-4,
#     n_steps=1024,
#     batch_size=64,
#     n_epochs=10, 
#     gamma=0.99,
#     gae_lambda=0.95,
#     clip_range=0.2,
#     ent_coef=0.0,
# )

# eval_callback = EvalCallback(
#     eval_env,
#     best_model_save_path=BEST_DIR,
#     log_path=LOG_DIR,
#     eval_freq=5000,
#     deterministic=True,
#     render=False
# )

# model.learn(
#     total_timesteps=2000000,
#     callback=eval_callback
# )

# # salva também o final
# final_model_path = os.path.join(SAVE_DIR, "ppo_atom_final")
# model.save(final_model_path)

# print(f"Modelo final salvo em: {final_model_path}.zip")
# print(f"Melhor modelo salvo em: {os.path.join(BEST_DIR, 'best_model.zip')}")

# train_env.close()
# eval_env.close()

import os
import sys

import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback


# ============================================================
# CONFIGURAÇÃO DE CAMINHOS
# ============================================================

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_PATH = os.path.join(PROJECT_ROOT, "op3_model", "src")

# Adiciona o src do op3_model ao path do Python
sys.path.insert(0, SRC_PATH)

# Importa o ambiente Atom-v1
# Esse import é importante porque registra o ambiente no Gymnasium
import atom  # noqa: F401


SAVE_DIR = "saida_treino_atom"
BEST_DIR = os.path.join(SAVE_DIR, "melhor_modelo")
LOG_DIR = os.path.join(SAVE_DIR, "logs")
PLOTS_DIR = os.path.join(SAVE_DIR, "graficos")

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(BEST_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


# O Monitor do Stable-Baselines3 adiciona ".monitor.csv" automaticamente
TRAIN_MONITOR_PREFIX = os.path.join(LOG_DIR, "train")
EVAL_MONITOR_PREFIX = os.path.join(LOG_DIR, "eval")


# ============================================================
# FUNÇÕES PARA GERAR GRÁFICOS
# ============================================================

def moving_average(values, window=50):
    """
    Calcula média móvel para suavizar curvas.
    """
    values = np.array(values, dtype=float)

    if len(values) == 0:
        return values

    if len(values) < window:
        return values

    weights = np.ones(window) / window
    return np.convolve(values, weights, mode="valid")


def find_monitor_file(prefix):
    """
    Encontra o arquivo gerado pelo Monitor.

    Quando usamos:
        Monitor(env, filename="saida_treino_atom/logs/train")

    O Stable-Baselines3 normalmente gera:
        saida_treino_atom/logs/train.monitor.csv
    """
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
    """
    Lê o CSV gerado pelo Monitor do Stable-Baselines3.

    O arquivo do Monitor tem uma primeira linha de metadados começando com #.
    Por isso usamos skiprows=1.
    """
    monitor_file = find_monitor_file(prefix)

    if monitor_file is None:
        print(f"Arquivo Monitor não encontrado para prefixo: {prefix}")
        return None

    print(f"Lendo Monitor: {monitor_file}")

    try:
        return pd.read_csv(monitor_file, skiprows=1)
    except Exception as error:
        print(f"Erro ao ler o arquivo Monitor: {error}")
        return None


def save_plot(path):
    """
    Salva o gráfico com qualidade boa para TCC.
    """
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Gráfico salvo: {path}")


def plot_training_rewards(monitor_prefix, output_dir):
    """
    Gera gráficos a partir dos dados de treino:
    - recompensa por episódio
    - recompensa com média móvel
    - duração dos episódios
    - duração com média móvel
    - distribuição das recompensas
    - comparação início vs final do treino
    """
    df = load_monitor_csv(monitor_prefix)

    if df is None or df.empty:
        print("Sem dados de treino para plotar.")
        return

    if "r" not in df.columns or "l" not in df.columns:
        print("Arquivo Monitor não contém as colunas esperadas: r e l.")
        print("Colunas encontradas:", list(df.columns))
        return

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
    plt.title("Curva de aprendizado - recompensa por episódio")
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
    plt.title("Curva de aprendizado - recompensa suavizada")
    plt.grid(True, alpha=0.3)
    save_plot(os.path.join(output_dir, "02_recompensa_media_movel.png"))

    # ------------------------------------------------------------
    # 3. Duração dos episódios
    # ------------------------------------------------------------
    plt.figure(figsize=(12, 6))
    plt.plot(episodes, episode_lengths, linewidth=1)
    plt.xlabel("Episódio")
    plt.ylabel("Duração do episódio em timesteps")
    plt.title("Duração dos episódios durante o treinamento")
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
    plt.title("Evolução da duração média dos episódios")
    plt.grid(True, alpha=0.3)
    save_plot(os.path.join(output_dir, "04_duracao_media_movel.png"))

    # ------------------------------------------------------------
    # 5. Distribuição das recompensas
    # ------------------------------------------------------------
    plt.figure(figsize=(10, 6))
    plt.hist(rewards, bins=40)
    plt.xlabel("Recompensa total do episódio")
    plt.ylabel("Frequência")
    plt.title("Distribuição das recompensas por episódio")
    plt.grid(True, alpha=0.3)
    save_plot(os.path.join(output_dir, "05_distribuicao_recompensas.png"))

    # ------------------------------------------------------------
    # 6. Comparação início vs final do treino
    # ------------------------------------------------------------
    n_compare = min(100, len(rewards))

    if n_compare > 0:
        initial_rewards = rewards[:n_compare]
        final_rewards = rewards[-n_compare:]

        labels = [f"Primeiros {n_compare} episódios", f"Últimos {n_compare} episódios"]
        values = [np.mean(initial_rewards), np.mean(final_rewards)]

        plt.figure(figsize=(9, 6))
        plt.bar(labels, values)
        plt.ylabel("Recompensa média")
        plt.title("Comparação da recompensa média no início e no final do treino")
        plt.grid(True, axis="y", alpha=0.3)
        save_plot(os.path.join(output_dir, "06_comparacao_inicio_fim_recompensa.png"))

    # ------------------------------------------------------------
    # Resumo numérico
    # ------------------------------------------------------------
    summary = {
        "total_episodes": int(len(rewards)),
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "min_reward": float(np.min(rewards)),
        "max_reward": float(np.max(rewards)),
        "last_100_mean_reward": float(np.mean(rewards[-100:])) if len(rewards) >= 100 else float(np.mean(rewards)),
        "mean_episode_length": float(np.mean(episode_lengths)),
        "last_100_mean_episode_length": float(np.mean(episode_lengths[-100:])) if len(episode_lengths) >= 100 else float(np.mean(episode_lengths)),
    }

    summary_path = os.path.join(output_dir, "resumo_treinamento.csv")
    pd.DataFrame([summary]).to_csv(summary_path, index=False)

    print("\nResumo do treinamento:")
    for key, value in summary.items():
        print(f"{key}: {value}")

    print(f"\nResumo salvo em: {summary_path}")


def plot_eval_results(log_dir, output_dir):
    """
    Gera gráficos a partir do evaluations.npz criado pelo EvalCallback.
    """
    eval_file = os.path.join(log_dir, "evaluations.npz")

    if not os.path.exists(eval_file):
        print(f"Arquivo de avaliação não encontrado: {eval_file}")
        return

    print(f"Lendo avaliações: {eval_file}")

    data = np.load(eval_file)

    if "timesteps" not in data or "results" not in data:
        print("Arquivo evaluations.npz não contém timesteps/results.")
        print("Chaves encontradas:", list(data.keys()))
        return

    timesteps = data["timesteps"]
    results = data["results"]

    mean_rewards = np.mean(results, axis=1)
    std_rewards = np.std(results, axis=1)

    # ------------------------------------------------------------
    # 7. Curva de avaliação
    # ------------------------------------------------------------
    plt.figure(figsize=(12, 6))
    plt.plot(timesteps, mean_rewards, linewidth=2, label="Recompensa média de avaliação")
    plt.fill_between(
        timesteps,
        mean_rewards - std_rewards,
        mean_rewards + std_rewards,
        alpha=0.2,
        label="Desvio padrão",
    )
    plt.xlabel("Timesteps")
    plt.ylabel("Recompensa média")
    plt.title("Curva de avaliação do modelo durante o treinamento")
    plt.legend()
    plt.grid(True, alpha=0.3)
    save_plot(os.path.join(output_dir, "07_avaliacao_recompensa_media.png"))

    # ------------------------------------------------------------
    # 8. Melhor avaliação acumulada
    # ------------------------------------------------------------
    best_so_far = np.maximum.accumulate(mean_rewards)

    plt.figure(figsize=(12, 6))
    plt.plot(timesteps, best_so_far, linewidth=2)
    plt.xlabel("Timesteps")
    plt.ylabel("Melhor recompensa média até o momento")
    plt.title("Evolução do melhor desempenho durante o treinamento")
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

    print(f"Dados de avaliação salvos em: {eval_csv_path}")


def generate_all_plots():
    """
    Gera todos os gráficos pós-treinamento.
    """
    print("\n============================================================")
    print("GERANDO GRÁFICOS PÓS-TREINAMENTO")
    print("============================================================")

    plot_training_rewards(TRAIN_MONITOR_PREFIX, PLOTS_DIR)
    plot_eval_results(LOG_DIR, PLOTS_DIR)

    print("\n============================================================")
    print(f"Gráficos salvos em: {PLOTS_DIR}")
    print("============================================================")


# ============================================================
# TREINAMENTO
# ============================================================

def main():
    # ------------------------------------------------------------
    # Ambiente de treino
    # ------------------------------------------------------------
    train_env = gym.make("Atom-v1", render_mode=None)
    train_env = Monitor(train_env, filename=TRAIN_MONITOR_PREFIX)

    # ------------------------------------------------------------
    # Ambiente de avaliação
    # ------------------------------------------------------------
    eval_env = gym.make("Atom-v1", render_mode=None)
    eval_env = Monitor(eval_env, filename=EVAL_MONITOR_PREFIX)

    # ------------------------------------------------------------
    # Modelo PPO
    # ------------------------------------------------------------
    model = PPO(
        policy="MlpPolicy",
        env=train_env,
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
        tensorboard_log=LOG_DIR,
    )

    # ------------------------------------------------------------
    # Callback de avaliação
    # ------------------------------------------------------------
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=BEST_DIR,
        log_path=LOG_DIR,
        eval_freq=5000,
        deterministic=True,
        render=False,
    )

    # ------------------------------------------------------------
    # Treinamento
    # ------------------------------------------------------------
    model.learn(
        total_timesteps=2_000_000,
        callback=eval_callback,
        tb_log_name="ppo_atom",
    )

    # ------------------------------------------------------------
    # Salvando modelo final
    # ------------------------------------------------------------
    final_model_path = os.path.join(SAVE_DIR, "ppo_atom_final")
    model.save(final_model_path)

    print("\n============================================================")
    print(f"Modelo final salvo em: {final_model_path}.zip")
    print(f"Melhor modelo salvo em: {os.path.join(BEST_DIR, 'best_model.zip')}")
    print("============================================================")

    # ------------------------------------------------------------
    # Gerando gráficos
    # ------------------------------------------------------------
    generate_all_plots()

    # ------------------------------------------------------------
    # Fecha os ambientes
    # ------------------------------------------------------------
    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()