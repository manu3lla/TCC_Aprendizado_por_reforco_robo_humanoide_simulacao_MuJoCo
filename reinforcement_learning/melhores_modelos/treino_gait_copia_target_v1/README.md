# treino_gait_copia_target_v1

- Registrado em: `2026-09-16T13:30:10.121158+00:00`
- Origem: `reinforcement_learning/saida_optuna_atom/treino_gait_copia_target_v1`
- Device: `cuda`
- Timesteps finais: `1000000`
- Frequencia de avaliacao: `50000`
- Episodios por avaliacao: `5`
- Gait inicial: `True`
- Modo gait parado: `False`
- Modo gait andando: `True`

## Modelos

- Melhor avaliacao: `reinforcement_learning/melhores_modelos/treino_gait_copia_target_v1/best_model.zip`
- Modelo final: `reinforcement_learning/melhores_modelos/treino_gait_copia_target_v1/final_model.zip`

## Graficos

![recompensa_treino](graficos/recompensa_treino.png)
![convergencia_avaliacao](graficos/convergencia_avaliacao.png)
![recompensa_episodios_avaliacao](graficos/recompensa_episodios_avaliacao.png)

## Ver no MuJoCo

```bash
python reinforcement_learning/ver_modelo_atom.py --modelo reinforcement_learning/melhores_modelos/treino_gait_copia_target_v1/best_model.zip --usar-gait-inicial --modo-gait-andando
```

## TensorBoard

```bash
tensorboard --logdir reinforcement_learning/saida_optuna_atom/treino_gait_copia_target_v1/logs
```

## Parametros PPO

```json
{
  "learning_rate": 0.0008286946154789709,
  "n_steps": 512,
  "batch_size": 64,
  "n_epochs": 10,
  "gamma": 0.98,
  "gae_lambda": 0.95,
  "clip_range": 0.137826935848043,
  "ent_coef": 0.0,
  "vf_coef": 0.32937558580716303,
  "max_grad_norm": 0.5222902926081441
}
```

## Avaliacao

```json
{
  "train_monitor": {
    "episodes": 2643,
    "last_reward": 2184.86607,
    "best_episode_reward": 16181.13679,
    "mean_last_10_reward": 933.6135836,
    "last_episode_length": 325,
    "elapsed_seconds": 1231.98156
  },
  "eval_monitor": {
    "episodes": 100,
    "last_reward": 1846.95367,
    "best_episode_reward": 8841.104305,
    "mean_last_10_reward": 2079.583047,
    "last_episode_length": 287,
    "elapsed_seconds": 1233.450521
  },
  "evaluations_npz": {
    "path": "reinforcement_learning/saida_optuna_atom/treino_gait_copia_target_v1/logs/evaluations.npz",
    "num_evaluations": 20,
    "best_mean_reward": 8841.104305,
    "best_timestep": 700000,
    "last_mean_reward": 1846.95367,
    "last_timestep": 1650000
  }
}
```
