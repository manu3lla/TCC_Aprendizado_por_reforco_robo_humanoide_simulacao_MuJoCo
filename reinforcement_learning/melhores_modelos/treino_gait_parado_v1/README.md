# treino_gait_parado_v1

- Registrado em: `2026-09-16T01:17:52.392888+00:00`
- Origem: `reinforcement_learning/saida_optuna_atom/treino_gait_parado_v1`
- Device: `cuda`
- Timesteps finais: `10000000`
- Frequencia de avaliacao: `50000`
- Episodios por avaliacao: `5`
- Gait inicial: `True`
- Modo gait parado: `True`

## Modelos

- Melhor avaliacao: `reinforcement_learning/melhores_modelos/treino_gait_parado_v1/best_model.zip`
- Modelo final: `reinforcement_learning/melhores_modelos/treino_gait_parado_v1/final_model.zip`

## Graficos

![recompensa_treino](graficos/recompensa_treino.png)
![convergencia_avaliacao](graficos/convergencia_avaliacao.png)
![recompensa_episodios_avaliacao](graficos/recompensa_episodios_avaliacao.png)

## Ver no MuJoCo

```bash
python reinforcement_learning/ver_modelo_atom.py --modelo reinforcement_learning/melhores_modelos/treino_gait_parado_v1/best_model.zip --usar-gait-inicial --modo-gait-parado
```

## TensorBoard

```bash
tensorboard --logdir reinforcement_learning/saida_optuna_atom/treino_gait_parado_v1/logs
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
    "episodes": 19321,
    "last_reward": 6062.487965,
    "best_episode_reward": 8279.757041,
    "mean_last_10_reward": 5570.7176325,
    "last_episode_length": 1000,
    "elapsed_seconds": 12236.765524
  },
  "eval_monitor": {
    "episodes": 1000,
    "last_reward": 2651.934954,
    "best_episode_reward": 7658.298248,
    "mean_last_10_reward": 4541.4484175,
    "last_episode_length": 580,
    "elapsed_seconds": 12239.400533
  },
  "evaluations_npz": {
    "path": "reinforcement_learning/saida_optuna_atom/treino_gait_parado_v1/logs/evaluations.npz",
    "num_evaluations": 200,
    "best_mean_reward": 7658.298248000001,
    "best_timestep": 500000,
    "last_mean_reward": 2651.934954,
    "last_timestep": 10000000
  }
}
```
