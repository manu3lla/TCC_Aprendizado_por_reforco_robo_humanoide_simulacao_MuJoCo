# treino_reward_estavel_5m

- Registrado em: `2026-09-14T00:38:15.643028+00:00`
- Origem: `reinforcement_learning/saida_optuna_atom/treino_reward_estavel_5m`
- Device: `cuda`
- Timesteps finais: `5000000`
- Frequencia de avaliacao: `50000`
- Episodios por avaliacao: `5`

## Modelos

- Melhor avaliacao: `reinforcement_learning/melhores_modelos/treino_reward_estavel_5m/best_model.zip`
- Modelo final: `reinforcement_learning/melhores_modelos/treino_reward_estavel_5m/final_model.zip`

## Graficos

![recompensa_treino](graficos/recompensa_treino.png)
![convergencia_avaliacao](graficos/convergencia_avaliacao.png)
![recompensa_episodios_avaliacao](graficos/recompensa_episodios_avaliacao.png)

## Ver no MuJoCo

```bash
python reinforcement_learning/ver_modelo_atom.py --modelo reinforcement_learning/melhores_modelos/treino_reward_estavel_5m/best_model.zip
```

## TensorBoard

```bash
tensorboard --logdir reinforcement_learning/saida_optuna_atom/treino_reward_estavel_5m/logs
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
    "episodes": 17060,
    "last_reward": -292.60106,
    "best_episode_reward": 694.383061,
    "mean_last_10_reward": -183.97301190000002,
    "last_episode_length": 284,
    "elapsed_seconds": 5627.238441
  },
  "eval_monitor": {
    "episodes": 500,
    "last_reward": -245.974423,
    "best_episode_reward": 645.619586,
    "mean_last_10_reward": -235.449148,
    "last_episode_length": 180,
    "elapsed_seconds": 5626.892884
  },
  "evaluations_npz": {
    "path": "reinforcement_learning/saida_optuna_atom/treino_reward_estavel_5m/logs/evaluations.npz",
    "num_evaluations": 100,
    "best_mean_reward": 645.619586,
    "best_timestep": 350000,
    "last_mean_reward": -245.97442300000003,
    "last_timestep": 5000000
  }
}
```
