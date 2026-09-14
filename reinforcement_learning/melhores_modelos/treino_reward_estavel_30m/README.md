# treino_reward_estavel_30m

- Registrado em: `2026-09-14T10:40:39.176674+00:00`
- Origem: `reinforcement_learning/saida_optuna_atom/treino_reward_estavel_30m`
- Device: `cuda`
- Timesteps finais: `30000000`
- Frequencia de avaliacao: `150000`
- Episodios por avaliacao: `5`

## Modelos

- Melhor avaliacao: `reinforcement_learning/melhores_modelos/treino_reward_estavel_30m/best_model.zip`
- Modelo final: `reinforcement_learning/melhores_modelos/treino_reward_estavel_30m/final_model.zip`

## Graficos

![recompensa_treino](graficos/recompensa_treino.png)
![convergencia_avaliacao](graficos/convergencia_avaliacao.png)
![recompensa_episodios_avaliacao](graficos/recompensa_episodios_avaliacao.png)

## Ver no MuJoCo

```bash
python reinforcement_learning/ver_modelo_atom.py --modelo reinforcement_learning/melhores_modelos/treino_reward_estavel_30m/best_model.zip
```

## TensorBoard

```bash
tensorboard --logdir reinforcement_learning/saida_optuna_atom/treino_reward_estavel_30m/logs
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
    "episodes": 88188,
    "last_reward": -164.96733,
    "best_episode_reward": 743.151629,
    "mean_last_10_reward": -138.06150029999998,
    "last_episode_length": 130,
    "elapsed_seconds": 33601.346157
  },
  "eval_monitor": {
    "episodes": 1000,
    "last_reward": -111.647915,
    "best_episode_reward": 339.867411,
    "mean_last_10_reward": -118.029909,
    "last_episode_length": 112,
    "elapsed_seconds": 33601.03775
  },
  "evaluations_npz": {
    "path": "reinforcement_learning/saida_optuna_atom/treino_reward_estavel_30m/logs/evaluations.npz",
    "num_evaluations": 200,
    "best_mean_reward": 339.867411,
    "best_timestep": 1950000,
    "last_mean_reward": -111.64791499999998,
    "last_timestep": 30000000
  }
}
```
