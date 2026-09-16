# treino_rapido_reto_target_v1

- Registrado em: `2026-09-15T16:42:57.191219+00:00`
- Origem: `reinforcement_learning/saida_optuna_atom/treino_rapido_reto_target_v1`
- Device: `cuda`
- Timesteps finais: `1000000`
- Frequencia de avaliacao: `50000`
- Episodios por avaliacao: `5`

## Modelos

- Melhor avaliacao: `reinforcement_learning/melhores_modelos/treino_rapido_reto_target_v1/best_model.zip`
- Modelo final: `reinforcement_learning/melhores_modelos/treino_rapido_reto_target_v1/final_model.zip`

## Graficos

![recompensa_treino](graficos/recompensa_treino.png)
![convergencia_avaliacao](graficos/convergencia_avaliacao.png)
![recompensa_episodios_avaliacao](graficos/recompensa_episodios_avaliacao.png)

## Ver no MuJoCo

```bash
python reinforcement_learning/ver_modelo_atom.py --modelo reinforcement_learning/melhores_modelos/treino_rapido_reto_target_v1/best_model.zip
```

## TensorBoard

```bash
tensorboard --logdir reinforcement_learning/saida_optuna_atom/treino_rapido_reto_target_v1/logs
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
    "episodes": 3020,
    "last_reward": -989.603426,
    "best_episode_reward": 1800.983784,
    "mean_last_10_reward": -913.2891536999999,
    "last_episode_length": 254,
    "elapsed_seconds": 1215.036833
  },
  "eval_monitor": {
    "episodes": 100,
    "last_reward": -1505.403166,
    "best_episode_reward": 819.718922,
    "mean_last_10_reward": -1232.3714100000002,
    "last_episode_length": 464,
    "elapsed_seconds": 1214.693132
  },
  "evaluations_npz": {
    "path": "reinforcement_learning/saida_optuna_atom/treino_rapido_reto_target_v1/logs/evaluations.npz",
    "num_evaluations": 20,
    "best_mean_reward": 819.718922,
    "best_timestep": 550000,
    "last_mean_reward": -1505.403166,
    "last_timestep": 1450000
  }
}
```
