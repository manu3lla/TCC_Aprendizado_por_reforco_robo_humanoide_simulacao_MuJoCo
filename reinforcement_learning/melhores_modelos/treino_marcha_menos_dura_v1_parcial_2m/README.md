# treino_marcha_menos_dura_v1_parcial_2m

- Registrado em: `2026-09-14T14:35:41.519747+00:00`
- Origem: `reinforcement_learning/saida_optuna_atom/treino_marcha_menos_dura_v1`
- Device: `cuda`
- Timesteps finais: `3000000`
- Frequencia de avaliacao: `50000`
- Episodios por avaliacao: `5`

## Modelos

- Melhor avaliacao: `reinforcement_learning/melhores_modelos/treino_marcha_menos_dura_v1_parcial_2m/best_model.zip`

## Graficos

![recompensa_treino](graficos/recompensa_treino.png)
![convergencia_avaliacao](graficos/convergencia_avaliacao.png)
![recompensa_episodios_avaliacao](graficos/recompensa_episodios_avaliacao.png)

## Ver no MuJoCo

```bash
python reinforcement_learning/ver_modelo_atom.py --modelo reinforcement_learning/melhores_modelos/treino_marcha_menos_dura_v1_parcial_2m/best_model.zip
```

## TensorBoard

```bash
tensorboard --logdir reinforcement_learning/saida_optuna_atom/treino_marcha_menos_dura_v1/logs
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
    "episodes": 12310,
    "last_reward": 188.093012,
    "best_episode_reward": 1718.269942,
    "mean_last_10_reward": 90.3144242,
    "last_episode_length": 364,
    "elapsed_seconds": 2375.929266
  },
  "eval_monitor": {
    "episodes": 215,
    "last_reward": 127.760293,
    "best_episode_reward": 348.484702,
    "mean_last_10_reward": -65.506451,
    "last_episode_length": 139,
    "elapsed_seconds": 2331.572517
  },
  "evaluations_npz": {
    "path": "reinforcement_learning/saida_optuna_atom/treino_marcha_menos_dura_v1/logs/evaluations.npz",
    "num_evaluations": 43,
    "best_mean_reward": 348.484702,
    "best_timestep": 500000,
    "last_mean_reward": 127.760293,
    "last_timestep": 2150000
  }
}
```
