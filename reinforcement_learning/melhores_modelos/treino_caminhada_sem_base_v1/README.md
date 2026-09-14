# treino_caminhada_sem_base_v1

- Registrado em: `2026-09-14T17:13:55.519863+00:00`
- Origem: `reinforcement_learning/saida_optuna_atom/treino_caminhada_sem_base_v1`
- Device: `cuda`
- Timesteps finais: `3000000`
- Frequencia de avaliacao: `50000`
- Episodios por avaliacao: `5`

## Modelos

- Melhor avaliacao: `reinforcement_learning/melhores_modelos/treino_caminhada_sem_base_v1/best_model.zip`
- Modelo final: `reinforcement_learning/melhores_modelos/treino_caminhada_sem_base_v1/final_model.zip`

## Graficos

![recompensa_treino](graficos/recompensa_treino.png)
![convergencia_avaliacao](graficos/convergencia_avaliacao.png)
![recompensa_episodios_avaliacao](graficos/recompensa_episodios_avaliacao.png)

## Ver no MuJoCo

```bash
python reinforcement_learning/ver_modelo_atom.py --modelo reinforcement_learning/melhores_modelos/treino_caminhada_sem_base_v1/best_model.zip
```

## TensorBoard

```bash
tensorboard --logdir reinforcement_learning/saida_optuna_atom/treino_caminhada_sem_base_v1/logs
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
    "episodes": 11811,
    "last_reward": -567.816446,
    "best_episode_reward": 1911.907373,
    "mean_last_10_reward": -311.523007,
    "last_episode_length": 146,
    "elapsed_seconds": 3553.628066
  },
  "eval_monitor": {
    "episodes": 300,
    "last_reward": -635.68045,
    "best_episode_reward": 190.482498,
    "mean_last_10_reward": -573.1236124999999,
    "last_episode_length": 147,
    "elapsed_seconds": 3553.298196
  },
  "evaluations_npz": {
    "path": "reinforcement_learning/saida_optuna_atom/treino_caminhada_sem_base_v1/logs/evaluations.npz",
    "num_evaluations": 60,
    "best_mean_reward": 190.482498,
    "best_timestep": 300000,
    "last_mean_reward": -635.68045,
    "last_timestep": 3000000
  }
}
```
