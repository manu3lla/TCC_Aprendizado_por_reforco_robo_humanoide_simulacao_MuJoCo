# treino_cuda_5m

- Registrado em: `2026-09-13T23:03:25.351430+00:00`
- Origem: `reinforcement_learning/saida_optuna_atom/treino_cuda_5m`
- Device: `cuda`
- Timesteps finais: `2500000`
- Frequencia de avaliacao: `50000`
- Episodios por avaliacao: `5`

## Modelos

- Melhor avaliacao: `reinforcement_learning/melhores_modelos/treino_cuda_5m/best_model.zip`
- Modelo final: `reinforcement_learning/melhores_modelos/treino_cuda_5m/final_model.zip`

## Graficos

![recompensa_treino](graficos/recompensa_treino.png)
![convergencia_avaliacao](graficos/convergencia_avaliacao.png)
![recompensa_episodios_avaliacao](graficos/recompensa_episodios_avaliacao.png)

## Ver no MuJoCo

```bash
python reinforcement_learning/ver_modelo_atom.py --modelo reinforcement_learning/melhores_modelos/treino_cuda_5m/best_model.zip
```

## TensorBoard

```bash
tensorboard --logdir reinforcement_learning/saida_optuna_atom/treino_cuda_5m/logs
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
    "episodes": 13847,
    "last_reward": 71.862676,
    "best_episode_reward": 690.590568,
    "mean_last_10_reward": 42.9123415,
    "last_episode_length": 140,
    "elapsed_seconds": 542820.060064
  },
  "eval_monitor": {
    "episodes": 250,
    "last_reward": 179.390428,
    "best_episode_reward": 294.698553,
    "mean_last_10_reward": 126.0173465,
    "last_episode_length": 160,
    "elapsed_seconds": 70651.835274
  },
  "evaluations_npz": {
    "path": "reinforcement_learning/saida_optuna_atom/treino_cuda_5m/logs/evaluations.npz",
    "num_evaluations": 50,
    "best_mean_reward": 294.698553,
    "best_timestep": 200000,
    "last_mean_reward": 179.390428,
    "last_timestep": 2500000
  }
}
```
