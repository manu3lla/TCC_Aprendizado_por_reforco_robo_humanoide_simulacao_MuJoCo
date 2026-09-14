# treino_caminhada_passo_largo_v1

- Registrado em: `2026-09-14T21:56:41.696195+00:00`
- Origem: `reinforcement_learning/saida_optuna_atom/treino_caminhada_passo_largo_v1`
- Device: `cuda`
- Timesteps finais: `3000000`
- Frequencia de avaliacao: `50000`
- Episodios por avaliacao: `5`

## Modelos

- Melhor avaliacao: `reinforcement_learning/melhores_modelos/treino_caminhada_passo_largo_v1/best_model.zip`
- Modelo final: `reinforcement_learning/melhores_modelos/treino_caminhada_passo_largo_v1/final_model.zip`

## Graficos

![recompensa_treino](graficos/recompensa_treino.png)
![convergencia_avaliacao](graficos/convergencia_avaliacao.png)
![recompensa_episodios_avaliacao](graficos/recompensa_episodios_avaliacao.png)

## Ver no MuJoCo

```bash
python reinforcement_learning/ver_modelo_atom.py --modelo reinforcement_learning/melhores_modelos/treino_caminhada_passo_largo_v1/best_model.zip
```

## TensorBoard

```bash
tensorboard --logdir reinforcement_learning/saida_optuna_atom/treino_caminhada_passo_largo_v1/logs
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
    "episodes": 12831,
    "last_reward": -655.026741,
    "best_episode_reward": 1977.815708,
    "mean_last_10_reward": -509.5342697,
    "last_episode_length": 193,
    "elapsed_seconds": 4321.449373
  },
  "eval_monitor": {
    "episodes": 300,
    "last_reward": -418.992803,
    "best_episode_reward": 256.415618,
    "mean_last_10_reward": -544.3416475,
    "last_episode_length": 223,
    "elapsed_seconds": 4320.94586
  },
  "evaluations_npz": {
    "path": "reinforcement_learning/saida_optuna_atom/treino_caminhada_passo_largo_v1/logs/evaluations.npz",
    "num_evaluations": 60,
    "best_mean_reward": 256.415618,
    "best_timestep": 400000,
    "last_mean_reward": -418.992803,
    "last_timestep": 3300000
  }
}
```
