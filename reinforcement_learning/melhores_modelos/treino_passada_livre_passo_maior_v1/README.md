# treino_passada_livre_passo_maior_v1

- Registrado em: `2026-09-15T14:08:55.984525+00:00`
- Origem: `reinforcement_learning/saida_optuna_atom/treino_passada_livre_passo_maior_v1`
- Device: `cuda`
- Timesteps finais: `500000`
- Frequencia de avaliacao: `50000`
- Episodios por avaliacao: `5`

## Modelos

- Melhor avaliacao: `reinforcement_learning/melhores_modelos/treino_passada_livre_passo_maior_v1/best_model.zip`
- Modelo final: `reinforcement_learning/melhores_modelos/treino_passada_livre_passo_maior_v1/final_model.zip`

## Graficos

![recompensa_treino](graficos/recompensa_treino.png)
![convergencia_avaliacao](graficos/convergencia_avaliacao.png)
![recompensa_episodios_avaliacao](graficos/recompensa_episodios_avaliacao.png)

## Ver no MuJoCo

```bash
python reinforcement_learning/ver_modelo_atom.py --modelo reinforcement_learning/melhores_modelos/treino_passada_livre_passo_maior_v1/best_model.zip
```

## TensorBoard

```bash
tensorboard --logdir reinforcement_learning/saida_optuna_atom/treino_passada_livre_passo_maior_v1/logs
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
    "episodes": 1887,
    "last_reward": -528.391744,
    "best_episode_reward": 1338.349572,
    "mean_last_10_reward": -299.2673368,
    "last_episode_length": 190,
    "elapsed_seconds": 551.618274
  },
  "eval_monitor": {
    "episodes": 50,
    "last_reward": -521.479793,
    "best_episode_reward": 83.829498,
    "mean_last_10_reward": -421.92794699999996,
    "last_episode_length": 171,
    "elapsed_seconds": 552.116813
  },
  "evaluations_npz": {
    "path": "reinforcement_learning/saida_optuna_atom/treino_passada_livre_passo_maior_v1/logs/evaluations.npz",
    "num_evaluations": 10,
    "best_mean_reward": 83.829498,
    "best_timestep": 400000,
    "last_mean_reward": -521.479793,
    "last_timestep": 800000
  }
}
```
