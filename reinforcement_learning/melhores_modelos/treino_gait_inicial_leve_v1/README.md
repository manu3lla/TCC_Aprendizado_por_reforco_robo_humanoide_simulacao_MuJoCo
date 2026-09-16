# treino_gait_inicial_leve_v1

- Registrado em: `2026-09-15T18:25:29.339930+00:00`
- Origem: `reinforcement_learning/saida_optuna_atom/treino_gait_inicial_leve_v1`
- Device: `cuda`
- Timesteps finais: `500000`
- Frequencia de avaliacao: `50000`
- Episodios por avaliacao: `5`
- Gait inicial: `True`

## Modelos

- Melhor avaliacao: `reinforcement_learning/melhores_modelos/treino_gait_inicial_leve_v1/best_model.zip`
- Modelo final: `reinforcement_learning/melhores_modelos/treino_gait_inicial_leve_v1/final_model.zip`

## Graficos

![recompensa_treino](graficos/recompensa_treino.png)
![convergencia_avaliacao](graficos/convergencia_avaliacao.png)
![recompensa_episodios_avaliacao](graficos/recompensa_episodios_avaliacao.png)

## Ver no MuJoCo

```bash
python reinforcement_learning/ver_modelo_atom.py --modelo reinforcement_learning/melhores_modelos/treino_gait_inicial_leve_v1/best_model.zip --usar-gait-inicial
```

## TensorBoard

```bash
tensorboard --logdir reinforcement_learning/saida_optuna_atom/treino_gait_inicial_leve_v1/logs
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
    "episodes": 1994,
    "last_reward": -337.361673,
    "best_episode_reward": 745.961476,
    "mean_last_10_reward": -599.4276764,
    "last_episode_length": 117,
    "elapsed_seconds": 571.214501
  },
  "eval_monitor": {
    "episodes": 50,
    "last_reward": -694.880632,
    "best_episode_reward": -159.927892,
    "mean_last_10_reward": -493.38120749999996,
    "last_episode_length": 215,
    "elapsed_seconds": 570.85261
  },
  "evaluations_npz": {
    "path": "reinforcement_learning/saida_optuna_atom/treino_gait_inicial_leve_v1/logs/evaluations.npz",
    "num_evaluations": 10,
    "best_mean_reward": -159.927892,
    "best_timestep": 900000,
    "last_mean_reward": -694.880632,
    "last_timestep": 1150000
  }
}
```
