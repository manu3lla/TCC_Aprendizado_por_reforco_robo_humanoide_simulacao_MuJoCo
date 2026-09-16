# treino_gait_andando_v3_frente_certa

- Registrado em: `2026-09-16T02:32:27.978589+00:00`
- Origem: `reinforcement_learning/saida_optuna_atom/treino_gait_andando_v3_frente_certa`
- Device: `cuda`
- Timesteps finais: `2500000`
- Frequencia de avaliacao: `50000`
- Episodios por avaliacao: `5`
- Gait inicial: `True`
- Modo gait parado: `False`
- Modo gait andando: `True`

## Modelos

- Melhor avaliacao: `reinforcement_learning/melhores_modelos/treino_gait_andando_v3_frente_certa/best_model.zip`
- Modelo final: `reinforcement_learning/melhores_modelos/treino_gait_andando_v3_frente_certa/final_model.zip`

## Graficos

![recompensa_treino](graficos/recompensa_treino.png)
![convergencia_avaliacao](graficos/convergencia_avaliacao.png)
![recompensa_episodios_avaliacao](graficos/recompensa_episodios_avaliacao.png)

## Ver no MuJoCo

```bash
python reinforcement_learning/ver_modelo_atom.py --modelo reinforcement_learning/melhores_modelos/treino_gait_andando_v3_frente_certa/best_model.zip --usar-gait-inicial --modo-gait-andando
```

## TensorBoard

```bash
tensorboard --logdir reinforcement_learning/saida_optuna_atom/treino_gait_andando_v3_frente_certa/logs
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
    "episodes": 7672,
    "last_reward": 1340.331197,
    "best_episode_reward": 8705.131828,
    "mean_last_10_reward": 1307.5591379000002,
    "last_episode_length": 552,
    "elapsed_seconds": 2991.239798
  },
  "eval_monitor": {
    "episodes": 250,
    "last_reward": 616.167193,
    "best_episode_reward": 6666.861145,
    "mean_last_10_reward": 265.960333,
    "last_episode_length": 145,
    "elapsed_seconds": 2991.781786
  },
  "evaluations_npz": {
    "path": "reinforcement_learning/saida_optuna_atom/treino_gait_andando_v3_frente_certa/logs/evaluations.npz",
    "num_evaluations": 50,
    "best_mean_reward": 6666.861145,
    "best_timestep": 650000,
    "last_mean_reward": 616.167193,
    "last_timestep": 3000000
  }
}
```
