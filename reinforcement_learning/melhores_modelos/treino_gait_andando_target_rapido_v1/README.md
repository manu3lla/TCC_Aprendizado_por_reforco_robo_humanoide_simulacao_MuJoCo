# treino_gait_andando_target_rapido_v1

- Registrado em: `2026-09-16T11:58:42.422381+00:00`
- Origem: `reinforcement_learning/saida_optuna_atom/treino_gait_andando_target_rapido_v1`
- Device: `cuda`
- Timesteps finais: `1500000`
- Frequencia de avaliacao: `50000`
- Episodios por avaliacao: `5`
- Gait inicial: `True`
- Modo gait parado: `False`
- Modo gait andando: `True`

## Modelos

- Melhor avaliacao: `reinforcement_learning/melhores_modelos/treino_gait_andando_target_rapido_v1/best_model.zip`
- Modelo final: `reinforcement_learning/melhores_modelos/treino_gait_andando_target_rapido_v1/final_model.zip`

## Graficos

![recompensa_treino](graficos/recompensa_treino.png)
![convergencia_avaliacao](graficos/convergencia_avaliacao.png)
![recompensa_episodios_avaliacao](graficos/recompensa_episodios_avaliacao.png)

## Ver no MuJoCo

```bash
python reinforcement_learning/ver_modelo_atom.py --modelo reinforcement_learning/melhores_modelos/treino_gait_andando_target_rapido_v1/best_model.zip --usar-gait-inicial --modo-gait-andando
```

## TensorBoard

```bash
tensorboard --logdir reinforcement_learning/saida_optuna_atom/treino_gait_andando_target_rapido_v1/logs
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
    "episodes": 3508,
    "last_reward": 1026.763264,
    "best_episode_reward": 16380.398289,
    "mean_last_10_reward": 804.4257193,
    "last_episode_length": 839,
    "elapsed_seconds": 2138.932122
  },
  "eval_monitor": {
    "episodes": 150,
    "last_reward": 837.840322,
    "best_episode_reward": 10669.447544,
    "mean_last_10_reward": -47.15283449999998,
    "last_episode_length": 221,
    "elapsed_seconds": 2140.235873
  },
  "evaluations_npz": {
    "path": "reinforcement_learning/saida_optuna_atom/treino_gait_andando_target_rapido_v1/logs/evaluations.npz",
    "num_evaluations": 30,
    "best_mean_reward": 10669.447544,
    "best_timestep": 1000000,
    "last_mean_reward": 837.840322,
    "last_timestep": 2150000
  }
}
```
