# treino_gait_andando_reto_rapido_target_estavel_v1

- Registrado em: `2026-09-16T12:57:11.500676+00:00`
- Origem: `reinforcement_learning/saida_optuna_atom/treino_gait_andando_reto_rapido_target_estavel_v1`
- Device: `cuda`
- Timesteps finais: `1000000`
- Frequencia de avaliacao: `50000`
- Episodios por avaliacao: `5`
- Gait inicial: `True`
- Modo gait parado: `False`
- Modo gait andando: `True`

## Modelos

- Melhor avaliacao: `reinforcement_learning/melhores_modelos/treino_gait_andando_reto_rapido_target_estavel_v1/best_model.zip`
- Modelo final: `reinforcement_learning/melhores_modelos/treino_gait_andando_reto_rapido_target_estavel_v1/final_model.zip`

## Graficos

![recompensa_treino](graficos/recompensa_treino.png)
![convergencia_avaliacao](graficos/convergencia_avaliacao.png)
![recompensa_episodios_avaliacao](graficos/recompensa_episodios_avaliacao.png)

## Ver no MuJoCo

```bash
python reinforcement_learning/ver_modelo_atom.py --modelo reinforcement_learning/melhores_modelos/treino_gait_andando_reto_rapido_target_estavel_v1/best_model.zip --usar-gait-inicial --modo-gait-andando
```

## TensorBoard

```bash
tensorboard --logdir reinforcement_learning/saida_optuna_atom/treino_gait_andando_reto_rapido_target_estavel_v1/logs
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
    "episodes": 2508,
    "last_reward": 916.734239,
    "best_episode_reward": 17543.765023,
    "mean_last_10_reward": 451.78547149999997,
    "last_episode_length": 213,
    "elapsed_seconds": 1379.825015
  },
  "eval_monitor": {
    "episodes": 100,
    "last_reward": 464.797015,
    "best_episode_reward": 12610.426022,
    "mean_last_10_reward": 611.4137995,
    "last_episode_length": 110,
    "elapsed_seconds": 1379.287009
  },
  "evaluations_npz": {
    "path": "reinforcement_learning/saida_optuna_atom/treino_gait_andando_reto_rapido_target_estavel_v1/logs/evaluations.npz",
    "num_evaluations": 20,
    "best_mean_reward": 12610.426022,
    "best_timestep": 1100000,
    "last_mean_reward": 464.797015,
    "last_timestep": 2050000
  }
}
```
