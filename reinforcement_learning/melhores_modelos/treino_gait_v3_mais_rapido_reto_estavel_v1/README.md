# treino_gait_v3_mais_rapido_reto_estavel_v1

- Registrado em: `2026-10-14T00:37:55.596029+00:00`
- Origem: `reinforcement_learning/saida_optuna_atom/treino_gait_v3_mais_rapido_reto_estavel_v1`
- Device: `cuda`
- Timesteps finais: `3000000`
- Frequencia de avaliacao: `50000`
- Episodios por avaliacao: `5`
- Gait inicial: `True`
- Modo gait parado: `False`
- Modo gait andando: `True`

## Modelos

- Melhor avaliacao: `reinforcement_learning/melhores_modelos/treino_gait_v3_mais_rapido_reto_estavel_v1/best_model.zip`
- Modelo final: `reinforcement_learning/melhores_modelos/treino_gait_v3_mais_rapido_reto_estavel_v1/final_model.zip`

## Snapshot do ambiente

- Ambiente carregavel: `reinforcement_learning/melhores_modelos/treino_gait_v3_mais_rapido_reto_estavel_v1/env_snapshot/atom/env.py`
- Ambiente documentado: `reinforcement_learning/melhores_modelos/treino_gait_v3_mais_rapido_reto_estavel_v1/env_snapshot/env_usado_no_treino.py`
- Funcao de recompensa: `reinforcement_learning/melhores_modelos/treino_gait_v3_mais_rapido_reto_estavel_v1/env_snapshot/FUNCAO_RECOMPENSA_EXATA_USADA.md`
- Manifesto: `reinforcement_learning/melhores_modelos/treino_gait_v3_mais_rapido_reto_estavel_v1/env_snapshot/manifest.json`

## Graficos

![recompensa_treino](graficos/recompensa_treino.png)
![convergencia_avaliacao](graficos/convergencia_avaliacao.png)
![recompensa_episodios_avaliacao](graficos/recompensa_episodios_avaliacao.png)

## Ver no MuJoCo

```bash
python reinforcement_learning/ver_modelo_atom.py --modelo reinforcement_learning/melhores_modelos/treino_gait_v3_mais_rapido_reto_estavel_v1/best_model.zip --usar-env-snapshot --usar-gait-inicial --modo-gait-andando
```

## TensorBoard

```bash
tensorboard --logdir reinforcement_learning/saida_optuna_atom/treino_gait_v3_mais_rapido_reto_estavel_v1/logs
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
    "episodes": 12055,
    "last_reward": 880.366098,
    "best_episode_reward": 9912.595273,
    "mean_last_10_reward": 1352.7357354,
    "last_episode_length": 194,
    "elapsed_seconds": 6384.862523
  },
  "eval_monitor": {
    "episodes": 300,
    "last_reward": 950.43463,
    "best_episode_reward": 8667.845076,
    "mean_last_10_reward": 1168.6971545000001,
    "last_episode_length": 204,
    "elapsed_seconds": 6384.042699
  },
  "evaluations_npz": {
    "path": "reinforcement_learning/saida_optuna_atom/treino_gait_v3_mais_rapido_reto_estavel_v1/logs/evaluations.npz",
    "num_evaluations": 60,
    "best_mean_reward": 8667.845076,
    "best_timestep": 1650000,
    "last_mean_reward": 950.43463,
    "last_timestep": 4600000
  }
}
```
