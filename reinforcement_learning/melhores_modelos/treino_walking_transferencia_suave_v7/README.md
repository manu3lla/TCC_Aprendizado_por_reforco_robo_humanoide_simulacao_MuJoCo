# treino_walking_transferencia_suave_v7

- Registrado em: `2026-10-28T15:23:09.144486+00:00`
- Origem: `reinforcement_learning/saida_optuna_atom/treino_walking_transferencia_suave_v7`
- Device: `cuda`
- Timesteps finais: `2000000`
- Frequencia de avaliacao: `50000`
- Episodios por avaliacao: `5`
- Gait inicial: `True`
- Modo gait parado: `False`
- Modo gait andando: `True`

## Modelos

- Melhor avaliacao: `reinforcement_learning/melhores_modelos/treino_walking_transferencia_suave_v7/best_model.zip`
- Modelo final: `reinforcement_learning/melhores_modelos/treino_walking_transferencia_suave_v7/final_model.zip`

## Snapshot do ambiente

- Ambiente carregavel: `reinforcement_learning/melhores_modelos/treino_walking_transferencia_suave_v7/env_snapshot/atom/env.py`
- Ambiente documentado: `reinforcement_learning/melhores_modelos/treino_walking_transferencia_suave_v7/env_snapshot/env_usado_no_treino.py`
- Funcao de recompensa: `reinforcement_learning/melhores_modelos/treino_walking_transferencia_suave_v7/env_snapshot/FUNCAO_RECOMPENSA_EXATA_USADA.md`
- Manifesto: `reinforcement_learning/melhores_modelos/treino_walking_transferencia_suave_v7/env_snapshot/manifest.json`

## Graficos

![recompensa_treino](graficos/recompensa_treino.png)
![convergencia_avaliacao](graficos/convergencia_avaliacao.png)
![recompensa_episodios_avaliacao](graficos/recompensa_episodios_avaliacao.png)

## Ver no MuJoCo

```bash
python reinforcement_learning/ver_modelo_atom.py --modelo reinforcement_learning/melhores_modelos/treino_walking_transferencia_suave_v7/best_model.zip --usar-env-snapshot --usar-gait-inicial --modo-gait-andando
```

## TensorBoard

```bash
tensorboard --logdir reinforcement_learning/saida_optuna_atom/treino_walking_transferencia_suave_v7/logs
```

## Parametros PPO

```json
{
  "learning_rate": 1e-05,
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
    "episodes": 931,
    "last_reward": 16870.822704,
    "best_episode_reward": 28893.17382,
    "mean_last_10_reward": 18502.5956695,
    "last_episode_length": 2521,
    "elapsed_seconds": 5412.655612
  },
  "eval_monitor": {
    "episodes": 200,
    "last_reward": 24016.145787,
    "best_episode_reward": 27761.962361,
    "mean_last_10_reward": 22205.213213500003,
    "last_episode_length": 2017,
    "elapsed_seconds": 5430.919432
  },
  "evaluations_npz": {
    "path": "reinforcement_learning/saida_optuna_atom/treino_walking_transferencia_suave_v7/logs/evaluations.npz",
    "num_evaluations": 40,
    "best_mean_reward": 27761.962361,
    "best_timestep": 18400000,
    "last_mean_reward": 24016.145787,
    "last_timestep": 18750000
  }
}
```
