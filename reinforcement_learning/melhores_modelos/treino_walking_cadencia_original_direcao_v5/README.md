# treino_walking_cadencia_original_direcao_v5

- Registrado em: `2026-10-27T21:28:23.043499+00:00`
- Origem: `reinforcement_learning/saida_optuna_atom/treino_walking_cadencia_original_direcao_v5`
- Device: `cuda`
- Timesteps finais: `500000`
- Frequencia de avaliacao: `25000`
- Episodios por avaliacao: `5`
- Gait inicial: `True`
- Modo gait parado: `False`
- Modo gait andando: `True`

## Modelos

- Melhor avaliacao: `reinforcement_learning/melhores_modelos/treino_walking_cadencia_original_direcao_v5/best_model.zip`
- Modelo final: `reinforcement_learning/melhores_modelos/treino_walking_cadencia_original_direcao_v5/final_model.zip`

## Snapshot do ambiente

- Ambiente carregavel: `reinforcement_learning/melhores_modelos/treino_walking_cadencia_original_direcao_v5/env_snapshot/atom/env.py`
- Ambiente documentado: `reinforcement_learning/melhores_modelos/treino_walking_cadencia_original_direcao_v5/env_snapshot/env_usado_no_treino.py`
- Funcao de recompensa: `reinforcement_learning/melhores_modelos/treino_walking_cadencia_original_direcao_v5/env_snapshot/FUNCAO_RECOMPENSA_EXATA_USADA.md`
- Manifesto: `reinforcement_learning/melhores_modelos/treino_walking_cadencia_original_direcao_v5/env_snapshot/manifest.json`

## Graficos

![recompensa_treino](graficos/recompensa_treino.png)
![convergencia_avaliacao](graficos/convergencia_avaliacao.png)
![recompensa_episodios_avaliacao](graficos/recompensa_episodios_avaliacao.png)

## Ver no MuJoCo

```bash
python reinforcement_learning/ver_modelo_atom.py --modelo reinforcement_learning/melhores_modelos/treino_walking_cadencia_original_direcao_v5/best_model.zip --usar-env-snapshot --usar-gait-inicial --modo-gait-andando
```

## TensorBoard

```bash
tensorboard --logdir reinforcement_learning/saida_optuna_atom/treino_walking_cadencia_original_direcao_v5/logs
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
    "episodes": 119,
    "last_reward": -154404.70005,
    "best_episode_reward": 21807.536836,
    "mean_last_10_reward": -266368.4455672,
    "last_episode_length": 6000,
    "elapsed_seconds": 1284.170406
  },
  "eval_monitor": {
    "episodes": 100,
    "last_reward": -195462.414461,
    "best_episode_reward": 26537.173561,
    "mean_last_10_reward": -200131.4498225,
    "last_episode_length": 6000,
    "elapsed_seconds": 1321.02925
  },
  "evaluations_npz": {
    "path": "reinforcement_learning/saida_optuna_atom/treino_walking_cadencia_original_direcao_v5/logs/evaluations.npz",
    "num_evaluations": 20,
    "best_mean_reward": 26537.173560999996,
    "best_timestep": 2025000,
    "last_mean_reward": -195462.414461,
    "last_timestep": 2300000
  }
}
```
