# treino_walking_suavidade_isolada_v10

- Registrado em: `2026-10-28T19:51:54.861483+00:00`
- Origem: `reinforcement_learning/saida_optuna_atom/treino_walking_suavidade_isolada_v10`
- Device: `cuda`
- Timesteps finais: `1000000`
- Frequencia de avaliacao: `25000`
- Episodios por avaliacao: `5`
- Gait inicial: `True`
- Modo gait parado: `False`
- Modo gait andando: `True`

## Modelos

- Melhor avaliacao: `reinforcement_learning/melhores_modelos/treino_walking_suavidade_isolada_v10/best_model.zip`
- Modelo final: `reinforcement_learning/melhores_modelos/treino_walking_suavidade_isolada_v10/final_model.zip`

## Snapshot do ambiente

- Ambiente carregavel: `reinforcement_learning/melhores_modelos/treino_walking_suavidade_isolada_v10/env_snapshot/atom/env.py`
- Ambiente documentado: `reinforcement_learning/melhores_modelos/treino_walking_suavidade_isolada_v10/env_snapshot/env_usado_no_treino.py`
- Funcao de recompensa: `reinforcement_learning/melhores_modelos/treino_walking_suavidade_isolada_v10/env_snapshot/FUNCAO_RECOMPENSA_EXATA_USADA.md`
- Manifesto: `reinforcement_learning/melhores_modelos/treino_walking_suavidade_isolada_v10/env_snapshot/manifest.json`

## Graficos

![recompensa_treino](graficos/recompensa_treino.png)
![convergencia_avaliacao](graficos/convergencia_avaliacao.png)
![recompensa_episodios_avaliacao](graficos/recompensa_episodios_avaliacao.png)

## Ver no MuJoCo

```bash
python reinforcement_learning/ver_modelo_atom.py --modelo reinforcement_learning/melhores_modelos/treino_walking_suavidade_isolada_v10/best_model.zip --usar-env-snapshot --usar-gait-inicial --modo-gait-andando
```

## TensorBoard

```bash
tensorboard --logdir reinforcement_learning/saida_optuna_atom/treino_walking_suavidade_isolada_v10/logs
```

## Parametros PPO

```json
{
  "learning_rate": 5e-06,
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
    "episodes": 461,
    "last_reward": 20591.797885,
    "best_episode_reward": 28028.345667,
    "mean_last_10_reward": 21382.0097054,
    "last_episode_length": 2344,
    "elapsed_seconds": 3219.776956
  },
  "eval_monitor": {
    "episodes": 200,
    "last_reward": 17641.122305,
    "best_episode_reward": 27067.390932,
    "mean_last_10_reward": 21595.431046499998,
    "last_episode_length": 2493,
    "elapsed_seconds": 3242.184409
  },
  "evaluations_npz": {
    "path": "reinforcement_learning/saida_optuna_atom/treino_walking_suavidade_isolada_v10/logs/evaluations.npz",
    "num_evaluations": 40,
    "best_mean_reward": 27067.390932,
    "best_timestep": 17575000,
    "last_mean_reward": 17641.122305,
    "last_timestep": 18150000
  }
}
```
