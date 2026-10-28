# treino_walking_reto_target_60s_v2

- Registrado em: `2026-10-27T19:38:51.580863+00:00`
- Origem: `reinforcement_learning/saida_optuna_atom/treino_walking_reto_target_60s_v2`
- Device: `cuda`
- Timesteps finais: `1000000`
- Frequencia de avaliacao: `50000`
- Episodios por avaliacao: `5`
- Gait inicial: `True`
- Modo gait parado: `False`
- Modo gait andando: `True`

## Modelos

- Melhor avaliacao: `reinforcement_learning/melhores_modelos/treino_walking_reto_target_60s_v2/best_model.zip`
- Modelo final: `reinforcement_learning/melhores_modelos/treino_walking_reto_target_60s_v2/final_model.zip`

## Snapshot do ambiente

- Ambiente carregavel: `reinforcement_learning/melhores_modelos/treino_walking_reto_target_60s_v2/env_snapshot/atom/env.py`
- Ambiente documentado: `reinforcement_learning/melhores_modelos/treino_walking_reto_target_60s_v2/env_snapshot/env_usado_no_treino.py`
- Funcao de recompensa: `reinforcement_learning/melhores_modelos/treino_walking_reto_target_60s_v2/env_snapshot/FUNCAO_RECOMPENSA_EXATA_USADA.md`
- Manifesto: `reinforcement_learning/melhores_modelos/treino_walking_reto_target_60s_v2/env_snapshot/manifest.json`

## Graficos

![recompensa_treino](graficos/recompensa_treino.png)
![convergencia_avaliacao](graficos/convergencia_avaliacao.png)
![recompensa_episodios_avaliacao](graficos/recompensa_episodios_avaliacao.png)

## Ver no MuJoCo

```bash
python reinforcement_learning/ver_modelo_atom.py --modelo reinforcement_learning/melhores_modelos/treino_walking_reto_target_60s_v2/best_model.zip --usar-env-snapshot --usar-gait-inicial --modo-gait-andando
```

## TensorBoard

```bash
tensorboard --logdir reinforcement_learning/saida_optuna_atom/treino_walking_reto_target_60s_v2/logs
```

## Parametros PPO

```json
{
  "learning_rate": 3e-05,
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
    "episodes": 246,
    "last_reward": -12548.620946,
    "best_episode_reward": 19733.903708,
    "mean_last_10_reward": -204936.57104130002,
    "last_episode_length": 2745,
    "elapsed_seconds": 2167.219876
  },
  "eval_monitor": {
    "episodes": 100,
    "last_reward": -95993.921865,
    "best_episode_reward": 19980.211815,
    "mean_last_10_reward": -338589.547574,
    "last_episode_length": 3401,
    "elapsed_seconds": 2188.494659
  },
  "evaluations_npz": {
    "path": "reinforcement_learning/saida_optuna_atom/treino_walking_reto_target_60s_v2/logs/evaluations.npz",
    "num_evaluations": 20,
    "best_mean_reward": 19980.211815,
    "best_timestep": 2150000,
    "last_mean_reward": -95993.921865,
    "last_timestep": 2800000
  }
}
```
