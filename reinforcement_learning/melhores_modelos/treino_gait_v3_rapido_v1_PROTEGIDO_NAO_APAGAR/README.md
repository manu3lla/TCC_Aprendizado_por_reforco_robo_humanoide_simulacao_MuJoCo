# treino_gait_v3_rapido_v1_PROTEGIDO_NAO_APAGAR

- Registrado em: `2026-10-10T22:47:59.384296+00:00`
- Origem: `reinforcement_learning/saida_optuna_atom/treino_gait_v3_rapido_v1_PROTEGIDO_NAO_APAGAR`
- Device: `cuda`
- Timesteps finais: `2000000`
- Frequencia de avaliacao: `50000`
- Episodios por avaliacao: `5`
- Gait inicial: `True`
- Modo gait parado: `False`
- Modo gait andando: `True`

## Modelos

- Melhor avaliacao: `reinforcement_learning/melhores_modelos/treino_gait_v3_rapido_v1_PROTEGIDO_NAO_APAGAR_PROTEGIDO_NAO_APAGAR/best_model.zip`
- Modelo final: `reinforcement_learning/melhores_modelos/treino_gait_v3_rapido_v1_PROTEGIDO_NAO_APAGAR_PROTEGIDO_NAO_APAGAR/final_model.zip`

## Snapshot do ambiente

- Ambiente carregavel: `reinforcement_learning/melhores_modelos/treino_gait_v3_rapido_v1_PROTEGIDO_NAO_APAGAR_PROTEGIDO_NAO_APAGAR/env_snapshot/atom/env.py`
- Ambiente documentado: `reinforcement_learning/melhores_modelos/treino_gait_v3_rapido_v1_PROTEGIDO_NAO_APAGAR_PROTEGIDO_NAO_APAGAR/env_snapshot/env_usado_no_treino.py`
- Funcao de recompensa: `reinforcement_learning/melhores_modelos/treino_gait_v3_rapido_v1_PROTEGIDO_NAO_APAGAR_PROTEGIDO_NAO_APAGAR/env_snapshot/FUNCAO_RECOMPENSA_EXATA_USADA.md`
- Manifesto: `reinforcement_learning/melhores_modelos/treino_gait_v3_rapido_v1_PROTEGIDO_NAO_APAGAR_PROTEGIDO_NAO_APAGAR/env_snapshot/manifest.json`

## Graficos

![recompensa_treino](graficos/recompensa_treino.png)
![convergencia_avaliacao](graficos/convergencia_avaliacao.png)
![recompensa_episodios_avaliacao](graficos/recompensa_episodios_avaliacao.png)

## Ver no MuJoCo

```bash
python reinforcement_learning/ver_modelo_atom.py --modelo reinforcement_learning/melhores_modelos/treino_gait_v3_rapido_v1_PROTEGIDO_NAO_APAGAR_PROTEGIDO_NAO_APAGAR/best_model.zip --usar-env-snapshot --usar-gait-inicial --modo-gait-andando
```

## TensorBoard

```bash
tensorboard --logdir reinforcement_learning/saida_optuna_atom/treino_gait_v3_rapido_v1_PROTEGIDO_NAO_APAGAR/logs
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
    "episodes": 4794,
    "last_reward": 1241.609284,
    "best_episode_reward": 10144.043584,
    "mean_last_10_reward": 1990.152893,
    "last_episode_length": 266,
    "elapsed_seconds": 3187.689116
  },
  "eval_monitor": {
    "episodes": 200,
    "last_reward": 6581.579214,
    "best_episode_reward": 9267.07445,
    "mean_last_10_reward": 3926.2423415000003,
    "last_episode_length": 1000,
    "elapsed_seconds": 3192.90608
  },
  "evaluations_npz": {
    "path": "reinforcement_learning/saida_optuna_atom/treino_gait_v3_rapido_v1_PROTEGIDO_NAO_APAGAR/logs/evaluations.npz",
    "num_evaluations": 40,
    "best_mean_reward": 9267.07445,
    "best_timestep": 1600000,
    "last_mean_reward": 6581.579214,
    "last_timestep": 2650000
  }
}
```
