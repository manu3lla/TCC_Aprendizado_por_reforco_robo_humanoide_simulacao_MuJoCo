# treino_gait_v3_rapido_fluido_30m_v2

- Registrado em: `2026-10-12T00:49:21.450049+00:00`
- Origem: `reinforcement_learning/saida_optuna_atom/treino_gait_v3_rapido_fluido_30m_v2`
- Device: `cuda`
- Timesteps finais: `30000000`
- Frequencia de avaliacao: `150000`
- Episodios por avaliacao: `5`
- Gait inicial: `True`
- Modo gait parado: `False`
- Modo gait andando: `True`

## Modelos

- Melhor avaliacao: `reinforcement_learning/melhores_modelos/treino_gait_v3_rapido_fluido_30m_v2/best_model.zip`
- Modelo final: `reinforcement_learning/melhores_modelos/treino_gait_v3_rapido_fluido_30m_v2/final_model.zip`

## Snapshot do ambiente

- Ambiente carregavel: `reinforcement_learning/melhores_modelos/treino_gait_v3_rapido_fluido_30m_v2/env_snapshot/atom/env.py`
- Ambiente documentado: `reinforcement_learning/melhores_modelos/treino_gait_v3_rapido_fluido_30m_v2/env_snapshot/env_usado_no_treino.py`
- Funcao de recompensa: `reinforcement_learning/melhores_modelos/treino_gait_v3_rapido_fluido_30m_v2/env_snapshot/FUNCAO_RECOMPENSA_EXATA_USADA.md`
- Manifesto: `reinforcement_learning/melhores_modelos/treino_gait_v3_rapido_fluido_30m_v2/env_snapshot/manifest.json`

## Graficos

![recompensa_treino](graficos/recompensa_treino.png)
![convergencia_avaliacao](graficos/convergencia_avaliacao.png)
![recompensa_episodios_avaliacao](graficos/recompensa_episodios_avaliacao.png)

## Ver no MuJoCo

```bash
python reinforcement_learning/ver_modelo_atom.py --modelo reinforcement_learning/melhores_modelos/treino_gait_v3_rapido_fluido_30m_v2/best_model.zip --usar-env-snapshot --usar-gait-inicial --modo-gait-andando
```

## TensorBoard

```bash
tensorboard --logdir reinforcement_learning/saida_optuna_atom/treino_gait_v3_rapido_fluido_30m_v2/logs
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
    "episodes": 111765,
    "last_reward": 2064.658333,
    "best_episode_reward": 10873.004741,
    "mean_last_10_reward": 1298.3182875999998,
    "last_episode_length": 374,
    "elapsed_seconds": 45415.527386
  },
  "eval_monitor": {
    "episodes": 1000,
    "last_reward": 949.777234,
    "best_episode_reward": 9074.276222,
    "mean_last_10_reward": 963.3626385,
    "last_episode_length": 194,
    "elapsed_seconds": 45416.393637
  },
  "evaluations_npz": {
    "path": "reinforcement_learning/saida_optuna_atom/treino_gait_v3_rapido_fluido_30m_v2/logs/evaluations.npz",
    "num_evaluations": 200,
    "best_mean_reward": 9074.276222,
    "best_timestep": 11650000,
    "last_mean_reward": 949.7772339999999,
    "last_timestep": 31600000
  }
}
```
