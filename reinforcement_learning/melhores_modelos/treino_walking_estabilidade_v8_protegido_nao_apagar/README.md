# treino_walking_estabilidade_v8_PROTEGIDO_NAO_APAGAR

- Registrado em: `2026-10-28T17:41:28.752177+00:00`
- Origem: `reinforcement_learning/saida_optuna_atom/treino_walking_estabilidade_v8`
- Device: `cuda`
- Timesteps finais: `None`
- Frequencia de avaliacao: `None`
- Episodios por avaliacao: `None`
- Gait inicial: `True`
- Modo gait parado: `False`
- Modo gait andando: `True`

## Modelos

- Melhor avaliacao: `reinforcement_learning/melhores_modelos/treino_walking_estabilidade_v8_protegido_nao_apagar/best_model.zip`

## Snapshot do ambiente

- Ambiente carregavel: `reinforcement_learning/melhores_modelos/treino_walking_estabilidade_v8_protegido_nao_apagar/env_snapshot/atom/env.py`
- Ambiente documentado: `reinforcement_learning/melhores_modelos/treino_walking_estabilidade_v8_protegido_nao_apagar/env_snapshot/env_usado_no_treino.py`
- Funcao de recompensa: `reinforcement_learning/melhores_modelos/treino_walking_estabilidade_v8_protegido_nao_apagar/env_snapshot/FUNCAO_RECOMPENSA_EXATA_USADA.md`
- Manifesto: `reinforcement_learning/melhores_modelos/treino_walking_estabilidade_v8_protegido_nao_apagar/env_snapshot/manifest.json`

## Graficos

![recompensa_treino](graficos/recompensa_treino.png)
![convergencia_avaliacao](graficos/convergencia_avaliacao.png)
![recompensa_episodios_avaliacao](graficos/recompensa_episodios_avaliacao.png)

## Ver no MuJoCo

```bash
python reinforcement_learning/ver_modelo_atom.py --modelo reinforcement_learning/melhores_modelos/treino_walking_estabilidade_v8_protegido_nao_apagar/best_model.zip --usar-env-snapshot --usar-gait-inicial --modo-gait-andando
```

## TensorBoard

```bash
tensorboard --logdir /home/rafaelaaltheman/Projetos/TCC_Aprendizado_por_reforco_robo_humanoide_simulacao_MuJoCo/reinforcement_learning/melhores_modelos/treino_walking_estabilidade_v8_protegido_nao_apagar/logs
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
  "ent_coef": 0.0,
  "vf_coef": 0.32937558580716303,
  "max_grad_norm": 0.5222902926081441
}
```

## Avaliacao

```json
{
  "train_monitor": {
    "episodes": 429,
    "last_reward": 24097.677188,
    "best_episode_reward": 28189.148846,
    "mean_last_10_reward": 21312.8310138,
    "last_episode_length": 1999,
    "elapsed_seconds": 3396.84053
  },
  "eval_monitor": {
    "episodes": 180,
    "last_reward": 23906.188121,
    "best_episode_reward": 27244.74434,
    "mean_last_10_reward": 23926.089038,
    "last_episode_length": 2166,
    "elapsed_seconds": 3347.041422
  },
  "evaluations_npz": {
    "path": "/home/rafaelaaltheman/Projetos/TCC_Aprendizado_por_reforco_robo_humanoide_simulacao_MuJoCo/reinforcement_learning/melhores_modelos/treino_walking_estabilidade_v8_protegido_nao_apagar/logs/evaluations.npz",
    "num_evaluations": 36,
    "best_mean_reward": 27244.744339999997,
    "best_timestep": 17150000,
    "last_mean_reward": 23906.188121,
    "last_timestep": 17650000
  }
}
```
