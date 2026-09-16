# treino_passo_maior_corpo_sobre_pes_v1

- Registrado em: `2026-09-15T15:13:58.364379+00:00`
- Origem: `reinforcement_learning/saida_optuna_atom/treino_passo_maior_corpo_sobre_pes_v1`
- Device: `cuda`
- Timesteps finais: `1000000`
- Frequencia de avaliacao: `50000`
- Episodios por avaliacao: `5`

## Modelos

- Melhor avaliacao: `reinforcement_learning/melhores_modelos/treino_passo_maior_corpo_sobre_pes_v1/best_model.zip`
- Modelo final: `reinforcement_learning/melhores_modelos/treino_passo_maior_corpo_sobre_pes_v1/final_model.zip`

## Graficos

![recompensa_treino](graficos/recompensa_treino.png)
![convergencia_avaliacao](graficos/convergencia_avaliacao.png)
![recompensa_episodios_avaliacao](graficos/recompensa_episodios_avaliacao.png)

## Ver no MuJoCo

```bash
python reinforcement_learning/ver_modelo_atom.py --modelo reinforcement_learning/melhores_modelos/treino_passo_maior_corpo_sobre_pes_v1/best_model.zip
```

## TensorBoard

```bash
tensorboard --logdir reinforcement_learning/saida_optuna_atom/treino_passo_maior_corpo_sobre_pes_v1/logs
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
    "episodes": 2395,
    "last_reward": -866.365272,
    "best_episode_reward": 1007.076738,
    "mean_last_10_reward": -825.3224236,
    "last_episode_length": 322,
    "elapsed_seconds": 1189.134415
  },
  "eval_monitor": {
    "episodes": 100,
    "last_reward": -962.749913,
    "best_episode_reward": 390.619216,
    "mean_last_10_reward": -877.4981015,
    "last_episode_length": 350,
    "elapsed_seconds": 1188.741341
  },
  "evaluations_npz": {
    "path": "reinforcement_learning/saida_optuna_atom/treino_passo_maior_corpo_sobre_pes_v1/logs/evaluations.npz",
    "num_evaluations": 20,
    "best_mean_reward": 390.619216,
    "best_timestep": 400000,
    "last_mean_reward": -962.749913,
    "last_timestep": 1300000
  }
}
```
