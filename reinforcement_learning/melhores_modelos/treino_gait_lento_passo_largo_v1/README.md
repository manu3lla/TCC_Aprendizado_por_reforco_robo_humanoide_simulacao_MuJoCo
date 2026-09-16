# treino_gait_lento_passo_largo_v1

- Registrado em: `2026-09-16T19:46:54.853569+00:00`
- Origem: `reinforcement_learning/saida_optuna_atom/treino_gait_lento_passo_largo_v1`
- Device: `cuda`
- Timesteps finais: `10000000`
- Frequencia de avaliacao: `50000`
- Episodios por avaliacao: `5`
- Gait inicial: `True`
- Modo gait parado: `False`
- Modo gait andando: `True`

## Modelos

- Melhor avaliacao: `reinforcement_learning/melhores_modelos/treino_gait_lento_passo_largo_v1/best_model.zip`
- Modelo final: `reinforcement_learning/melhores_modelos/treino_gait_lento_passo_largo_v1/final_model.zip`

## Graficos

![recompensa_treino](graficos/recompensa_treino.png)
![convergencia_avaliacao](graficos/convergencia_avaliacao.png)
![recompensa_episodios_avaliacao](graficos/recompensa_episodios_avaliacao.png)

## Ver no MuJoCo

```bash
python reinforcement_learning/ver_modelo_atom.py --modelo reinforcement_learning/melhores_modelos/treino_gait_lento_passo_largo_v1/best_model.zip --usar-gait-inicial --modo-gait-andando
```

## TensorBoard

```bash
tensorboard --logdir reinforcement_learning/saida_optuna_atom/treino_gait_lento_passo_largo_v1/logs
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
    "episodes": 27255,
    "last_reward": -595.533484,
    "best_episode_reward": 16490.389189,
    "mean_last_10_reward": 3759.8716249,
    "last_episode_length": 343,
    "elapsed_seconds": 12315.45706
  },
  "eval_monitor": {
    "episodes": 1000,
    "last_reward": -1114.806484,
    "best_episode_reward": 15437.696938,
    "mean_last_10_reward": 319.5586035,
    "last_episode_length": 128,
    "elapsed_seconds": 12317.612985
  },
  "evaluations_npz": {
    "path": "reinforcement_learning/saida_optuna_atom/treino_gait_lento_passo_largo_v1/logs/evaluations.npz",
    "num_evaluations": 200,
    "best_mean_reward": 15437.696938,
    "best_timestep": 6750000,
    "last_mean_reward": -1114.806484,
    "last_timestep": 10650000
  }
}
```
