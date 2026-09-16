# treino_caminhada_passo_largo_v1

- Registrado em: `2026-09-15T09:56:46.757734+00:00`
- Origem: `reinforcement_learning/saida_optuna_atom/treino_caminhada_passo_largo_v1`
- Device: `cuda`
- Timesteps finais: `30000000`
- Frequencia de avaliacao: `50000`
- Episodios por avaliacao: `5`

## Modelos

- Melhor avaliacao: `reinforcement_learning/melhores_modelos/treino_caminhada_passo_largo_v1/best_model.zip`
- Modelo final: `reinforcement_learning/melhores_modelos/treino_caminhada_passo_largo_v1/final_model.zip`

## Graficos

![recompensa_treino](graficos/recompensa_treino.png)
![convergencia_avaliacao](graficos/convergencia_avaliacao.png)
![recompensa_episodios_avaliacao](graficos/recompensa_episodios_avaliacao.png)

## Ver no MuJoCo

```bash
python reinforcement_learning/ver_modelo_atom.py --modelo reinforcement_learning/melhores_modelos/treino_caminhada_passo_largo_v1/best_model.zip
```

## TensorBoard

```bash
tensorboard --logdir reinforcement_learning/saida_optuna_atom/treino_caminhada_passo_largo_v1/logs
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
    "episodes": 150853,
    "last_reward": -4.463444,
    "best_episode_reward": 1977.815708,
    "mean_last_10_reward": -190.374387,
    "last_episode_length": 258,
    "elapsed_seconds": 33019.782387
  },
  "eval_monitor": {
    "episodes": 3000,
    "last_reward": -489.915475,
    "best_episode_reward": 510.041296,
    "mean_last_10_reward": -421.9732695000001,
    "last_episode_length": 154,
    "elapsed_seconds": 33019.508892
  },
  "evaluations_npz": {
    "path": "reinforcement_learning/saida_optuna_atom/treino_caminhada_passo_largo_v1/logs/evaluations.npz",
    "num_evaluations": 600,
    "best_mean_reward": 510.041296,
    "best_timestep": 28350000,
    "last_mean_reward": -489.91547499999996,
    "last_timestep": 30300000
  }
}
```
