# treino_passada_livre_progresso_v1

- Registrado em: `2026-09-15T13:52:10.930726+00:00`
- Origem: `reinforcement_learning/saida_optuna_atom/treino_passada_livre_progresso_v1`
- Device: `cuda`
- Timesteps finais: `500000`
- Frequencia de avaliacao: `50000`
- Episodios por avaliacao: `5`

## Modelos

- Melhor avaliacao: `reinforcement_learning/melhores_modelos/treino_passada_livre_progresso_v1/best_model.zip`
- Modelo final: `reinforcement_learning/melhores_modelos/treino_passada_livre_progresso_v1/final_model.zip`

## Graficos

![recompensa_treino](graficos/recompensa_treino.png)
![convergencia_avaliacao](graficos/convergencia_avaliacao.png)
![recompensa_episodios_avaliacao](graficos/recompensa_episodios_avaliacao.png)

## Ver no MuJoCo

```bash
python reinforcement_learning/ver_modelo_atom.py --modelo reinforcement_learning/melhores_modelos/treino_passada_livre_progresso_v1/best_model.zip
```

## TensorBoard

```bash
tensorboard --logdir reinforcement_learning/saida_optuna_atom/treino_passada_livre_progresso_v1/logs
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
    "episodes": 1256,
    "last_reward": -598.517307,
    "best_episode_reward": 1590.434248,
    "mean_last_10_reward": -420.84348550000004,
    "last_episode_length": 182,
    "elapsed_seconds": 563.388541
  },
  "eval_monitor": {
    "episodes": 50,
    "last_reward": -118.962289,
    "best_episode_reward": 852.346497,
    "mean_last_10_reward": -151.4769335,
    "last_episode_length": 217,
    "elapsed_seconds": 564.197538
  },
  "evaluations_npz": {
    "path": "reinforcement_learning/saida_optuna_atom/treino_passada_livre_progresso_v1/logs/evaluations.npz",
    "num_evaluations": 10,
    "best_mean_reward": 852.3464970000001,
    "best_timestep": 550000,
    "last_mean_reward": -118.96228900000001,
    "last_timestep": 800000
  }
}
```
