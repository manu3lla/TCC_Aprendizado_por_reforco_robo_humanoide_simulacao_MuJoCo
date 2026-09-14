# treino_joelho_obrigatorio_v2

- Registrado em: `2026-09-14T13:50:52.218076+00:00`
- Origem: `reinforcement_learning/saida_optuna_atom/treino_joelho_obrigatorio_v2`
- Device: `cuda`
- Timesteps finais: `1000000`
- Frequencia de avaliacao: `50000`
- Episodios por avaliacao: `5`

## Modelos

- Melhor avaliacao: `reinforcement_learning/melhores_modelos/treino_joelho_obrigatorio_v2/best_model.zip`
- Modelo final: `reinforcement_learning/melhores_modelos/treino_joelho_obrigatorio_v2/final_model.zip`

## Graficos

![recompensa_treino](graficos/recompensa_treino.png)
![convergencia_avaliacao](graficos/convergencia_avaliacao.png)
![recompensa_episodios_avaliacao](graficos/recompensa_episodios_avaliacao.png)

## Ver no MuJoCo

```bash
python reinforcement_learning/ver_modelo_atom.py --modelo reinforcement_learning/melhores_modelos/treino_joelho_obrigatorio_v2/best_model.zip
```

## TensorBoard

```bash
tensorboard --logdir reinforcement_learning/saida_optuna_atom/treino_joelho_obrigatorio_v2/logs
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
    "episodes": 5470,
    "last_reward": -325.174525,
    "best_episode_reward": 958.401671,
    "mean_last_10_reward": 56.1115053,
    "last_episode_length": 197,
    "elapsed_seconds": 953.369948
  },
  "eval_monitor": {
    "episodes": 100,
    "last_reward": -22.718065,
    "best_episode_reward": 498.564615,
    "mean_last_10_reward": 100.71866800000001,
    "last_episode_length": 135,
    "elapsed_seconds": 952.987183
  },
  "evaluations_npz": {
    "path": "reinforcement_learning/saida_optuna_atom/treino_joelho_obrigatorio_v2/logs/evaluations.npz",
    "num_evaluations": 20,
    "best_mean_reward": 498.56461500000006,
    "best_timestep": 550000,
    "last_mean_reward": -22.718065,
    "last_timestep": 1000000
  }
}
```
