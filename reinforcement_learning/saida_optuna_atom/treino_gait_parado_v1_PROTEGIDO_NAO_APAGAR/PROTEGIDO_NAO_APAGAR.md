# PROTEGIDO - treino_gait_parado_v1

Esta pasta preserva a saida bruta do treino de gait parado que deu certo.

- Nao sobrescrever.
- Nao apagar.
- Contem modelo final, melhor modelo, logs, avaliacoes e TensorBoard.

Comando para continuar a partir do melhor modelo:

```bash
python reinforcement_learning/otimizar_hpps_optuna.py \
  --somente-treinar-melhor \
  --device cuda \
  --modo-gait-andando \
  --continuar-de reinforcement_learning/saida_optuna_atom/treino_gait_parado_v1_PROTEGIDO_NAO_APAGAR/melhor_modelo/best_model.zip \
  --final-timesteps 500000 \
  --eval-freq 50000 \
  --n-eval-episodes 5 \
  --final-run-name treino_gait_andando_v1
```
