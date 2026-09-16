# PROTEGIDO - NAO APAGAR

Este backup protege o treino `treino_gait_andando_v3_frente_certa`, que foi marcado como caminhada boa/perfeita.

Use este modelo para visualizar:

```bash
python reinforcement_learning/ver_modelo_atom.py \
  --modo-gait-andando \
  --modelo reinforcement_learning/melhores_modelos/treino_gait_andando_v3_frente_certa_PROTEGIDO_NAO_APAGAR/best_model.zip
```

Use este modelo como ponto de partida para continuar treinamento:

```bash
python reinforcement_learning/otimizar_hpps_optuna.py \
  --somente-treinar-melhor \
  --device cuda \
  --modo-gait-andando \
  --continuar-de reinforcement_learning/saida_optuna_atom/treino_gait_andando_v3_frente_certa_PROTEGIDO_NAO_APAGAR/melhor_modelo/best_model.zip \
  --final-timesteps 1000000 \
  --eval-freq 50000 \
  --n-eval-episodes 5 \
  --final-run-name novo_treino_partindo_do_walking_perfeito
```
