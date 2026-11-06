# TCC_Aprendizado_por_reforco_robo_humanoide_simulacao_MuJoCo
Repositório para o TCC "Aprendizado por reforço para locomoção bípede de um robô humanoide em simulação MuJoCo"

## Caminhada Rapida Ate o Alvo

Na raiz do repositorio, com as dependencias instaladas no `.venv`:

```bash
source .venv/bin/activate
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python reinforcement_learning/otimizar_hpps_optuna.py \
  --somente-treinar-melhor \
  --continuar-de reinforcement_learning/melhores_modelos/treino_walking_estabilidade_v8_protegido_nao_apagar/best_model.zip \
  --modo-gait-rapido --velocidade-alvo 0.35 \
  --learning-rate 0.00001 --device cpu \
  --final-timesteps 1000000 --eval-freq 25000 --n-eval-episodes 5 \
  --final-run-name treino_walking_rapido_v13
```

`0.35` m/s e o objetivo do treino, nao uma velocidade ja garantida. Por padrao,
o modo rapido preserva a cadencia, os controles e os limites de queda da marcha;
ajusta a recompensa de velocidade e de passada. Os bonus de postura e marcha
acompanham o progresso equilibrado; o erro de velocidade gera custo em vez de
um bonus fixo por passo, que favoreceria demorar mais para chegar.
Aumentar a cadencia diretamente
no checkpoint estavel prejudicou a direcao e o equilibrio nos testes.

O checkpoint inicial tambem e avaliado. `melhor_modelo/best_model.zip` so e
substituido por um modelo mais rapido que chegue ao alvo em todas as avaliacoes,
sem nenhuma queda. O tempo inclui a partida. A velocidade medida e a reducao
da distancia ao alvo dividida pelo tempo simulado; deslocamento lateral nao conta
como avanco. Os resultados ficam em `logs/speed_evaluations.json` e no TensorBoard.
O arquivo `ppo_atom_optuna_final.zip` guarda o ultimo estado do treino e pode ser
pior que o melhor checkpoint. Escolha um nome novo para cada treino.

Para medir o melhor checkpoint sem abrir a janela:

```bash
python reinforcement_learning/avaliar_locomocao.py \
  --modelo reinforcement_learning/saida_optuna_atom/treino_walking_rapido_v13/melhor_modelo/best_model.zip \
  --modo-gait-rapido --velocidade-alvo 0.35 --n-episodios 5
```

Essa medicao usa o `env.py` atual. O reset e a politica sao deterministicos:
repetir com outras seeds nao testa perturbacoes ou condicoes iniciais diferentes.
Ausencia de quedas nessa avaliacao nao garante equilibrio em todas as situacoes.

Para visualizar com a mesma configuracao:

```bash
python reinforcement_learning/ver_modelo_atom.py \
  --modelo reinforcement_learning/saida_optuna_atom/treino_walking_rapido_v13/melhor_modelo/best_model.zip \
  --modo-gait-rapido --velocidade-alvo 0.35 --seguir-robo
```

Quando o treino e registrado em `melhores_modelos`, seu `metadata.json` inclui os
parametros e o comando de visualizacao com o snapshot do ambiente correspondente.

Resultado dos primeiros pilotos: o melhor checkpoint preservado em
`saida_optuna_atom/treino_walking_rapido_v12/melhor_modelo/best_model.zip`
chegou em 20.07 s a 0.2342 m/s, contra 20.63 s e 0.2279 m/s do modelo estavel
v8. Ganho de aproximadamente 3%, sem queda na avaliacao deterministica.
A meta de 0.35 m/s ainda nao foi atingida. Os ultimos checkpoints de algumas
rodadas cairam e foram rejeitados; use o `best_model.zip`, nao o arquivo final.
Os detalhes estao em `saida_optuna_atom/treino_walking_rapido_v12/medicao_melhor.json`.

### Modelo Acelerado Com Cadencia Ajustada

O melhor modelo local passou a ser
`saida_optuna_atom/treino_walking_rapido_v17_cadencia68/melhor_modelo/best_model.zip`.
A nova rodada incluiu duas continuacoes PPO de aproximadamente 100 mil passos,
busca de ajustes na saida do ator e na cadencia, e mais 50 mil passos PPO com
a cadencia ajustada. As tentativas que cairam nao substituiram o melhor modelo.
Na verificacao sem janela, o melhor chegou em 18.68 s, a 0.25175 m/s,
aproximadamente 7.5% mais rapido que o v12, sem queda no cenario deterministico.
O desvio lateral maximo foi 0.298 m, contra 0.188 m do v12. A meta de 0.35 m/s
continua pendente. As medicoes ficam em
`saida_optuna_atom/treino_walking_rapido_v17_cadencia68/verificacao_melhor.json`.

Para visualizar:

```bash
.venv/bin/python reinforcement_learning/ver_modelo_atom.py \
  --modelo reinforcement_learning/saida_optuna_atom/treino_walking_rapido_v17_cadencia68/melhor_modelo/best_model.zip \
  --seguir-robo
```

`env_config.json`, ao lado do checkpoint, guarda a cadencia e o modo de treino.
Os scripts de visualizar, avaliar e continuar treino leem esse arquivo. Valores
explicitos de `--periodo-gait` ou `--velocidade-alvo` prevalecem sobre o arquivo.
A visualizacao agora sincroniza o processamento com o tempo simulado e renderiza
uma vez por passo de controle. `--sleep 0` desliga o limite de reproducao;
`--sleep` com valor positivo continua acrescentando a pausa solicitada.

Para continuar esse modelo com novos limites de atualizacao PPO:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python reinforcement_learning/otimizar_hpps_optuna.py \
  --somente-treinar-melhor \
  --continuar-de reinforcement_learning/saida_optuna_atom/treino_walking_rapido_v17_cadencia68/melhor_modelo/best_model.zip \
  --learning-rate 0.00001 --gamma 0.995 --gae-lambda 0.98 \
  --n-epochs 5 --clip-range 0.08 --target-kl 0.005 \
  --device cpu --final-timesteps 1000000 --eval-freq 10000 --n-eval-episodes 5 \
  --final-run-name treino_walking_rapido_v18
```

Os overrides de PPO tambem se aplicam ao carregar um checkpoint. `run_config.json`
registra os parametros efetivos, hash do modelo de origem e do ambiente, e a
quantidade adicional de passos executada. A busca direta de ajustes fica em
`ajustar_politica_velocidade.py`; seus `search.json` identificam candidatos e
resultados, sem contar essas avaliacoes como passos de treino PPO.
