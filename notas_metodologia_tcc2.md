# Conferencias para a metodologia

O arquivo `metodologia_tcc2.tex` substitui apenas a metodologia. Nao altera o texto anterior, o ambiente nem os modelos. E uma proposta de redacao para revisao pelas autoras, com as pendencias documentadas abaixo.

## Respostas ao roteiro

- A integracao aproveitou a infraestrutura do projeto atribuido a Fabio Gianzanti, adaptada para o Atom. O pacote `op3` e sua autoria ja aparecem no primeiro commit do ambiente (`61ba395`). A contribuicao local inclui `AtomEnv`, recompensas, observacoes, controle residual, treinamento e registro. Isso nao permite atribuir cada linha individual ao projeto original; a extensao exata do reaproveitamento exige comparar com o repositorio de origem.
- Existe uma funcao anterior sem target no Git: commit `e052d62`, arquivo `reinforcement_learning/op3_model/src/atom/env.py`, principalmente linhas 287 a 338 daquela revisao. Ela soma deslocamento em x, alive e estabilidade, subtrai magnitude da acao e aplica penalidade ao encerrar por queda. A equacao no texto resume essa versao, nao uma suposta formula unica desde o inicio.
- Uma versao inicial de target aparece em `856d10d`: projecao da velocidade na direcao do alvo, constante de 0.1 e bonus de chegada. Nessa revisao a penalidade explicita de queda estava comentada, embora a queda encerrasse o episodio. Os termos de joelho/orientacao aparecem em outras revisoes; nao atribuir todos a essa primeira versao.
- O gait foi construido como referencia cinematica. Depois houve treino PPO no lugar e continuacao para deslocamento. A politica aprende correcoes residuais sobre comandos de posicao; nao e aprendizado do zero nem clonagem de demonstracoes humanas.

## Ajustes de consistencia com o artigo anexado

- No checkpoint protegido recente: observacao 62; acao 20; redes separadas da politica e do valor 64--64, Tanh. Nao usar 52 observacoes e 256--256/ReLU para descrever esse checkpoint.
- As acoes sao correcoes de referencias de posicao, nao torques diretos. Penalizar acoes nao mede diretamente energia consumida.
- O treinamento atual de `otimizar_hpps_optuna.py` cria um ambiente de treino e outro de avaliacao. Nao comprova oito ambientes paralelos nem treinamento via servidor/socket da RoboCup.
- Altura do tronco nao e altura do centro de massa. `torso_upright` nao e uma medicao completa da orientacao em tres eixos.
- `ctrlrange` limita comandos do atuador; nao deve ser descrito automaticamente como limite mecanico da junta. Validacao dos limites reais requer evidencia adicional.
- O terreno tem textura de grama: a imagem, por si so, nao comprova reproducao de propriedades fisicas de grama sintetica.
- Modelos protegidos antigos podem nao conter assets completos. A copia parcial do v5 inclui snapshot capturado no momento da protecao, e nao um snapshot comprovadamente obtido no inicio do treinamento.
- O checkpoint parcial protegido do v5 tem `num_timesteps=16750000`; em continuacoes esse contador pode incluir passos anteriores. Nao apresentar como quantidade adicional desta tentativa sem conferir os logs.
- Versoes v6/v7 e teste v8 sao refinamentos exploratorios. Nao afirmar suavidade ou estabilidade superiores sem resultados comparativos.

## Referencias e figuras pendentes

- O endereco indicado da Unicamp (`https://repositorio.unicamp.br/acervo/detalhe/1162057`) respondeu com uma tela de validacao, sem disponibilizar titulo/autoria/conteudo na consulta. Nao acrescentei citacao nem atribui a esse documento uma formula especifica. Conferir o PDF e a entrada BibTeX antes de atribuir a origem da estrategia de target.
- As chaves `gianzanti2022`, `gondry2019` e `allali2024` foram reaproveitadas do texto anexado, mas o arquivo `referencias.bib` nao foi fornecido. Confirmar autoria, ano e correspondencia das fontes. A autoria do pacote local e identificavel; isso nao valida automaticamente a entrada bibliografica.
- Nao localizei nem identifiquei a foto mencionada com os pesos da tentativa antiga que atingiu o alvo. Nao inventei valores nem referencia a uma figura inexistente.
- Sugestoes de figuras: fluxo das etapas; Atom e alvo na cena; tres poses da referencia de gait; curvas de um mesmo experimento. Numeros finais, comparacoes e sucesso observado pertencem aos resultados, e nao precisam ser antecipados como fatos na metodologia.
- As equacoes agrupadas sao explicacoes do metodo. Para a funcao exata de uma tentativa, usar o respectivo `env.py`/snapshot e informar seus pesos.

## Validacao editorial

O trecho nao foi compilado como documento completo: depende do preambulo e da bibliografia do artigo. O anexo tem duas cargas de `biblatex`; revisar separadamente para evitar conflito de opcoes. A revisao desta entrega e apenas da metodologia, conforme solicitado.
