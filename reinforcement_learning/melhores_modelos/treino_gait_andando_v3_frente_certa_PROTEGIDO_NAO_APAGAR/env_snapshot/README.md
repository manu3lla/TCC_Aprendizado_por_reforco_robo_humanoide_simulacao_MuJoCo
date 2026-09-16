# Snapshot do ambiente usado

Backup do ambiente usado no treino protegido `treino_gait_andando_v3_frente_certa_PROTEGIDO_NAO_APAGAR`.

Modo do treino:

```bash
--usar-gait-inicial --modo-gait-andando
```

Arquivos salvos:

- `env_usado_no_treino.py`: copia completa do ambiente `AtomEnv`.
- `FUNCAO_RECOMPENSA_EXATA_USADA.md`: recorte exato dos pesos de recompensa e do metodo `step()`, onde `reward_terms` e `reward` sao calculados.
- `model/scene.xml`: cena MuJoCo carregada pelo ambiente.
- `model/robot.xml`: modelo fisico do robo, incluido por `scene.xml`.

Hashes SHA-256:

```text
46b0da83cb09a2a4bf579fa29baa52a0cbfc5ee39c74a40fcbffbefd13e5c5ab  env_usado_no_treino.py
0a970c57de8a487e4ea0a84b0723403e38c5759ce03632c3c307554458ed7229  model/scene.xml
cc59e6446bc589e2954e2a9d82ad5c1c704dc8fccc3a1249fc2a88199444c29e  model/robot.xml
99190acd26162fc5e2c9c26f3808c917783dec35495440916ebcc6d09d380f99  FUNCAO_RECOMPENSA_EXATA_USADA.md
```
