# Snapshot do ambiente usado

Backup do ambiente usado no treino protegido `treino_gait_parado_v1_PROTEGIDO_NAO_APAGAR`.

Modo do treino:

```bash
--usar-gait-inicial --modo-gait-parado
```

Arquivos salvos:

- `env_usado_no_treino.py`: copia completa do ambiente `AtomEnv`.
- `FUNCAO_RECOMPENSA_EXATA_USADA.md`: recorte exato dos pesos de recompensa e do metodo `step()`, onde `reward_terms` e `reward` sao calculados.
- `model/scene.xml`: cena MuJoCo carregada pelo ambiente.
- `model/robot.xml`: modelo fisico do robo, incluido por `scene.xml`.

Hashes SHA-256:

```text
df1154ffa870fe80f0fd4a1bb147d63851613c09220fd6c34eb6e11a8a2c74d7  env_usado_no_treino.py
0a970c57de8a487e4ea0a84b0723403e38c5759ce03632c3c307554458ed7229  model/scene.xml
cc59e6446bc589e2954e2a9d82ad5c1c704dc8fccc3a1249fc2a88199444c29e  model/robot.xml
b4bc25d3e9276b6d50e37d5fc6eaf044b58fcdf86e3a82569b9cc8a700e2c819  FUNCAO_RECOMPENSA_EXATA_USADA.md
```
