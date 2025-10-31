# artifacts/

Modelos e artefatos de experimento (gerados pelo pipeline). Este diretório é **parcialmente** ignorado pelo Git:
- binários pesados (`*.joblib`, `*.bin`, etc.) ficam fora do versionamento;
- arquivos de **configuração/Índice** em JSON (ex.: `specialist_map_*.json`, `feature_pool_*.json`) **podem** ser versionados para reprodutibilidade.

## Estrutura típica
```
artifacts/
├─ gatekeeper_*.joblib           # modelo gatekeeper (não versionar)
├─ specialists_*/                # especialistas por classe
│  └─ <class_key>/model.joblib   # modelo da classe (não versionar)
├─ feature_pool_*.json           # pools de atributos (versionar)
└─ specialist_map_*.json         # mapa de especialistas (versionar)
```

> Dica: adicione um `.gitkeep` onde quiser preservar a árvore sem arquivos binários.
