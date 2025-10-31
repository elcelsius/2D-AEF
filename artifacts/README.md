# artifacts/

Modelos, mapas e metadados **gerados** pelo treino/inferência.

- **Não versionar** binários pesados (`.joblib`, `.bin`, etc.).
- Versionar somente JSONs essenciais (ex.: `feature_pool_*.json`, `specialist_map_*.json`) quando fizer sentido.

## Estrutura típica
```
artifacts/
├─ feature_pool_unsw.json
├─ gatekeeper.joblib                 # (ignorado pelo git)
├─ specialists/                      # (gerados / ignorados)
│   ├─ 0/ model.joblib
│   └─ 1/ model.joblib
├─ specialist_map.json               # mapeia classe → modelo e features
├─ gatekeeper_cic.joblib             # (ignorado)
├─ specialists_cic/                  # (gerados / ignorados)
└─ specialist_map_cic.json
```
