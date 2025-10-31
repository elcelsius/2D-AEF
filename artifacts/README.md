# artifacts/
Repositório local de **artefatos de treino/inferência** do 2D‑AEF.

> **Importante:** por padrão **NÃO** versionamos binários pesados (ex.: `*.joblib`, `*.bin`, `*.model`).  
> O que pode/deve ser versionado: **metadados leves** (ex.: `feature_pool_*.json`, `specialist_map_*.json`) e este `README.md`.

---

## Estrutura sugerida

```
artifacts/
├─ feature_pool_unsw.json            # pool de atributos (UNSW)
├─ feature_pool_cic.json             # pool de atributos (CIC binário)
├─ gatekeeper.joblib                 # modelo Gatekeeper (UNSW) — ignorado no git
├─ gatekeeper_cic.joblib             # modelo Gatekeeper (CIC) — ignorado no git
├─ specialists/                      # especialistas por classe (UNSW) — ignorado no git
│  ├─ 0/ model.joblib
│  └─ 1/ model.joblib
├─ specialists_cic/                  # especialistas por classe (CIC) — ignorado no git
│  ├─ Benign/ model.joblib
│  └─ Others/ model.joblib
├─ specialist_map.json               # mapeia classe→(modelo, features, caminho) (UNSW)
└─ specialist_map_cic.json           # mapeia classe→(modelo, features, caminho) (CIC)
```

> Dica: use arquivos **`.gitkeep`** vazios em pastas vazias para manter a estrutura no repo.

---

## Convenções de nomes
- **Gatekeeper:** `gatekeeper[_dataset].joblib`  (ex.: `gatekeeper.joblib`, `gatekeeper_cic.joblib`)
- **Mapa de especialistas:** `specialist_map[_dataset].json`
- **Especialistas:** `specialists[_dataset]/{classe}/model.joblib`
- **Pool de atributos:** `feature_pool[_dataset].json`

---

## O que vai para o Git?
✅ **Sim (texto leve):**
- `feature_pool_*.json`
- `specialist_map*.json`
- `README.md`
- opcionais: checksums (`.sha256`) dos binários locais

⛔ **Não (binários grandes/derivados):**
- `*.joblib`, `*.bin`, `*.model`, `*.pkl`
- diretórios `specialists*/` com os modelos
- qualquer export pesado de treino

> A política acima está refletida no `.gitignore` do projeto.

---

## Como (re)gerar
- **Gatekeeper (ex.: UNSW)**  
  ```powershell
  gatekeeper-train `
    --train_csv data\raw\unsw\UNSW_NB15_training-set.csv `
    --target_col label `
    --features cols.txt `
    --model_out artifacts\gatekeeper.joblib
  ```

- **Especialistas + mapa (ex.: UNSW)**  
  ```powershell
  train-specialists `
    --train_csv data\raw\unsw\UNSW_NB15_training-set.csv `
    --target_col label `
    --feature_pool_json artifacts\feature_pool_unsw.json `
    --out_dir artifacts\specialists `
    --map_path artifacts\specialist_map.json
  ```

- **Inferência 2 estágios**  
  ```powershell
  infer-twostage `
    --gatekeeper_model artifacts\gatekeeper.joblib `
    --gatekeeper_features cols.txt `
    --specialist_map artifacts\specialist_map.json `
    --input_csv data\unsw_infer.csv `
    --output_csv outputs\preds_twostage_unsw.csv
  ```

> Ajuste caminhos/datasets conforme necessário (UNSW ↔ CIC).

---

## Boas práticas
- Mantenha os **metadados `.json`** consistentes com o conteúdo local das pastas `specialists*/`.
- Quando trocar de máquina/ambiente, re‑treine os binários localmente usando os mesmos scripts/semillas.
- Se precisar compartilhar modelos, prefira **artefatos versionados externamente** (ex.: releases, storage) e **checksums** para verificação.

