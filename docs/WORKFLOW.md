# 2D-AEF — Fluxo de Desenvolvimento, Treinamento e Avaliação
Versão: v0.1.0

Este documento descreve, ponta-a-ponta, as **etapas do projeto** — desde a
preparação dos dados até a publicação de resultados (métricas, XAI e relatórios).  
Tudo aqui está alinhado com os _CLIs_ já presentes no repositório (UNSW e CIC).

---

## Visão Geral das Etapas

1) **Preparar dados**
   - Baixar/organizar datasets em `data/` (fora do Git por padrão).
   - Gerar splits de **train** / **infer** / **eval** conforme o caso.

2) **Feature Pool (meta-busca de atributos)**
   - Gera `artifacts/feature_pool_*.json` a partir do CSV base (ex.: UNSW, CIC).
   - Cada *feature set* é um candidato para os especialistas.

3) **Treinar Gatekeeper**
   - Modelo **rápido** que roteia as amostras para o especialista apropriado.
   - Salva `artifacts/gatekeeper_*.joblib` e um arquivo `.txt` com as colunas usadas.

4) **Treinar Especialistas por Classe**
   - Busca o melhor par **(modelo + feature set)** por classe, maximizando `F1_k` e
     desempate por latência.
   - Salva `artifacts/specialists_*/*/model.joblib` e o mapa `artifacts/specialist_map_*.json`.

5) **Inferência em 2 estágios**
   - Usa Gatekeeper → Especialista para gerar `preds.csv` em `outputs/<exp>/`.

6) **Avaliação**
   - Recalcula métricas (F1-macro, acc) e gera artefatos de avaliação em `outputs/<exp>/`.

7) **Plots**
   - **plot-eval**: matriz de confusão e F1 por classe (PNG).

8) **XAI**
   - **explain-specialist** (SHAP/LIME) por classe → `outputs/xai_<exp>/class_*`.
   - **aggregate-xai** → consolidados CSV/MD.

9) **Relatórios**
   - Relatórios Markdown prontos para anexar ao artigo: `reports/<dataset>/RELATORIO_*.md`.

---

## 1) Preparação de Dados

### UNSW-NB15 (exemplo)
- Arquivo base: `data/raw/unsw/UNSW_NB15_training-set.csv` (train) e variações para infer/eval.
- (Opcional) Gerar listas de colunas para Gatekeeper a partir de CSV:
```powershell
python scripts/make_gatekeeper_cols_from_csv.py `
  --csv data\raw\unsw\UNSW_NB15_training-set.csv `
  --target_col label `
  --out gatekeeper_unsw_cols.txt
```

### CIC-IDS2018 (exemplo)
- Download via Kaggle (já guiado em `scripts/download_cicids2018.py`) **ou** manual.
- Preparar *train* e *infer*/*eval* com os scripts utilitários:
```powershell
python scripts\prep_cic_train.py
python scripts\make_cic_eval.py
```
Isso produz:
- `data/train_cic.csv` (amostra para treino)
- `data/cic_infer.csv` (amostra para inferência rápida)
- `data/cic_eval.csv` (amostra rotulada para avaliação)

> Observação: pastas `data/`, `artifacts/` e `outputs/` estão ignoradas no Git por padrão;
> cada uma contém um `README.md` com instruções locais.

---

## 2) Gerar Feature Pool

Exemplo (UNSW):
```powershell
python -m twodaef.cli_make_feature_pool `
  --csv data\raw\unsw\UNSW_NB15_training-set.csv `
  --target_col label `
  --max_features_per_set 20 `
  --total_sets 30 `
  --seed 42 `
  --out_json artifacts\feature_pool_unsw.json
```

Exemplo (CIC):
```powershell
python -m twodaef.cli_make_feature_pool `
  --csv data\train_cic.csv `
  --target_col label `
  --max_features_per_set 20 `
  --total_sets 30 `
  --seed 42 `
  --out_json artifacts\feature_pool_cic.json
```

---

## 3) Treinar Gatekeeper

Exemplo (UNSW):
```powershell
gatekeeper-train `
  --train_csv data\raw\unsw\UNSW_NB15_training-set.csv `
  --target_col label `
  --features cols.txt `
  --model_out artifacts\gatekeeper.joblib
```

Exemplo (CIC):
```powershell
gatekeeper-train `
  --train_csv data\train_cic.csv `
  --target_col label `
  --features gatekeeper_cic_cols.txt `
  --model_out artifacts\gatekeeper_cic.joblib
```

---

## 4) Treinar Especialistas por Classe

Exemplo (UNSW — com busca automática de modelos disponíveis):
```powershell
train-specialists `
  --train_csv data\raw\unsw\UNSW_NB15_training-set.csv `
  --target_col label `
  --feature_pool_json artifacts\feature_pool_unsw.json `
  --out_dir artifacts\specialists `
  --map_path artifacts\specialist_map.json `
  --test_size 0.2 `
  --seed 42 `
  --models auto
```

Exemplo (CIC — salvando em pastas separadas):
```powershell
train-specialists `
  --train_csv data\train_cic.csv `
  --target_col label `
  --feature_pool_json artifacts\feature_pool_cic.json `
  --out_dir artifacts\specialists_cic `
  --map_path artifacts\specialist_map_cic.json `
  --test_size 0.2 `
  --seed 42 `
  --models auto
```

---

## 5) Inferência em 2 Estágios

Exemplo (UNSW — *quick check*):
```powershell
infer-twostage `
  --gatekeeper_model artifacts\gatekeeper.joblib `
  --gatekeeper_features cols.txt `
  --specialist_map artifacts\specialist_map.json `
  --input_csv data\unsw_infer.csv `
  --output_csv outputs\preds_twostage_unsw.csv `
  --fill_missing 0.0
```

Exemplo (CIC — para avaliação):
```powershell
infer-twostage `
  --gatekeeper_model artifacts\gatekeeper_cic.joblib `
  --gatekeeper_features gatekeeper_cic_cols.txt `
  --specialist_map artifacts\specialist_map_cic.json `
  --input_csv data\cic_eval.csv `
  --output_csv outputs\eval_cic\preds.csv `
  --fill_missing 0.0
```

---

## 6) Avaliação Oficial (gera métricas + guarda artefatos)

```powershell
eval-twostage `
  --gatekeeper_model artifacts\gatekeeper_cic.joblib `
  --gatekeeper_features gatekeeper_cic_cols.txt `
  --specialist_map artifacts\specialist_map_cic.json `
  --input_csv data\cic_eval.csv `
  --label_col label `
  --output_dir outputs\eval_cic
```

Saída típica:
- `F1-macro`, `accuracy`, `n`
- `outputs/<exp>/preds.csv` (cópia de segurança ou gerado no passo anterior)
- `outputs/<exp>/metrics.json` (dependendo da versão do script)

---

## 7) Plots (Matriz de Confusão, F1 por classe)

```powershell
plot-eval `
  --preds_csv outputs\eval_cic\preds.csv `
  --label_col label `
  --out_dir outputs\eval_cic
```

Saídas:
- `outputs/<exp>/confusion_matrix.png`
- `outputs/<exp>/f1_per_class.png`
- `outputs/<exp>/metrics_again.json`

> O `plot-eval` possui uma heurística de **alinhamento automático** entre
> espaços de rótulos numéricos/textuais (ex.: `0/1` ↔ `Benign/Others`)
> para evitar incompatibilidades.

---

## 8) XAI (Por Especialista) e Consolidação

### XAI por classe
```powershell
# Ex.: CIC — classe 'Benign'
explain-specialist `
  --specialist_map artifacts\specialist_map_cic.json `
  --class_key Benign `
  --input_csv data\cic_eval.csv `
  --output_dir outputs\xai_cic `
  --limit_samples 200 `
  --top_k_global 12 `
  --top_k_local 12
```

```powershell
# Ex.: CIC — classe 'Others'
explain-specialist `
  --specialist_map artifacts\specialist_map_cic.json `
  --class_key Others `
  --input_csv data\cic_eval.csv `
  --output_dir outputs\xai_cic `
  --limit_samples 200 `
  --top_k_global 12 `
  --top_k_local 12
```

### Consolidação XAI
```powershell
aggregate-xai `
  --xai_root outputs\xai_cic `
  --out_dir outputs\xai_cic\_consolidado
```
Saídas:
- `xai_shap_consolidado.csv`
- `xai_shap_consolidado.md`

---

## 9) Relatórios

- **UNSW**: `reports/UNSW/RELATORIO_UNSW.md`
- **CIC**:  `reports/cic/RELATORIO_CIC.md`

> Dica: gere os gráficos (passo 7) e consolide XAI (passo 8) **antes** de exportar os relatórios — assim você inclui as figuras/links corretos.

---

## Boas Práticas de Organização

- **Experimentos por pasta** em `outputs/` (ex.: `outputs/eval_unsw`, `outputs/eval_cic`).
- **Artefatos por dataset** em `artifacts/` com nomes claros (ex.: `specialist_map_unsw.json`, `gatekeeper_cic.joblib`).
- **Controle de versão**: evite versionar dados/binários. `README.md` locais (placeholders) explicam como reproduzir.
- **Issues & Milestones** (GitHub): crie _issues_ por dataset/feature nova e agrupe milestones por release (`v0.2.0`, etc.).

---

## Como Estender para um Novo Dataset

1. Adapte/crie um script `scripts/prep_<novo>.py` para gerar `train_<novo>.csv`, `infer_<novo>.csv`, `eval_<novo>.csv`.
2. Gere `feature_pool_<novo>.json` (passo 2).
3. Treine Gatekeeper (passo 3) e Especialistas (passo 4).
4. Rode inferência/avaliação/plots/XAI (passos 5–8).
5. Documente em `reports/<novo>/RELATORIO_<NOVO>.md` e atualize `docs/ARCHITECTURE.md` se necessário.

---

## Troubleshooting Rápido

- **Tipos de rótulo divergentes** (ex.: `0/1` vs `Benign/Others`): use o `plot-eval` (já alinha automaticamente) **ou** normalize rótulos antes.
- **SHAP com XGBoost**: usamos `Explainer(..., algorithm="permutation")` quando necessário, evitando problemas de *base_score*.
- **Latência 0.000** no Gatekeeper: os _benchmarks_ fazem repetições; valores muito pequenos podem aparecer quase zero — use o total por linha para decisões de desempenho.

---

## Referências Internas
- `docs/ARCHITECTURE.md` (diagrama e visão macro)
- `README.md` (setup rápido)
- `CONTRIBUTING.md` (regras de branches/commits)
- `CHANGELOG.md` (histórico de versões)
