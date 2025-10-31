# 2D-AEF — Ensemble Adaptativo Bidimensional para IDS

**Objetivo:** framework original que combina:
1) **Gatekeeper** (pré-classificador leve),  
2) **Especialista por classe** (par ótimo *classificador + feature set*),  
3) **XAI (SHAP)** para transparência das decisões.

Este repositório acompanha o artigo e os experimentos com **UNSW-NB15** (binário) e, em seguida, **CIC-IDS 2018** (multiataque).

---

## Sumário
- [Requisitos](#requisitos)
- [Instalação](#instalação)
- [Dados (UNSW e CIC-IDS2018)](#dados-unsw-e-cic-ids2018)
- [Pipeline Rápido (UNSW)](#pipeline-rápido-unsw)
- [Avaliação + Plots](#avaliação--plots)
- [XAI (SHAP)](#xai-shap)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Problemas Comuns](#problemas-comuns)
- [Licença](#licença)

---

## Requisitos
- **Python 3.11+** (recomendado)
- Windows 10/11 (PowerShell) **ou** WSL/Ubuntu
- (Opcional) **Kaggle CLI** para baixar datasets
- (Opcional) **Git** para versionar o projeto

---

## Instalação

### Via `pip` (recomendado aqui)
```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```
> Se preferir **Poetry**, também funciona; mas este repo está configurado com **setuptools + pip**.

---

## Dados (UNSW e CIC-IDS2018)

### UNSW-NB15 (Kaggle)
```powershell
# listar datasets (exemplo)
kaggle datasets list -s "unsw nb15"

# baixar
mkdir data\raw\unsw
kaggle datasets download -d mrwellsdavid/unsw-nb15 -p data\raw\unsw --unzip
```
Arquivos esperados (exemplos):
- `data/raw/unsw/UNSW_NB15_training-set.csv`
- `data/raw/unsw/UNSW_NB15_testing-set.csv`
- `data/raw/unsw/NUSW-NB15_features.csv`

### CIC-IDS2018 (Kaggle)
```powershell
mkdir data\raw\cicids2018
kaggle datasets download -d solarmainframe/ids-intrusion-csv -p data\raw\cicids2018 --unzip
```

---

## Pipeline Rápido (UNSW)

> **1) Gatekeeper** (treinar + prever)

- Crie um arquivo `cols.txt` (features baratas do Gatekeeper: **uma por linha**).
- Treine:
```powershell
gatekeeper-train `
  --train_csv data\raw\unsw\UNSW_NB15_training-set.csv `
  --target_col label `
  --features cols.txt `
  --model_out artifacts\gatekeeper.joblib
```

- Predição rápida (teste):
```powershell
gatekeeper-predict `
  --model artifacts\gatekeeper.joblib `
  --input_csv data\raw\unsw\UNSW_NB15_testing-set.csv `
  --output_csv outputs\preds_gatekeeper.csv
```

> **2) Pool de Features** (geração inicial baseada em **Mutual Information** + aleatório controlado)

```powershell
make-feature-pool `
  --csv data\raw\unsw\UNSW_NB15_training-set.csv `
  --target_col label `
  --max_features_per_set 20 `
  --total_sets 30 `
  --seed 42 `
  --out_json artifacts\feature_pool_unsw.json
```

> **3) Treinar Especialistas e gerar o mapa**

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

> **4) Inferência 2 estágios (Gatekeeper + Especialista)**

Crie (se não existir) `artifacts\gk_labelmap_unsw.json`:
```json
{
  "Benign": "0",
  "benign": "0",
  "Normal": "0",
  "normal": "0",
  "0": "0",
  "1": "1",
  "Attack": "1",
  "attack": "1",
  "Malicious": "1",
  "malicious": "1"
}
```

Rode:
```powershell
infer-twostage `
  --gatekeeper_model artifacts\gatekeeper.joblib `
  --gatekeeper_features cols.txt `
  --specialist_map artifacts\specialist_map.json `
  --input_csv data\raw\unsw\UNSW_NB15_testing-set.csv `
  --output_csv outputs\eval_unsw\preds.csv `
  --fill_missing 0.0
```

> **5) Avaliar (F1-macro / Acc)**
```powershell
eval-twostage `
  --gatekeeper_model artifacts\gatekeeper.joblib `
  --gatekeeper_features cols.txt `
  --specialist_map artifacts\specialist_map.json `
  --input_csv data\raw\unsw\UNSW_NB15_testing-set.csv `
  --label_col label `
  --output_dir outputs\eval_unsw `
  --fill_missing 0.0
```

---

## Avaliação + Plots

Gera **matriz de confusão** e **barras de F1 por classe**:
```powershell
plot-eval `
  --preds_csv outputs\eval_unsw\preds.csv `
  --label_col label `
  --out_dir outputs\eval_unsw
```
Saídas esperadas:
- `outputs\eval_unsw\confusion_matrix.png`
- `outputs\eval_unsw\f1_per_class.png`
- `outputs\eval_unsw\metrics_again.json`

---

## XAI (SHAP)

**Por classe de especialista** (exemplo, classe `1`):
```powershell
explain-specialist `
  --specialist_map artifacts\specialist_map.json `
  --class_key 1 `
  --input_csv data\unsw_infer.csv `
  --output_dir outputs\xai_unsw `
  --limit_samples 200 `
  --top_k_global 12 `
  --top_k_local 12
```

Consolida SHAP de múltiplas classes:
```powershell
aggregate-xai `
  --root_dir outputs\xai_unsw `
  --out_csv outputs\xai_unsw\_consolidado\xai_shap_consolidado.csv `
  --out_md  outputs\xai_unsw\_consolidado\xai_shap_consolidado.md
```

---

## Estrutura do Projeto
```
2D-AEF/
├── .git/
├── .idea/
├── .venv/
├── artifacts/
├── catboost_info/
├── data/
│   └── raw/
├── docs/
├── outputs/
├── reports/
├── scripts/
├── src/
│   ├── twodaef/
│   │   ├── __init__.py
│   │   ├── cli_aggregate_xai.py
│   │   ├── cli_eval_twostage.py
│   │   ├── cli_explain_specialist.py
│   │   ├── cli_infer_twostage.py
│   │   ├── cli_make_feature_pool.py
│   │   ├── cli_plot_eval.py
│   │   ├── cli_predict_gatekeeper.py
│   │   ├── cli_train_gatekeeper.py
│   │   ├── cli_train_specialists.py
│   │   ├── cli_xai_report.py
│   │   ├── eval/
│   │   │   ├── __init__.py
│   │   │   └── evaluate.py
│   │   ├── features/
│   │   │   ├── __init__.py
│   │   │   ├── costs.py
│   │   │   └── pools.py
│   │   ├── gatekeeper.py
│   │   ├── infer/
│   │   │   ├── __init__.py
│   │   │   └── two_stage.py
│   │   ├── reports/
│   │   │   ├── __init__.py
│   │   │   ├── aggregate_xai.py
│   │   │   └── plots_eval.py
│   │   ├── specialists/
│   │   │   ├── __init__.py
│   │   │   └── train_specialists.py
│   │   ├── utils/
│   │   │   ├── __init__.py
│   │   │   ├── io.py
│   │   │   └── metrics.py
│   │   └── xai/
│   │       ├── __init__.py
│   │       └── shap_explain.py
│   └── twodaef.egg-info/
├── .gitattributes
├── .gitignore
├── CHANGELOG.md
├── cols.txt
├── CONTRIBUTING.md
├── gatekeeper_cic_cols.txt
├── poetry.lock
├── pyproject.toml
├── README.md
└── requirements.txt
```

---

## Problemas Comuns

- **`ModuleNotFoundError: twodaef`**  
  → Rode `pip install -e .` na raiz do projeto (com a venv ativa).

- **`ValueError: n_jobs == 0`** no RandomForest  
  → Já está `n_jobs=-1`. Se aparecer, atualize scikit-learn.

- **Tipos mistos em avaliação** (`str` vs `int`)  
  → O `two_stage.py` normaliza e o `preds.csv` contém `pred_gatekeeper_mapped`. No `eval`, garantimos `astype(int)`.

- **LightGBM no Windows**  
  → Se der erro de compilação, comente `lightgbm` no `requirements.txt` e rode apenas com `sk_hgb`/`sk_rf`. Depois reinstalamos.

---

## 🧭 Arquitetura do Projeto

A arquitetura completa (com diagramas Mermaid) está documentada em **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**.

**Conteúdo do documento:**
- Visão geral do pipeline (ingestão → gatekeeper → matriz de especialistas → inferência 2 estágios → avaliação → XAI).
- Diagrama de módulos do pacote `twodaef/` (CLIs, `eval/`, `reports/`, `xai/`).
- Fluxos de treino e de inferência (inclui contrato de I/O e artefatos gerados).
- Padrões de pastas (`data/`, `artifacts/`, `outputs/`, `reports/`) e boas práticas de versionamento.
- Roadmap e pontos de extensão (novos datasets, novos especialistas, tuning, serving).

> Dica: o GitHub renderiza Mermaid nativamente; se o preview local não mostrar os diagramas, abra o arquivo no GitHub.

---

## Licença
MIT © Celso de Oliveira Lisboa
