# Relatório de Avaliação — CIC‑IDS2018 (2D‑AEF)

**Versão do experimento:** v0.1.0-cic-mvp  
**Data:** 2025-10-31

## 1) Métricas Globais

- **F1-macro:** **1.000000**  
- **Acurácia:** **1.000000**  
- **Amostras (n):** **100,000**  
- **Latência média por linha (inferência 2 estágios):** ~**0.0001 ms**  
  - Gatekeeper: ~0.00008 ms  
  - Especialista: ~0.00000 ms

> Observação: As latências referem-se ao pipeline de inferência `two_stage` (Gatekeeper → Especialista) para o conjunto de avaliação do CIC‑IDS2018.

## 2) Figuras

**Matriz de Confusão**  
![Confusion Matrix](confusion_matrix.png)

**F1 por Classe**  
![F1 por Classe](f1_per_class.png)

> As imagens acima foram geradas pelo comando `python -m twodaef.cli_plot_eval` a partir de `outputs/eval_cic/preds.csv`.

## 3) Metodologia (resumo)

- **Framework:** 2D‑AEF (Gatekeeper → Especialista por classe).  
- **Gatekeeper:** classificador binário para *Benign* vs *Others*.  
- **Especialistas (por classe):** seleção do par _modelo + feature set_ que maximiza **F1_k**, com desempate por latência.  
- **Dados:** amostras do CIC‑IDS2018 preparadas por `scripts/prep_cic_train.py` (split para treino/avaliação).  
- **Pré-processamento:** seleção de atributos numéricos, tratamento de `inf/NaN`, interseção com feature‑sets.  
- **Métricas:** F1 por classe, **F1‑macro**, Acurácia; latência média (ms/linha) por estágio e total.

## 4) Interpretação (XAI — opcional)

> Para o CIC‑IDS2018, o resultado foi *perfeito* no conjunto de avaliação selecionado. Caso deseje explorar relevâncias:  
> - Execute explicações SHAP para o especialista de *Others* (ou *Benign*) com `python -m twodaef.cli_explain_specialist` e consolide com `python -m twodaef.cli_xai_aggregate` (ver README).

## 5) Reprodutibilidade (comandos‑chave)

```powershell
# Avaliação (gera preds.csv + métricas)
python -m twodaef.cli_eval_twostage `
  --gatekeeper_model artifacts\gatekeeper_cic.joblib `
  --gatekeeper_features gatekeeper_cic_cols.txt `
  --specialist_map artifacts\specialist_map_cic.json `
  --input_csv data\cic_eval.csv `
  --label_col label `
  --output_dir outputs\eval_cic `
  --fill_missing 0.0

# Plots (usa outputs\eval_cic\preds.csv)
python -m twodaef.cli_plot_eval `
  --preds_csv outputs\eval_cic\preds.csv `
  --label_col label `
  --out_dir outputs\eval_cic
```

## 6) Como citar (Seção 4 — Resultados e Discussão)

> No CIC‑IDS2018, o 2D‑AEF atingiu **F1‑macro=1,000** e **Acurácia=1,000** (n=100.000), com latência média de ~0,0001 ms por amostra no fluxo de inferência em dois estágios (Gatekeeper→Especialista). A estratégia de pareamento **(modelo + conjunto de atributos)** por classe garantiu decisão consistente e de baixíssimo custo computacional neste cenário.
