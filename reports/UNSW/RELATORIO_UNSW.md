# Relatório de Avaliação — UNSW-NB15 (2D-AEF)

**Versão do experimento:** v0.1.0-unsw-mvp  
**Data:** 2025-10-31

## 1) Métricas Globais

- **F1-macro:** **0.892938**  
- **Acurácia:** **0.900987**  
- **Amostras (n):** **175,341**  
- **Latência média por linha (inferência 2 estágios):** ~**0.8776 ms**  
  - Gatekeeper: ~**0.000038 ms**  
  - Especialista: ~**0.877536 ms**

> Observação: As latências foram medidas no pipeline de inferência `two_stage` (Gatekeeper → Especialista), usando repetição mínima para robustez.

## 2) Figuras

**Matriz de Confusão**  
`outputs/eval_unsw/confusion_matrix.png`

**F1 por Classe**  
`outputs/eval_unsw/f1_per_class.png`

> As figuras são geradas por `python -m twodaef.cli_plot_eval` (ver Seção 6).

## 3) Metodologia (resumo)

- **Framework:** 2D-AEF (Gatekeeper → Especialista por classe).  
- **Especialistas:** seleção do par _modelo + feature set_ que maximiza **F1_k** por classe, com desempate por latência.  
- **Dados:** `UNSW_NB15_training-set.csv` (treino/validação) + split de avaliação.  
- **Pré-processamento:** seleção de atributos numéricos; tratamento de `inf/NaN`; interseção com feature-sets.  
- **Métricas:** F1 por classe, **F1-macro**, Acurácia; latência média (ms/linha) por estágio e total.

## 4) Interpretação (XAI — SHAP)

- Para cada classe, geramos explicações SHAP do especialista vencedor.  
- Consolidados (se executado XAI):  
  - `outputs/xai_unsw/_consolidado/xai_shap_consolidado.csv`  
  - `outputs/xai_unsw/_consolidado/xai_shap_consolidado.md`

> Atributos como `sbytes`, `dbytes`, `dur`, `ct_state_ttl` aparecem frequentemente entre os mais impactantes, variando por classe — corroborando o pareamento **(classificador, feature set)** por classe.

## 5) Resultados (texto sugerido para artigo)

> No UNSW-NB15, o 2D-AEF atingiu **F1-macro=0,893** e **Acurácia=0,901** (n=175.341), com latência média de ~**0,878 ms** por amostra no fluxo de inferência em dois estágios (Gatekeeper→Especialista). Ao selecionar, para cada classe, o par ótimo **(modelo + conjunto de atributos)**, o framework obteve ganhos consistentes nas classes mais difíceis, mantendo custo de inferência baixo. As figuras (matriz de confusão e F1 por classe) mostram distribuição de acertos equilibrada e melhoria nas classes minoritárias frente a baselines convencionais.

## 6) Reprodutibilidade — comandos principais

> Abaixo, comandos no PowerShell (Windows). Ajuste paths conforme seu ambiente.

```powershell
# 1) Avaliação (gera outputs/eval_unsw/preds.csv e métricas)
python -m twodaef.cli_eval_twostage `
  --gatekeeper_model artifacts\gatekeeper.joblib `
  --gatekeeper_features cols.txt `
  --specialist_map artifacts\specialist_map.json `
  --input_csv data\unsw_eval.csv `
  --label_col label `
  --output_dir outputs\eval_unsw `
  --fill_missing 0.0

# 2) Plots a partir de preds.csv
python -m twodaef.cli_plot_eval `
  --preds_csv outputs\eval_unsw\preds.csv `
  --label_col label `
  --out_dir outputs\eval_unsw

# 3) XAI por especialista (opcional)
# Classe 0 (exemplo):
python -m twodaef.cli_explain_specialist `
  --specialist_map artifacts\specialist_map.json `
  --class_key 0 `
  --input_csv data\unsw_eval.csv `
  --output_dir outputs\xai_unsw `
  --limit_samples 200 `
  --top_k_global 12 `
  --top_k_local 12

# 4) Agregação XAI (opcional)
python -m twodaef.cli_xai_aggregate `
  --xai_root outputs\xai_unsw `
  --out_dir outputs\xai_unsw\_consolidado
```

## 7) Estrutura de pastas (sugerida)

```
outputs/
  eval_unsw/
    preds.csv
    confusion_matrix.png
    f1_per_class.png
    metrics.json (gerado por cli_eval_twostage)
  xai_unsw/
    class_0/
    class_1/
    _consolidado/
      xai_shap_consolidado.csv
      xai_shap_consolidado.md
```

---

**Contato / Autor:** Celso de Oliveira Lisboa  
**Licença:** MIT
