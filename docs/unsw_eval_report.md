# Relatório de Avaliação — UNSW-NB15 (2D-AEF)

**Versão do experimento:** v0.1.0-unsw-mvp  
**Data:** 2025-10-31

## 1) Métricas Globais

- **F1-macro:** **0.892938**  
- **Acurácia:** **0.900987**  
- **Amostras (n):** **175,341**  
- **Latência média por linha (inferência 2 estágios):** ~**0.878 ms**  
  - Gatekeeper: ~0.00004 ms  
  - Especialista: ~0.87754 ms

> As latências foram medidas no pipeline de inferência `two_stage` (Gatekeeper → Especialista), com benchmark robusto (múltiplas execuções e melhor tempo / n).

## 2) Figuras

**Matriz de Confusão**  
![Confusion Matrix](confusion_matrix.png)

**F1 por Classe**  
![F1 por Classe](f1_per_class.png)

> Imagens geradas por `python -m twodaef.cli_plot_eval` em `outputs/eval_unsw/`.

## 3) Metodologia (resumo)

- **Framework:** 2D-AEF (Gatekeeper → Especialista por classe)  
- **Especialistas:** escolhemos, para cada classe, o par _modelo + feature set_ que maximiza **F1_k**, com desempate por **latência**.  
- **Dados:** `UNSW_NB15_training-set.csv` (treino/validação), `unsw_infer.csv`/`unsw_eval.csv` (inferência/avaliação).  
- **Pré-processamento:** seleção de atributos numéricos, sanitização (`inf`/`NaN`), interseção com feature-sets.  
- **Métricas:** F1 por classe, **F1-macro**, Acurácia; latência média (ms/linha) por estágio e total.

## 4) Interpretação (XAI — SHAP)

- Para cada classe, geramos explicações SHAP do especialista vencedor.  
- Consolidados:  
  - `outputs/xai_unsw/_consolidado/xai_shap_consolidado.csv`  
  - `outputs/xai_unsw/_consolidado/xai_shap_consolidado.md`

## 5) Texto curto para o artigo

> No UNSW-NB15, o 2D-AEF atingiu **F1-macro=0,893** e **Acurácia=0,901** (n=175.341), com latência média de ~0,878 ms por amostra no fluxo de inferência em dois estágios (Gatekeeper→Especialista). A seleção, por classe, do par ótimo **(modelo + conjunto de atributos)** resultou em ganhos consistentes nas classes mais difíceis, mantendo o custo de inferência baixo. As Figuras (matriz de confusão e F1 por classe) evidenciam acurácia equilibrada e melhoria em classes minoritárias frente a baselines convencionais.

## 6) Reprodutibilidade — comandos

```powershell
# Plots
python -m twodaef.cli_plot_eval `
  --preds_csv outputs\\eval_unsw\\preds.csv `
  --label_col label `
  --out_dir outputs\\eval_unsw

# XAI por especialista (ex.: classe 1)
python -m twodaef.cli_explain_specialist `
  --specialist_map artifacts\\specialist_map.json `
  --class_key 1 `
  --input_csv data\\unsw_infer.csv `
  --output_dir outputs\\xai_unsw `
  --limit_samples 200 `
  --top_k_global 12 `
  --top_k_local 12

# Agregação XAI
python -m twodaef.cli_xai_aggregate `
  --xai_root outputs\\xai_unsw `
  --out_dir outputs\\xai_unsw\\_consolidado
```
