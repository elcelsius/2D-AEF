# Relatório de Avaliação — CIC-IDS2018 (2D-AEF)

**Versão:** v0.1.0-cic-mvp  
**Data:** 2025-10-31

## 1) Métricas Globais
- Ver `reports/cic/metrics.json` (ou `outputs/eval_cic/metrics.json` se preferir a origem).
- Figuras (abaixo) correspondem ao conjunto de avaliação `data/cic_eval.csv`.

## 2) Figuras
**Matriz de Confusão**  
![Confusion Matrix](confusion_matrix.png)

**F1 por Classe**  
![F1 por Classe](f1_per_class.png)

## 3) Metodologia (resumo)
- Framework 2D-AEF (Gatekeeper → Especialista por classe).
- Especialistas (CIC): escolhidos via melhor **F1_k** por classe com desempate por latência.
- Dados: `data/train_cic.csv` (treino) e `data/cic_eval.csv` (avaliação).
- Métricas: F1 por classe, **F1-macro**, Acurácia; latências médias por estágio (ms/linha).

## 4) Interpretação (XAI — SHAP)
- Consolidados:
  - `reports/cic/xai/xai_shap_consolidado.csv`
  - `reports/cic/xai/xai_shap_consolidado.md`

> Observação: Importâncias globais e locais reforçam o pareamento (modelo + atributos) ótimo por classe no 2D-AEF.

## 5) Reprodutibilidade (comandos-chave)
```powershell
# Avaliação (já executada)
python -m twodaef.cli_eval_twostage `
  --gatekeeper_model artifacts\\gatekeeper_cic.joblib `
  --gatekeeper_features gatekeeper_cic_cols.txt `
  --specialist_map artifacts\\specialist_map_cic.json `
  --input_csv data\\cic_eval.csv `
  --label_col label `
  --output_dir outputs\\eval_cic `
  --fill_missing 0.0

# Plots
python -m twodaef.cli_plot_eval `
  --preds_csv outputs\\eval_cic\\preds.csv `
  --label_col label `
  --out_dir outputs\\eval_cic

# XAI (por especialista)
python -m twodaef.cli_explain_specialist `
  --specialist_map artifacts\\specialist_map_cic.json `
  --class_key Benign `
  --input_csv data\\cic_eval.csv `
  --output_dir outputs\\xai_cic `
  --limit_samples 200 `
  --top_k_global 12 `
  --top_k_local 12

python -m twodaef.cli_explain_specialist `
  --specialist_map artifacts\\specialist_map_cic.json `
  --class_key Others `
  --input_csv data\\cic_eval.csv `
  --output_dir outputs\\xai_cic `
  --limit_samples 200 `
  --top_k_global 12 `
  --top_k_local 12

# Agregação XAI
python -m twodaef.cli_xai_aggregate `
  --xai_root outputs\\xai_cic `
  --out_dir outputs\\xai_cic\\_consolidado
```
