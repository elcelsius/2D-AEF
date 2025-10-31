# outputs/

Artefatos **temporários ou reprodutíveis** (predições, métricas, plots, XAI).

- Este diretório é ignorado no git, exceto READMEs/MDs.
- Exemplo:
```
outputs/
├─ eval_unsw/
│  ├─ preds.csv
│  ├─ confusion_matrix.png
│  ├─ f1_per_class.png
│  └─ metrics_again.json
├─ eval_cic/
│  ├─ preds.csv
│  ├─ confusion_matrix.png
│  ├─ f1_per_class.png
│  └─ metrics_again.json
├─ xai_unsw/
│  ├─ class_0/ summary_mean_abs_shap.csv ...
│  └─ _consolidado/ xai_shap_consolidado.(csv|md)
└─ xai_cic/
   ├─ class_Benign/ summary_mean_abs_shap.csv ...
   └─ _consolidado/ xai_shap_consolidado.(csv|md)
```
