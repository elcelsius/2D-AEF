from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Tuple

import json
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score

from twodaef.infer.two_stage import TwoStageConfig, TwoStageInferencer


def run_eval_twostage(
    gatekeeper_model: str,
    gatekeeper_features_file: str,
    specialist_map_json: str,
    input_csv: str,
    label_col: str,
    output_dir: str,
    fill_missing: float = 0.0,
) -> Dict[str, Any]:
    """
    Executa a inferência two-stage e avalia contra a coluna de rótulo.
    Salva:
      - preds.csv (predições detalhadas)
      - metrics.json
      - classification_report.txt
      - confusion_matrix.csv
    Retorna dict com paths e métricas principais.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    preds_csv = out_dir / "preds.csv"

    # 1) Inferência two-stage
    cfg = TwoStageConfig(
        gatekeeper_model=gatekeeper_model,
        gatekeeper_features_file=gatekeeper_features_file,
        specialist_map_json=specialist_map_json,
        input_csv=input_csv,
        output_csv=str(preds_csv),
        fill_missing=fill_missing,
    )
    inf = TwoStageInferencer(cfg)
    gk_ms, s2_ms, tot_ms = inf.predict_csv()  # salva preds.csv

    # 2) Avaliação
    df = pd.read_csv(preds_csv)
    if label_col not in df.columns:
        # se o CSV de entrada tinha o label, o preds.csv também terá.
        # Se não tiver, falhar explicitamente.
        raise KeyError(f"Coluna de rótulo '{label_col}' não encontrada em {preds_csv}")

    y_true = df[label_col].values
    y_pred = df["pred_final"].values

    # métricas
    f1_macro = float(f1_score(y_true, y_pred, average="macro"))
    acc = float(accuracy_score(y_true, y_pred))

    # por classe
    report = classification_report(y_true, y_pred, digits=6)
    cm = confusion_matrix(y_true, y_pred)

    # salvar artefatos
    (out_dir / "classification_report.txt").write_text(report, encoding="utf-8")

    cm_df = pd.DataFrame(cm)
    cm_df.to_csv(out_dir / "confusion_matrix.csv", index=False)

    # Também salvar um resumo CSV com y_true/y_pred (útil pra debugging/plots)
    df_out = df[["pred_gatekeeper", "pred_final", "specialist_model", "specialist_featureset", label_col]].copy()
    df_out.to_csv(out_dir / "preds_min.csv", index=False)

    metrics = {
        "f1_macro": f1_macro,
        "accuracy": acc,
        "latency_ms_gatekeeper": round(gk_ms, 6),
        "latency_ms_specialist_mean": round(s2_ms, 6),
        "latency_ms_total_per_row": round(tot_ms, 6),
        "n_rows": int(df.shape[0]),
        "label_col": label_col,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    logger.success(f"F1-macro={f1_macro:.6f} | Acc={acc:.6f} | n={df.shape[0]}")
    logger.info(f"Latências (ms/linha): GK={gk_ms:.6f} | S2={s2_ms:.6f} | Total={tot_ms:.6f}")
    logger.success(f"Artefatos salvos em {out_dir}")

    return {
        "preds_csv": str(preds_csv),
        "preds_min_csv": str(out_dir / "preds_min.csv"),
        "metrics_json": str(out_dir / "metrics.json"),
        "report_txt": str(out_dir / "classification_report.txt"),
        "confusion_csv": str(out_dir / "confusion_matrix.csv"),
        **metrics,
    }
