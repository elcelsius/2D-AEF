# src/twodaef/eval/evaluate.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List, Any, Dict

import json
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report


@dataclass
class EvalConfig:
    preds_csv: str            # outputs\eval_xxx\preds.csv (gerado pelo two_stage)
    label_col: str            # nome da coluna-verdade (ex.: "label")
    specialist_map: Optional[str] = None  # artifacts\specialist_map_*.json (opcional)
    out_dir: Optional[str] = None         # pasta para salvar artefatos (png/csv/metrics.json)


def _load_classes_from_map(path: Optional[str]) -> Optional[List[str]]:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        logger.warning(f"specialist_map não encontrado: {p}")
        return None
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        classes = d.get("classes")
        if isinstance(classes, list) and classes:
            # normaliza para str
            return [str(x) for x in classes]
    except Exception as e:
        logger.warning(f"Falha ao ler specialist_map {p}: {e}")
    return None


def _coerce_label_types(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: Optional[List[str]] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Alinha tipos de rótulo para permitir cálculo de métricas:
    - Se y_true é texto (ex.: 'Benign', 'Others') e y_pred é numérico (0/1),
      e houver `classes`, mapeia y_pred -> nomes via índice.
    - Caso contrário, como fallback, converte ambos para string.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    true_is_str = y_true.dtype.kind in {"U", "S", "O"}  # unicode/bytes/object => provável string
    pred_is_num = np.issubdtype(y_pred.dtype, np.number)

    if true_is_str and pred_is_num and classes is not None:
        # mapeia inteiros para nomes de classes pelo índice
        max_idx = int(np.max(y_pred)) if y_pred.size else -1
        if max_idx >= len(classes):
            logger.warning(
                f"Predições possuem índice ({max_idx}) fora do range de classes ({len(classes)}). "
                "Aplicando fallback para string."
            )
        else:
            mapped = []
            for v in y_pred:
                try:
                    mapped.append(classes[int(v)])
                except Exception:
                    mapped.append(str(v))
            return y_true.astype(str), np.asarray(mapped, dtype=str)

    # fallback robusto
    return y_true.astype(str), y_pred.astype(str)


def run_eval_twostage(
    preds_csv: str,
    label_col: str,
    specialist_map: Optional[str] = None,
    out_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Lê o preds.csv (two-stage), alinha rótulos e calcula métricas + artefatos.
    Retorna dict com métricas principais.
    """
    p = Path(preds_csv)
    if not p.exists():
        raise FileNotFoundError(f"preds_csv não encontrado: {p}")

    df = pd.read_csv(p, low_memory=False)
    if label_col not in df.columns:
        raise RuntimeError(f"Coluna de rótulo '{label_col}' não encontrada em {p}")

    if "pred_final" not in df.columns:
        raise RuntimeError(f"Coluna 'pred_final' não encontrada em {p}. Gere com two_stage.")

    y_true_raw = df[label_col].values
    y_pred_raw = df["pred_final"].values

    classes = _load_classes_from_map(specialist_map)
    y_true, y_pred = _coerce_label_types(y_true_raw, y_pred_raw, classes)

    # métricas
    f1_macro = float(f1_score(y_true, y_pred, average="macro"))
    acc = float(accuracy_score(y_true, y_pred))
    n = int(df.shape[0])

    logger.success(f"F1-macro={f1_macro:.6f} | Acc={acc:.6f} | n={n}")

    # artefatos opcionais
    out = Path(out_dir) if out_dir else p.parent
    out.mkdir(parents=True, exist_ok=True)

    # salvar confusion matrix e relatório em CSV/TXT
    labels_sorted = sorted(sorted(np.unique(y_true).tolist() + np.unique(y_pred).tolist(), key=str))
    cm = confusion_matrix(y_true, y_pred, labels=labels_sorted)
    cm_df = pd.DataFrame(cm, index=[f"true_{x}" for x in labels_sorted], columns=[f"pred_{x}" for x in labels_sorted])
    cm_df.to_csv(out / "confusion_matrix_eval.csv", index=True, encoding="utf-8")

    cls_rep = classification_report(y_true, y_pred, labels=labels_sorted, zero_division=0, output_dict=True)
    pd.DataFrame(cls_rep).to_csv(out / "classification_report_eval.csv", encoding="utf-8")

    # salvar métricas em JSON
    metrics = {
        "f1_macro": f1_macro,
        "accuracy": acc,
        "n_samples": n,
        "labels": labels_sorted,
        "classes_from_map": classes,
        "preds_csv": str(p),
    }
    (out / "metrics_eval.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    logger.success(f"Artefatos de avaliação salvos em {out}")
    return metrics
