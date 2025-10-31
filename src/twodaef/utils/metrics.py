from __future__ import annotations
from typing import Dict, Any, Sequence
import numpy as np
from sklearn.metrics import f1_score

def f1_per_class(y_true: Sequence, y_pred: Sequence) -> Dict[Any, float]:
    """Retorna F1 por classe (macro de 1-vs-rest por rótulo)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    classes = np.unique(y_true)
    out: Dict[Any, float] = {}
    for c in classes:
        y_true_bin = (y_true == c).astype(int)
        y_pred_bin = (y_pred == c).astype(int)
        out[c] = f1_score(y_true_bin, y_pred_bin, zero_division=0)
    return out
