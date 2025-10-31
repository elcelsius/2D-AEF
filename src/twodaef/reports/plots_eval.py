from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Tuple

import json
import numpy as np
import pandas as pd
from loguru import logger
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score


def _ensure_outdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def plot_confusion_matrix(cm: np.ndarray, labels: list[str], out_png: Path, title: str = "Confusion Matrix") -> None:
    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    # valores por célula
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def plot_f1_per_class(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str], out_png: Path, title: str = "F1 per class") -> None:
    # calcula f1 individual por label na ordem de `labels`
    f1s = []
    for li in range(len(labels)):
        mask_pos = (y_true == li)
        # evita divisão por zero em classe ausente
        if mask_pos.sum() == 0:
            f1s.append(0.0)
        else:
            y_true_bin = (y_true == li).astype(int)
            y_pred_bin = (y_pred == li).astype(int)
            f1s.append(f1_score(y_true_bin, y_pred_bin))

    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    ax.bar(range(len(labels)), f1s)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("F1-score")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def make_eval_plots(preds_csv: str, label_col: str, out_dir: str, class_labels: list[str] | None = None) -> Dict[str, Any]:
    """
    Lê o preds.csv gerado pelo two-stage, recomputa métricas e salva:
      - confusion_matrix.png
      - f1_per_class.png
      - metrics_again.json (métricas recomputadas)
    """
    outp = Path(out_dir)
    _ensure_outdir(outp)

    df = pd.read_csv(preds_csv)
    if label_col not in df.columns:
        raise KeyError(f"Coluna '{label_col}' não está presente em {preds_csv}")

    # Converte para inteiros (garante consistência)
    y_true = df[label_col].astype(int).values
    y_pred = df["pred_final"].astype(int).values

    # Labels padrão: [0, 1, 2, ...] como string
    uniq = sorted(set(y_true) | set(y_pred))
    if class_labels is None:
        labels = [str(u) for u in uniq]
    else:
        # se forneceram labels, use-os; caso contrário, padroniza pelo uniq
        labels = class_labels

    # Métricas
    cm = confusion_matrix(y_true, y_pred, labels=uniq)
    f1_macro = float(f1_score(y_true, y_pred, average="macro"))
    acc = float(accuracy_score(y_true, y_pred))

    # Plots
    plot_confusion_matrix(cm, labels, outp / "confusion_matrix.png", title="Confusion Matrix (2D-AEF)")
    plot_f1_per_class(y_true, y_pred, labels, outp / "f1_per_class.png", title="F1 per class (2D-AEF)")

    # Salva métricas recomputadas
    payload = {
        "f1_macro": f1_macro,
        "accuracy": acc,
        "n": int(df.shape[0]),
        "labels": labels,
        "preds_csv": preds_csv,
        "out_dir": out_dir,
    }
    (outp / "metrics_again.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    logger.success(f"Plots salvos em {out_dir} (confusion_matrix.png, f1_per_class.png)")
    logger.info(f"F1-macro={f1_macro:.6f} | Acc={acc:.6f} | n={df.shape[0]}")
    return payload
