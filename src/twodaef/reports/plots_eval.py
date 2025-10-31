from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import json
import numpy as np
import pandas as pd
from loguru import logger
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score


def _ensure_outdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _try_align_spaces(y_true_raw: np.ndarray, y_pred_raw: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Optional[Dict[str, str]]]:
    """
    Tenta alinhar espaços de rótulos quando tipos diferem (ex.: y_true=['Benign','Others'] e y_pred=[0,1]).
    Implementa heurística binária 0↔Benign/1↔Others.
    Retorna (y_true_int, y_pred_int, mapping_str) onde mapping_str (opcional) descreve o mapeamento feito.
    """
    # já inteiros?
    def _to_int_safe(a):
        try:
            return a.astype(int), True
        except Exception:
            return a, False

    y_true_int, ok_true = _to_int_safe(y_true_raw)
    y_pred_int, ok_pred = _to_int_safe(y_pred_raw)

    if ok_true and ok_pred:
        return y_true_int, y_pred_int, None

    # Heurística binária
    # mapeia strings -> ints
    def _str2bin(x: Any) -> Optional[int]:
        s = str(x).strip().lower()
        if s in {"0", "1"}:
            return int(s)
        if any(t in s for t in ("benign", "normal", "clean", "legit")):
            return 0
        if any(t in s for t in ("mal", "attack", "other", "others", "anomaly", "intrusion")):
            return 1
        return None

    def _map_array(a: np.ndarray) -> Tuple[np.ndarray, bool]:
        out = []
        for v in a:
            try:
                out.append(int(v))
            except Exception:
                b = _str2bin(v)
                if b is None:
                    return a, False
                out.append(b)
        return np.asarray(out, dtype=int), True

    y_true_m, ok_true_m = _map_array(y_true_raw)
    y_pred_m, ok_pred_m = _map_array(y_pred_raw)

    if ok_true_m and ok_pred_m:
        mapping = {"0": "Benign", "1": "Others"}
        acc = float(accuracy_score(y_true_m, y_pred_m))
        logger.info(f"Alinhamento automático aplicado (binário): {mapping} | acc={acc:.6f}")
        return y_true_m, y_pred_m, mapping

    # fallback: nada a fazer
    return y_true_raw, y_pred_raw, None


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
    uniq = sorted(set(y_true) | set(y_pred))
    for li in uniq:
        mask_pos = (y_true == li)
        if mask_pos.sum() == 0:
            f1s.append(0.0)
        else:
            y_true_bin = (y_true == li).astype(int)
            y_pred_bin = (y_pred == li).astype(int)
            f1s.append(f1_score(y_true_bin, y_pred_bin))

    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    ax.bar(range(len(uniq)), f1s)
    ax.set_xticks(range(len(uniq)))
    # mostrar rótulos amigáveis
    try:
        xticks = [labels[li] if li < len(labels) else str(li) for li in uniq]
    except Exception:
        xticks = [str(li) for li in uniq]
    ax.set_xticklabels(xticks, rotation=45, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("F1-score")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def _resolve_preds_csv(preds_csv: str | None, dataset_tag: str | None) -> Path:
    """
    Resolve automaticamente o caminho do preds.csv quando:
    - preds_csv é fornecido: usa direto (precisa existir).
    - caso contrário, tenta usar dataset_tag para achar:
        outputs/eval_<tag>/preds.csv
        reports/<TAG>/preds.csv
    """
    if preds_csv:
        p = Path(preds_csv)
        if not p.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {p}")
        return p

    if not dataset_tag:
        raise ValueError("Forneça --preds_csv ou --dataset_tag.")

    tag = dataset_tag.strip().lower()
    candidates = [
        Path("outputs") / f"eval_{tag}" / "preds.csv",
        Path("reports") / tag / "preds.csv",
        Path("reports") / tag.upper() / "preds.csv",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"Não encontrei preds.csv para dataset_tag='{dataset_tag}'. "
                            f"Tente passar --preds_csv explicitamente.")


def make_eval_plots(
    preds_csv: str | None,
    label_col: str,
    out_dir: str,
    dataset_tag: str | None = None,
    class_labels: list[str] | None = None
) -> Dict[str, Any]:
    """
    Lê o preds.csv gerado pelo two-stage, recomputa métricas e salva:
      - confusion_matrix[_<tag>].png
      - f1_per_class[_<tag>].png
      - metrics_again.json (métricas recomputadas)
    """
    preds_csv_p = _resolve_preds_csv(preds_csv, dataset_tag)
    outp = Path(out_dir)
    _ensure_outdir(outp)

    df = pd.read_csv(preds_csv_p)
    if label_col not in df.columns:
        raise KeyError(f"Coluna '{label_col}' não está presente em {preds_csv_p}")

    # colunas obrigatórias
    if "pred_final" not in df.columns:
        raise KeyError(f"Coluna 'pred_final' não está presente em {preds_csv_p}")

    y_true_raw = df[label_col].values
    y_pred_raw = df["pred_final"].values

    # alinhar espaços/formatos
    y_true, y_pred, mapping = _try_align_spaces(y_true_raw, y_pred_raw)

    # Labels (legendas) para o gráfico
    uniq = sorted(set(y_true) | set(y_pred))
    if class_labels is not None:
        labels = class_labels
    else:
        if mapping:  # binário com rótulos amigáveis
            labels = ["Benign", "Others"]
        else:
            labels = [str(u) for u in uniq]

    # Métricas
    cm = confusion_matrix(y_true, y_pred, labels=uniq)
    f1_macro = float(f1_score(y_true, y_pred, average="macro"))
    acc = float(accuracy_score(y_true, y_pred))

    # Sufixo de arquivo por dataset
    suffix = f"_{dataset_tag.lower()}" if dataset_tag else ""
    cm_png = outp / f"confusion_matrix{suffix}.png"
    f1_png = outp / f"f1_per_class{suffix}.png"

    # Plots
    plot_confusion_matrix(cm, labels, cm_png, title=f"Confusion Matrix (2D-AEF{suffix})")
    plot_f1_per_class(y_true, y_pred, labels, f1_png, title=f"F1 per class (2D-AEF{suffix})")

    # Salva métricas recomputadas
    payload = {
        "f1_macro": f1_macro,
        "accuracy": acc,
        "n": int(df.shape[0]),
        "labels": labels,
        "preds_csv": preds_csv_p.as_posix(),
        "out_dir": outp.as_posix(),
        "dataset_tag": dataset_tag or "",
    }
    (outp / "metrics_again.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    logger.success(f"Plots salvos em {outp} ({cm_png.name}, {f1_png.name})")
    logger.info(f"F1-macro={f1_macro:.6f} | Acc={acc:.6f} | n={df.shape[0]}")
    return payload
