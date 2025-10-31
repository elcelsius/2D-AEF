# src/twodaef/reports/plots_eval.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List, Tuple

import json
import itertools
import numpy as np
import pandas as pd
from loguru import logger
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    classification_report,
    accuracy_score,
)

def _ensure_outdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _to_str_array(series: pd.Series) -> np.ndarray:
    """Converte Series para vetor de strings (evita conflitos int/str)."""
    return series.astype(str).values

def _is_numeric_like(vals: List[str]) -> bool:
    for v in vals:
        s = str(v).strip()
        if s.startswith("-"):
            s = s[1:]
        if not s.isdigit():
            return False
    return True

def _try_align_spaces(y_true_str: np.ndarray, y_pred_str: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, str]]:
    """
    Se o espaço de rótulos de y_true (strings) e y_pred (strings) for disjunto,
    tenta alinhar automaticamente quando ambos são binários:
      - Se y_true tem 2 rótulos (p.ex. 'Benign','Others') e y_pred tem 2 rótulos numéricos ('0','1'),
        escolhe o mapeamento de {'0','1'} -> {labels_true} que maximiza a acurácia.
    Retorna: (y_true_str, y_pred_alinhado_str, mapping_aplicado)
    """
    uniq_true = sorted(set(y_true_str.tolist()))
    uniq_pred = sorted(set(y_pred_str.tolist()))

    # Se já houver interseção, não faz nada
    if len(set(uniq_true) & set(uniq_pred)) > 0:
        return y_true_str, y_pred_str, {}

    # Caso clássico: 2 classes vs 2 classes
    if len(uniq_true) == 2 and len(uniq_pred) == 2:
        # Tenta todas as bijeções B->A e escolhe a melhor
        best_acc = -1.0
        best_map: Dict[str, str] = {}
        for perm in itertools.permutations(uniq_true, 2):
            candidate_map = {uniq_pred[0]: perm[0], uniq_pred[1]: perm[1]}
            mapped = np.array([candidate_map[p] for p in y_pred_str], dtype=str)
            acc = accuracy_score(y_true_str, mapped)
            if acc > best_acc:
                best_acc = acc
                best_map = candidate_map

        if best_map:
            mapped = np.array([best_map[p] for p in y_pred_str], dtype=str)
            logger.info(f"Alinhamento automático aplicado (binário): {best_map} | acc={best_acc:.6f}")
            return y_true_str, mapped, best_map

    # Caso geral: tenta mapeamento por maioria para qualquer cardinalidade igual
    if len(uniq_true) == len(uniq_pred) and len(uniq_true) > 0:
        # constrói matriz de confusão entre espaços disjuntos e faz atribuição gulosa por maioria
        contingency = pd.crosstab(pd.Series(y_pred_str, name="pred"), pd.Series(y_true_str, name="true"))
        # pred_label -> true_label mais frequente
        candidate_map = {pred: contingency.loc[pred].idxmax() for pred in contingency.index if contingency.loc[pred].sum() > 0}
        if len(candidate_map) == len(uniq_pred):
            mapped = np.array([candidate_map.get(p, p) for p in y_pred_str], dtype=str)
            acc = accuracy_score(y_true_str, mapped)
            logger.info(f"Alinhamento automático aplicado (multi): {candidate_map} | acc={acc:.6f}")
            return y_true_str, mapped, candidate_map

    # Sem alinhamento
    return y_true_str, y_pred_str, {}

def _plot_confusion_matrix(cm: np.ndarray, labels: List[str], out_png: Path, title: str = "Confusion Matrix (2D-AEF)") -> None:
    plt.figure(figsize=(6, 5), dpi=150)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(values_format="d", cmap="Blues", colorbar=True)
    plt.title(title)
    plt.tight_layout()
    out_png.unlink(missing_ok=True)
    plt.savefig(out_png)
    plt.close()

def _plot_f1_per_class(y_true_str: np.ndarray, y_pred_str: np.ndarray, labels: List[str], out_png: Path, title: str = "F1 per class (2D-AEF)") -> None:
    rep = classification_report(y_true_str, y_pred_str, labels=labels, output_dict=True, zero_division=0)
    f1s = [float(rep.get(lbl, {}).get("f1-score", 0.0)) for lbl in labels]

    plt.figure(figsize=(8, 4), dpi=150)
    xs = np.arange(len(labels))
    plt.bar(xs, f1s)
    plt.xticks(xs, labels, rotation=30, ha="right")
    plt.ylim(0, 1.05)
    plt.ylabel("F1-score")
    plt.title(title)
    plt.tight_layout()
    out_png.unlink(missing_ok=True)
    plt.savefig(out_png)
    plt.close()

def make_eval_plots(preds_csv: str, label_col: str, out_dir: str, class_labels: List[str] | None = None) -> Dict[str, Any]:
    """
    Lê o preds.csv gerado pelo two-stage, recomputa métricas e salva:
      - confusion_matrix.png
      - f1_per_class.png
      - metrics_again.json (métricas recomputadas)

    Compatível com rótulos numéricos ou textuais. Se y_true e y_pred estiverem em espaços disjuntos
    (ex.: y_true='Benign/Others' e y_pred='0/1'), faz alinhamento automático.
    """
    outp = Path(out_dir)
    _ensure_outdir(outp)

    preds_path = Path(preds_csv)
    if not preds_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {preds_path}")

    df = pd.read_csv(preds_path)
    if label_col not in df.columns:
        raise KeyError(f"Coluna '{label_col}' não está presente em {preds_csv}")
    if "pred_final" not in df.columns:
        raise KeyError("Coluna 'pred_final' não está presente em preds.csv")

    # Converte para strings
    y_true_str = _to_str_array(df[label_col])
    y_pred_str = _to_str_array(df["pred_final"])

    # Tenta alinhar os espaços de rótulos, se necessário
    y_true_str, y_pred_str, applied_map = _try_align_spaces(y_true_str, y_pred_str)
    if applied_map:
        (Path(out_dir) / "label_space_mapping.json").write_text(json.dumps(applied_map, indent=2), encoding="utf-8")

    # Labels para gráficos: use APENAS os que aparecem em y_true (ordem alfabética)
    labels = sorted(set(y_true_str.tolist()))
    # Métricas globais
    f1_macro = float(f1_score(y_true_str, y_pred_str, average="macro"))
    acc = float(accuracy_score(y_true_str, y_pred_str))
    # Matriz de confusão restrita ao espaço de y_true
    cm = confusion_matrix(y_true_str, y_pred_str, labels=labels)

    # Plots
    _plot_confusion_matrix(cm, labels, outp / "confusion_matrix.png")
    _plot_f1_per_class(y_true_str, y_pred_str, labels, outp / "f1_per_class.png")

    # Salva métricas recomputadas
    payload = {
        "f1_macro": f1_macro,
        "accuracy": acc,
        "n": int(df.shape[0]),
        "labels": labels,
        "preds_csv": preds_csv,
        "out_dir": out_dir,
        "label_space_mapping": applied_map,
    }
    (outp / "metrics_again.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    logger.success(f"Plots salvos em {out_dir} (confusion_matrix.png, f1_per_class.png)")
    logger.info(f"F1-macro={f1_macro:.6f} | Acc={acc:.6f} | n={df.shape[0]}")
    return payload
