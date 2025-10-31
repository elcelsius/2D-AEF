from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import json
import numpy as np
import pandas as pd
from loguru import logger

import joblib  # <— usa joblib direto para carregar o modelo

# SHAP e LIME
import shap
from matplotlib import pyplot as plt
from lime.lime_tabular import LimeTabularExplainer


def _ensure_columns(df: pd.DataFrame, cols: List[str], fill: float = 0.0) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c not in df.columns:
            df[c] = fill
    return df[cols]


def _is_tree_model(model: Any) -> bool:
    """True para modelos suportados nativamente pelo TreeExplainer (LightGBM, XGBoost, CatBoost, sklearn tree ensembles)."""
    name = model.__class__.__name__.lower()
    return any(k in name for k in [
        "lgbm", "lightgbm", "xgb", "xgboost", "catboost",
        "randomforest", "extratrees", "histgradientboosting", "gradientboosting"
    ])


def _compute_shap_values(model: Any, X: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    """
    Retorna (shap_values_2d, feature_names). Para binário:
      - alguns modelos retornam [arr_neg, arr_pos]; pegamos o da classe positiva (índice 1) se for lista.
      - outros retornam um array 2D diretamente.
    Usa background explícito para o modo interventional.
    """
    # background com no máx. 256 amostras representativas
    bg_rows = min(256, max(32, int(len(X) * 0.1)))
    background = shap.sample(X, bg_rows, random_state=0)

    explainer = shap.TreeExplainer(
        model,
        data=background,                    # <- background explícito
        feature_perturbation="interventional"
    )
    vals = explainer.shap_values(X)

    if isinstance(vals, list):
        shap_2d = vals[1] if len(vals) > 1 else vals[-1]
    else:
        shap_2d = vals

    shap_2d = np.asarray(shap_2d, dtype=float)
    return shap_2d, list(X.columns)

def _save_summary_outputs(
    out_dir: Path,
    X: pd.DataFrame,
    shap_vals: np.ndarray,
    feature_names: List[str],
    top_k: int = 10,
    plot_png: bool = True,
) -> Path:
    """
    Salva:
      - summary_mean_abs_shap.csv (importância global |SHAP| média)
      - summary_bar.png (barplot SHAP) [opcional]
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    mean_abs = np.mean(np.abs(shap_vals), axis=0)
    df_imp = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs
    }).sort_values("mean_abs_shap", ascending=False)

    csv_path = out_dir / "summary_mean_abs_shap.csv"
    df_imp.to_csv(csv_path, index=False)

    if plot_png and len(feature_names) > 0:
        top = df_imp.head(top_k)
        plt.figure()
        plt.barh(top["feature"][::-1], top["mean_abs_shap"][::-1])
        plt.xlabel("|SHAP| médio")
        plt.ylabel("feature")
        plt.title("Importância global (|SHAP| médio)")
        plt.tight_layout()
        png_path = out_dir / "summary_bar.png"
        plt.savefig(png_path, dpi=150)
        plt.close()

    return csv_path


def _save_per_sample_topk(
    out_dir: Path,
    X: pd.DataFrame,
    shap_vals: np.ndarray,
    feature_names: List[str],
    top_k: int = 10,
) -> None:
    """
    Para cada amostra, salva um CSV com as top-k contribuições absolutas.
    Arquivo: sample_{idx:05d}_topk.csv
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    absvals = np.abs(shap_vals)
    for i in range(X.shape[0]):
        idx_sort = np.argsort(-absvals[i])[:top_k]
        df_i = pd.DataFrame({
            "feature": [feature_names[j] for j in idx_sort],
            "shap_value": [shap_vals[i, j] for j in idx_sort],
            "abs_shap": [absvals[i, j] for j in idx_sort],
            "value": [X.iloc[i, j] for j in idx_sort],
        })
        (out_dir / f"sample_{i:05d}_topk.csv").write_text(df_i.to_csv(index=False), encoding="utf-8")


def explain_with_shap(
    model: Any,
    X: pd.DataFrame,
    out_dir: Path,
    top_k_global: int = 10,
    top_k_local: int = 10,
) -> None:
    shap_vals, feats = _compute_shap_values(model, X)
    _save_summary_outputs(out_dir, X, shap_vals, feats, top_k=top_k_global, plot_png=True)
    _save_per_sample_topk(out_dir, X, shap_vals, feats, top_k=top_k_local)


def explain_with_lime(
    model: Any,
    X: pd.DataFrame,
    out_dir: Path,
    class_names: Optional[List[str]] = None,
    num_features: int = 10,
    max_samples: int = 20,
) -> None:
    """
    Fallback quando SHAP (TreeExplainer) não se aplica.
    Gera explicações LIME por amostra (limitando max_samples).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    X_bg = X.values
    explainer = LimeTabularExplainer(
        X_bg,
        feature_names=list(X.columns),
        class_names=class_names if class_names else None,
        discretize_continuous=True
    )
    n = min(X.shape[0], max_samples)
    for i in range(n):
        exp = explainer.explain_instance(
            X.iloc[i].values, model.predict_proba, num_features=num_features
        )
        df_exp = pd.DataFrame(exp.as_list(), columns=["feature", "weight"])
        (out_dir / f"lime_sample_{i:05d}.csv").write_text(df_exp.to_csv(index=False), encoding="utf-8")


def run_xai_for_specialist(
    specialist_map_json: str,
    class_key: str,
    input_csv: str,
    output_dir: str,
    fill_missing: float = 0.0,
    limit_samples: Optional[int] = 200,
    top_k_global: int = 10,
    top_k_local: int = 10,
) -> Dict[str, Any]:
    """
    Carrega o especialista da classe (pelo key do mapa), prepara X com as features dedicadas,
    executa SHAP (se suportado), senão LIME, e salva artefatos em output_dir/class_{key}.
    """
    spec_map = json.loads(Path(specialist_map_json).read_text(encoding="utf-8")).get("specialists", {})
    if class_key not in spec_map:
        raise KeyError(f"Classe '{class_key}' não encontrada no mapa de especialistas.")
    payload = spec_map[class_key]
    model_path = Path(payload["model_path"])
    feats = list(payload["features"])
    if not model_path.exists():
        raise FileNotFoundError(f"Model path não encontrado: {model_path}")

    model = joblib.load(model_path)  # <— correção aqui

    df = pd.read_csv(input_csv)
    X = _ensure_columns(df, feats, fill=fill_missing)

    if limit_samples is not None and X.shape[0] > limit_samples:
        X = X.sample(n=limit_samples, random_state=0)

    out_dir = Path(output_dir) / f"class_{class_key}"
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "class_key": class_key,
        "model_path": str(model_path),
        "features": feats,
        "n_rows": int(X.shape[0]),
        "output_dir": str(out_dir),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    if _is_tree_model(model):
        logger.info(f"SHAP (TreeExplainer) para classe {class_key} — n_amostras={X.shape[0]} / n_feats={X.shape[1]}")
        explain_with_shap(model, X, out_dir, top_k_global=top_k_global, top_k_local=top_k_local)
        method = "shap_tree"
    else:
        logger.info(f"LIME (fallback) para classe {class_key} — n_amostras={X.shape[0]} / n_feats={X.shape[1]}")
        explain_with_lime(model, X, out_dir, class_names=None, num_features=top_k_local, max_samples=min(50, X.shape[0]))
        method = "lime_fallback"

    (out_dir / "method.txt").write_text(method, encoding="utf-8")
    logger.success(f"XAI salvo em {out_dir}")
    return {"method": method, **meta}
