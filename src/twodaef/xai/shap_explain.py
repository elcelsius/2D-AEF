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

def _compute_shap_values(model, X):
    """
    Retorna (shap_vals_2d, feature_names) no formato padronizado:
      - shape final: (n_amostras, n_features)
      - n_features == len(X.columns)
      - n_amostras == len(X) (se necessário, faz broadcast/recorte)
    Regras:
      * XGBoost -> SHAP 'permutation' em predict_proba (evita bug base_score)
      * LightGBM / árvores sklearn -> TreeExplainer
      * Genérico -> 'permutation'
    """
    import numpy as _np
    import shap

    feats = list(X.columns)
    p = len(feats)
    n = len(X)

    # Detecta tipo
    try:
        import xgboost as _xgb
        _is_xgb = isinstance(model, (_xgb.XGBClassifier, getattr(_xgb, "XGBRFClassifier", tuple())))
    except Exception:
        _is_xgb = False

    try:
        import lightgbm as _lgb
        _is_lgbm = isinstance(model, _lgb.LGBMClassifier)
    except Exception:
        _is_lgbm = False

    from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
    _is_sklearn_tree = isinstance(model, (RandomForestClassifier, HistGradientBoostingClassifier))

    def _to_2d(vals: _np.ndarray) -> _np.ndarray:
        """Normaliza para (n, p). Resolve eixos/classe; trunca/pad colunas; e garante n linhas."""
        vals = _np.asarray(vals)

        # (n, classes, feats) -> pega classe positiva
        if vals.ndim == 3:
            if vals.shape[1] > 1 and vals.shape[2] == p:
                vals = vals[:, 1, :]  # (n, p)
            elif vals.shape[2] > 1 and vals.shape[1] == p:
                vals = vals[:, 1, :].T  # (n, p) após ajuste
            else:
                if vals.shape[-1] == p:
                    vals = vals[:, 0, :]
                else:
                    vals = vals.reshape(vals.shape[0], -1)

        elif vals.ndim == 2:
            pass  # (n, p) ou (m, q)

        elif vals.ndim == 1:
            vals = vals.reshape(1, -1)

        else:
            vals = vals.reshape(vals.shape[0], -1)

        # Ajusta colunas para p
        if vals.shape[1] > p:
            vals = vals[:, :p]
        elif vals.shape[1] < p:
            pad_c = _np.zeros((vals.shape[0], p - vals.shape[1]), dtype=vals.dtype)
            vals = _np.concatenate([vals, pad_c], axis=1)

        # Ajusta linhas para n
        if vals.shape[0] == n:
            return vals
        if vals.shape[0] == 1:
            return _np.repeat(vals, n, axis=0)
        if vals.shape[0] < n:
            # repete a última linha até bater n
            reps = n - vals.shape[0]
            tail = _np.repeat(vals[-1:, :], reps, axis=0)
            return _np.concatenate([vals, tail], axis=0)
        # se veio com mais, recorta
        return vals[:n, :]

    if _is_xgb:
        # XGBoost: permutation em predict_proba
        bg_size = min(200, len(X))
        background = X.sample(bg_size, random_state=123) if len(X) > bg_size else X
        f = getattr(model, "predict_proba", None) or getattr(model, "predict", None)
        exp = shap.Explainer(f, background, algorithm="permutation")(X)
        vals = _to_2d(exp.values)

    elif _is_lgbm or _is_sklearn_tree:
        # LightGBM / sklearn trees: TreeExplainer
        exp = shap.TreeExplainer(model, feature_perturbation="interventional")(X)
        vals = exp.values
        if isinstance(vals, list):
            vals = vals[1] if len(vals) > 1 else vals[0]
        vals = _to_2d(vals)

    else:
        # Genérico: permutation
        bg_size = min(200, len(X))
        background = X.sample(bg_size, random_state=123) if len(X) > bg_size else X
        f = getattr(model, "predict_proba", None) or getattr(model, "predict", None)
        exp = shap.Explainer(f, background, algorithm="permutation")(X)
        vals = _to_2d(exp.values)

    return vals, feats

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
    Garante alinhamento: shap_vals.shape[1] == len(feature_names) == X.shape[1]
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Alinhamento robusto
    p = X.shape[1]
    if shap_vals.ndim == 1:
        shap_vals = np.reshape(shap_vals, (1, -1))
    if shap_vals.shape[1] != p:
        # Reajusta para colunas de X
        if shap_vals.shape[0] == p:
            shap_vals = shap_vals.T
        elif shap_vals.shape[1] > p:
            shap_vals = shap_vals[:, :p]
        elif shap_vals.shape[1] < p:
            pad = np.zeros((shap_vals.shape[0], p - shap_vals.shape[1]), dtype=shap_vals.dtype)
            shap_vals = np.concatenate([shap_vals, pad], axis=1)

    if len(feature_names) != p:
        feature_names = list(X.columns)

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
    Garante alinhamento de linhas e colunas, e evita index errors.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Alinhamento de colunas com X
    p = X.shape[1]
    if shap_vals.ndim == 1:
        shap_vals = np.reshape(shap_vals, (1, -1))
    if shap_vals.shape[1] != p:
        if shap_vals.shape[1] > p:
            shap_vals = shap_vals[:, :p]
        else:
            pad = np.zeros((shap_vals.shape[0], p - shap_vals.shape[1]), dtype=shap_vals.dtype)
            shap_vals = np.concatenate([shap_vals, pad], axis=1)
    if len(feature_names) != p:
        feature_names = list(X.columns)

    # Alinhamento de linhas com X
    n = X.shape[0]
    m = shap_vals.shape[0]
    if m == 1 and n > 1:
        shap_vals = np.repeat(shap_vals, n, axis=0)
        m = n
    elif m < n:
        tail = np.repeat(shap_vals[-1:, :], n - m, axis=0)
        shap_vals = np.concatenate([shap_vals, tail], axis=0)
        m = n
    elif m > n:
        shap_vals = shap_vals[:n, :]
        m = n

    absvals = np.abs(shap_vals)
    for i in range(n):
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
