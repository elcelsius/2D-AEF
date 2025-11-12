from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
from loguru import logger

# I/O helpers centralizados (passo 1 já criado)
from twodaef.utils.io import (
    read_csv_utf8,
    write_json_utf8,
    ensure_dir,
)

import joblib

# SHAP e LIME
import shap
from matplotlib import pyplot as plt
from lime.lime_tabular import LimeTabularExplainer


# -----------------------------
# Helpers de preparo de dados
# -----------------------------

def _ensure_columns(df, cols: List[str], fill: float = 0.0):
    import pandas as pd
    df = df.copy()
    for c in cols:
        if c not in df.columns:
            df[c] = fill
    return df[cols]


def _is_tree_model(model: Any) -> bool:
    """
    True para modelos suportados nativamente pelo TreeExplainer (LightGBM,
    XGBoost, CatBoost, árvores/ensembles do sklearn).
    """
    name = model.__class__.__name__.lower()
    return any(k in name for k in [
        "lgbm", "lightgbm", "xgb", "xgboost", "catboost",
        "randomforest", "extratrees", "histgradientboosting", "gradientboosting"
    ])


# -----------------------------
# SHAP — cálculo robusto
# -----------------------------

def _compute_shap_values(model, X):
    """
    Retorna (shap_values_2d, feature_names)
    Regras:
      - XGBoost: usa SHAP com algorithm='permutation' sobre predict_proba
        (evita bug do base_score e mudanças de formato do TreeExplainer).
      - LightGBM / Árvores sklearn: TreeExplainer ('interventional').
      - Genérico: permutation explainer.
    Normaliza a saída para shape (n_amostras, n_features).
    """
    import shap as _shap
    import numpy as _np

    feats = list(X.columns)

    # Detecta tipos
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

    from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, ExtraTreesClassifier, GradientBoostingClassifier
    _is_sklearn_tree = isinstance(model, (RandomForestClassifier, HistGradientBoostingClassifier, ExtraTreesClassifier, GradientBoostingClassifier))

    def _normalize_vals(vals: _np.ndarray) -> _np.ndarray:
        """
        Normaliza qualquer saída para (n, feats).
        Possíveis formatos:
          - (n, feats)
          - (n, classes, feats)  -> pega classe positiva (índice 1) se existir; senão 0
          - (n, feats, classes)  -> idem
        """
        v = _np.asarray(vals)
        if v.ndim == 2:
            return v
        if v.ndim == 3:
            n, a, b = v.shape
            # caso  (n, classes, feats)
            if a <= 10 and b > 10:
                c_idx = 1 if a > 1 else 0
                return v[:, c_idx, :]
            # caso  (n, feats, classes)
            if b <= 10 and a > 10:
                c_idx = 1 if b > 1 else 0
                return v[:, :, c_idx]
            # fallback
            v = _np.squeeze(v)
            if v.ndim == 2:
                return v
        v = _np.squeeze(v)
        if v.ndim != 2:
            raise RuntimeError(f"Formato inesperado de SHAP values: {v.shape}")
        return v

    if _is_xgb:
        # ---- XGBoost: permutation explainer com predict_proba ----
        bg_size = min(200, len(X))
        background = X.sample(bg_size, random_state=123) if len(X) > bg_size else X

        f = getattr(model, "predict_proba", None) or getattr(model, "predict", None)
        explainer = _shap.Explainer(f, background, algorithm="permutation")
        exp = explainer(X)
        vals = _normalize_vals(exp.values)

    elif _is_lgbm or _is_sklearn_tree:
        # ---- LightGBM / Sklearn: TreeExplainer interventional ----
        explainer = _shap.TreeExplainer(model, feature_perturbation="interventional")
        exp = explainer(X)
        vals = exp.values
        if isinstance(vals, list):  # versões novas retornam lista por classe
            vals = vals[1] if len(vals) > 1 else vals[0]
        vals = _normalize_vals(vals)

    else:
        # ---- Genérico: permutation ----
        bg_size = min(200, len(X))
        background = X.sample(bg_size, random_state=123) if len(X) > bg_size else X
        f = getattr(model, "predict_proba", None) or getattr(model, "predict", None)
        explainer = _shap.Explainer(f, background, algorithm="permutation")
        exp = explainer(X)
        vals = _normalize_vals(exp.values)

    # Garantir mesma contagem entre features e shap
    n_feats = vals.shape[1]
    if n_feats != len(feats):
        m = min(n_feats, len(feats))
        logger.warning(f"Ajustando desalinhamento SHAP/features: shap={n_feats}, feats={len(feats)} -> usando {m}.")
        vals = vals[:, :m]
        feats = feats[:m]

    return vals, feats


# -----------------------------
# Persistência de outputs SHAP
# -----------------------------

def _save_summary_outputs(
    out_dir: Path,
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
    ensure_dir(out_dir)
    import pandas as pd

    # Alinha dimensões por segurança
    m = min(shap_vals.shape[1], len(feature_names))
    sv = shap_vals[:, :m]
    fn = feature_names[:m]

    mean_abs = np.mean(np.abs(sv), axis=0)
    df_imp = pd.DataFrame({
        "feature": fn,
        "mean_abs_shap": mean_abs
    }).sort_values("mean_abs_shap", ascending=False)

    csv_path = out_dir / "summary_mean_abs_shap.csv"
    df_imp.to_csv(csv_path, index=False, encoding="utf-8")

    if plot_png and len(fn) > 0:
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
    X,
    shap_vals: np.ndarray,
    feature_names: List[str],
    top_k: int = 10,
) -> None:
    """
    Para cada amostra, salva um CSV com as top-k contribuições absolutas.
    Arquivo: sample_{idx:05d}_topk.csv
    """
    ensure_dir(out_dir)
    import pandas as pd

    m = min(shap_vals.shape[1], len(feature_names))
    absvals = np.abs(shap_vals[:, :m])

    for i in range(absvals.shape[0]):
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
    X,
    out_dir: Path,
    top_k_global: int = 10,
    top_k_local: int = 10,
) -> None:
    shap_vals, feats = _compute_shap_values(model, X)
    _save_summary_outputs(out_dir, shap_vals, feats, top_k=top_k_global, plot_png=True)
    _save_per_sample_topk(out_dir, X, shap_vals, feats, top_k=top_k_local)


# -----------------------------
# LIME (fallback)
# -----------------------------

def explain_with_lime(
    model: Any,
    X,
    out_dir: Path,
    class_names: Optional[List[str]] = None,
    num_features: int = 10,
    max_samples: int = 20,
) -> None:
    """
    Fallback quando SHAP (TreeExplainer/permutation) não se aplica.
    Gera explicações LIME por amostra (limitando max_samples).
    """
    ensure_dir(out_dir)
    import numpy as _np

    X_bg = X.values
    explainer = LimeTabularExplainer(
        X_bg,
        feature_names=list(X.columns),
        class_names=class_names if class_names else None,
        discretize_continuous=True
    )
    n = min(X.shape[0], max_samples)

    # tenta usar predict_proba; se não houver, cria pseudo-proba
    pred_proba = getattr(model, "predict_proba", None)
    if pred_proba is None:
        def _pseudo_proba(arr):
            preds = model.predict(arr)
            preds = preds.astype(float)
            return _np.column_stack([1.0 - preds, preds])
        pred_fn = _pseudo_proba
    else:
        pred_fn = pred_proba

    for i in range(n):
        exp = explainer.explain_instance(X.iloc[i].values, pred_fn, num_features=num_features)
        df_exp = _lime_to_df(exp)
        (out_dir / f"lime_sample_{i:05d}.csv").write_text(df_exp.to_csv(index=False), encoding="utf-8")


def _lime_to_df(exp) -> "pandas.DataFrame":
    import pandas as pd
    pairs = exp.as_list()  # [(feature, weight), ...]
    return pd.DataFrame(pairs, columns=["feature", "weight"])


# -----------------------------
# Pipeline por especialista
# -----------------------------

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
    Carrega o especialista da classe (pela chave do mapa), prepara X com as features dedicadas,
    executa SHAP (se suportado), senão LIME, e salva artefatos em output_dir/class_{key}.
    """
    import json

    spec_root = Path(specialist_map_json)
    d = json.loads(spec_root.read_text(encoding="utf-8"))
    spec_map = d.get("specialists", {})

    if class_key not in spec_map:
        raise KeyError(f"Classe '{class_key}' não encontrada no mapa de especialistas.")
    payload = spec_map[class_key]

    model_path = Path(payload["model_path"])
    feats = list(payload["features"])
    if not model_path.exists():
        raise FileNotFoundError(f"Model path não encontrado: {model_path}")

    model = joblib.load(model_path)

    df = read_csv_utf8(input_csv)
    X = _ensure_columns(df, feats, fill=fill_missing)

    if limit_samples is not None and X.shape[0] > limit_samples:
        X = X.sample(n=limit_samples, random_state=0)

    out_dir = Path(output_dir) / f"class_{class_key}"
    ensure_dir(out_dir)

    # Salva metadados
    meta = {
        "class_key": class_key,
        "model_path": str(model_path),
        "features": feats,
        "n_rows": int(X.shape[0]),
        "output_dir": str(out_dir),
    }
    write_json_utf8(meta, out_dir / "meta.json")

    # Escolha de método
    if _is_tree_model(model):
        logger.info(f"SHAP (TreeExplainer/permutation) para classe {class_key} — n_amostras={X.shape[0]} / n_feats={X.shape[1]}")
        explain_with_shap(model, X, out_dir, top_k_global=top_k_global, top_k_local=top_k_local)
        method = "shap_tree_or_perm"
    else:
        logger.info(f"LIME (fallback) para classe {class_key} — n_amostras={X.shape[0]} / n_feats={X.shape[1]}")
        explain_with_lime(model, X, out_dir, class_names=None, num_features=top_k_local, max_samples=min(50, X.shape[0]))
        method = "lime_fallback"

    (out_dir / "method.txt").write_text(method, encoding="utf-8")
    logger.success(f"XAI salvo em {out_dir}")
    return {"method": method, **meta}
