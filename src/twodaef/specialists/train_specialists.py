from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Tuple
import json
import time
import importlib

import joblib
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder

from twodaef.utils.metrics import f1_per_class

# ---------- Modelo Factory (dinâmico) ----------
def _available_models() -> Dict[str, Any]:
    models: Dict[str, Any] = {}
    # Try LightGBM
    if importlib.util.find_spec("lightgbm"):
        from lightgbm import LGBMClassifier
        models["lgbm"] = lambda: LGBMClassifier(
            n_estimators=300, learning_rate=0.05, num_leaves=63, subsample=0.8, colsample_bytree=0.8
        )
    # Try XGBoost
    if importlib.util.find_spec("xgboost"):
        from xgboost import XGBClassifier
        models["xgb"] = lambda: XGBClassifier(
            n_estimators=400, max_depth=8, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
            tree_method="hist", eval_metric="mlogloss", n_jobs=0
        )
    # Try CatBoost
    if importlib.util.find_spec("catboost"):
        from catboost import CatBoostClassifier
        models["cat"] = lambda: CatBoostClassifier(
            iterations=500, depth=8, learning_rate=0.05, loss_function="MultiClass", verbose=False
        )
    # Always have sklearn fallbacks
    models["sk_hgb"] = lambda: HistGradientBoostingClassifier(max_depth=None, learning_rate=0.1)
    models["sk_rf"]  = lambda: RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=42)
    return models

@dataclass
class TrainConfig:
    train_csv: str
    target_col: str
    feature_pool_json: str
    out_dir: str = "artifacts/specialists"
    map_path: str = "artifacts/specialist_map.json"
    test_size: float = 0.2
    seed: int = 42
    models: List[str] | None = None  # None => usa disponíveis
    max_features_per_set: int | None = None  # opcional: limitar k

def _load_feature_pool(path: str) -> List[Dict[str, Any]]:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    return d["pool"]

def _subset_columns(df: pd.DataFrame, feats: List[str]) -> List[str]:
    return [c for c in feats if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]

def train_specialists(cfg: TrainConfig) -> Dict[str, Any]:
    # 1) Dados
    df = pd.read_csv(cfg.train_csv)
    assert cfg.target_col in df.columns, f"target_col {cfg.target_col} não existe em {cfg.train_csv}"
    y_raw = df[cfg.target_col].values
    # --- blacklist para evitar vazamentos ---
    blacklist = {cfg.target_col, "Label", "label", "attack_cat", "Attack_cat", "id", "ID"}
    X = df.drop(columns=[c for c in blacklist if c in df.columns])

    # filtro numérico simples (mantém alinhado com MI do passo 2a)
    X = X.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="any")
    cols_universe = X.columns.tolist()

    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    classes_str = le.classes_.tolist()

    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=cfg.test_size, random_state=cfg.seed, stratify=y)

    # 2) Pool de feature sets
    pool = _load_feature_pool(cfg.feature_pool_json)
    if cfg.max_features_per_set:
        for p in pool:
            p["features"] = p["features"][: cfg.max_features_per_set]

    # 3) Modelos candidatos
    avail = _available_models()
    if cfg.models:
        # keep only requested that exist
        model_keys = [m for m in cfg.models if m in avail]
    else:
        # prefer boosters se existirem, senão sklearn:
        pref = ["lgbm", "xgb", "cat", "sk_hgb", "sk_rf"]
        model_keys = [m for m in pref if m in avail]

    logger.info(f"Modelos candidatos: {model_keys}")
    logger.info(f"Total de feature sets: {len(pool)}")

    # 4) Busca do especialista por classe
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Any] = {
        "classes": classes_str,
        "models_used": model_keys,
        "specialists": {}  # class_name -> {model_key, feature_set_name, f1, latency_ms, model_path}
    }

    for k_idx, k_name in enumerate(classes_str):
        best: Dict[str, Any] | None = None

        for mkey in model_keys:
            model_fn = avail[mkey]

            for f in pool:
                feats = _subset_columns(X_tr.assign(**X_va), f["features"])  # reinterseção com universo
                if len(feats) == 0:
                    continue

                clf = model_fn()
                # Treina no subconjunto
                clf.fit(X_tr[feats], y_tr)

                # Predição + F1_k
                y_pred = clf.predict(X_va[feats])
                f1_dict = f1_per_class(y_va, y_pred)
                f1_k = float(f1_dict.get(k_idx, 0.0))

                # Latência média (ms/amostra) no conjunto de validação
                t0 = time.perf_counter()
                _ = clf.predict(X_va[feats])
                dt = (time.perf_counter() - t0)
                latency_ms = (dt / max(1, X_va.shape[0])) * 1000.0

                cand = {
                    "model_key": mkey,
                    "feature_set_name": f["name"],
                    "k": int(len(feats)),
                    "f1_k": f1_k,
                    "latency_ms": latency_ms,
                    "features": feats,
                }
                if (best is None) or (f1_k > best["f1_k"]) or (np.isclose(f1_k, best["f1_k"]) and latency_ms < best["latency_ms"]):
                    best = cand

        # salvar o melhor modelo para a classe k
        if best is None:
            logger.warning(f"Nenhum especialista encontrado para classe {k_name}.")
            continue

        # Re-treina no TR completo (opcional: TR+VA) e salva
        final_model = avail[best["model_key"]]()
        feats = best["features"]
        final_model.fit(X[feats], y)  # treina no dataset completo para robustez

        class_name = str(k_name)
        class_dir = out_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        model_path = class_dir / "model.joblib"
        joblib.dump(final_model, model_path)

        best["model_path"] = str(model_path)
        results["specialists"][class_name] = best
        logger.success(f"Classe '{k_name}': {best['model_key']} + {best['feature_set_name']} (F1_k={best['f1_k']:.4f}, {best['k']} feats)")

    # salva o mapa
    map_path = Path(cfg.map_path)
    map_path.parent.mkdir(parents=True, exist_ok=True)
    map_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    logger.success(f"Mapa de especialistas salvo em {map_path}")
    return results
