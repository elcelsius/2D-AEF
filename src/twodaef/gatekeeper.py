from dataclasses import dataclass
from typing import List, Optional, Tuple
import time
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
from sklearn import tree

@dataclass
class GatekeeperConfig:
    max_depth: int = 6
    min_samples_leaf: int = 10
    random_state: int = 42

class GatekeeperModel:
    def __init__(self, cfg: GatekeeperConfig | None = None):
        self.cfg = cfg or GatekeeperConfig()
        self.model = DecisionTreeClassifier(
            max_depth=self.cfg.max_depth,
            min_samples_leaf=self.cfg.min_samples_leaf,
            random_state=self.cfg.random_state,
        )
        self.feature_names_: Optional[List[str]] = None
        self.classes_: Optional[np.ndarray] = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> dict:
        self.feature_names_ = list(X.columns)
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=self.cfg.random_state
        )
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_val)
        f1 = f1_score(y_val, y_pred, average="macro")
        self.classes_ = self.model.classes_
        return {
            "f1_macro": float(f1),
            "report": classification_report(y_val, y_pred, zero_division=0)
        }

    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, float]:
        # Retorna (predicoes, latencia_ms_por_amostra)
        if self.feature_names_ is None:
            self.feature_names_ = list(X.columns)
        X = X[self.feature_names_]
        t0 = time.perf_counter()
        y_pred = self.model.predict(X)
        lat_ms_total = (time.perf_counter() - t0) * 1000.0
        return y_pred, lat_ms_total / max(len(X), 1)

    def export_rules(self) -> str:
        return tree.export_text(self.model, feature_names=self.feature_names_ or None)
