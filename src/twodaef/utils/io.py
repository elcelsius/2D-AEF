from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import json
import joblib
import numpy as np
import pandas as pd


# =========================
# Diretórios
# =========================
def ensure_dir(path: str | Path) -> Path:
    """
    Garante que o diretório exista e retorna um Path.
    Aceita tanto um diretório quanto um arquivo (neste caso usa o parent).
    """
    p = Path(path)
    # Se tiver sufixo, tratamos como arquivo -> criamos o parent
    if p.suffix:
        p.parent.mkdir(parents=True, exist_ok=True)
        return p.parent
    p.mkdir(parents=True, exist_ok=True)
    return p


# =========================
# Modelos (joblib)
# =========================
def save_joblib(obj: Any, path: str | Path) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    joblib.dump(obj, p)


def load_joblib(path: str | Path) -> Any:
    return joblib.load(Path(path))


# =========================
# CSV (utf-8)
# =========================
def read_csv_utf8(path: str | Path) -> pd.DataFrame:
    """
    Leitura robusta de CSV em UTF-8, com low_memory desativado.
    """
    return pd.read_csv(Path(path), encoding="utf-8", low_memory=False)


def write_csv_utf8(df: pd.DataFrame, path: str | Path) -> None:
    """
    Escrita de CSV em UTF-8 garantindo a existência do diretório.
    """
    p = Path(path)
    ensure_dir(p.parent)
    df.to_csv(p, index=False, encoding="utf-8")


# =========================
# JSON (utf-8)
# =========================
def read_json_utf8(path: str | Path) -> Dict[str, Any]:
    """
    Lê JSON como dict (UTF-8).
    """
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8"))


def write_json_utf8(obj: Dict[str, Any], path: str | Path) -> None:
    """
    Escreve JSON (UTF-8) com indentação e ensure_ascii=False.
    """
    p = Path(path)
    ensure_dir(p.parent)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


# =========================
# Sanitização tabular
# =========================
def sanitize_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Substitui ±inf por NaN e, em seguida, preenche NaN com 0.
    Não força tipos; quem chama decide casts depois.
    """
    clean = df.replace([np.inf, -np.inf], np.nan)
    return clean.fillna(0)
