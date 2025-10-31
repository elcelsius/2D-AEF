# scripts/make_gatekeeper_cols_from_csv.py
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np

CSV = Path(r"data\train_cic.csv")
OUT = Path(r"gatekeeper_cic_cols.txt")
MAX_COLS = 12  # escolha leve para latência

def main():
    if not CSV.exists():
        raise FileNotFoundError(f"CSV não encontrado: {CSV}")
    # lê só o cabeçalho + primeira parte para inferir dtypes
    df = pd.read_csv(CSV, nrows=2000, low_memory=False)
    if "label" in df.columns:
        df = df.drop(columns=["label"])
    nums = df.select_dtypes(include=[np.number]).columns.tolist()
    if not nums:
        raise RuntimeError("Nenhuma coluna numérica encontrada no CSV.")
    cols = nums[:MAX_COLS]
    OUT.write_text("\n".join(cols), encoding="utf-8")
    print(f"[OK] {len(cols)} colunas salvas em {OUT}")
    for c in cols:
        print(" -", c)

if __name__ == "__main__":
    main()
