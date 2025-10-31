# scripts/make_cic_eval.py
from __future__ import annotations
from pathlib import Path
import pandas as pd

SRC = Path(r"data\train_cic.csv")
DST = Path(r"data\cic_eval.csv")
N   = 100_000  # amostra para avaliação rápida (ajuste se quiser)

def main():
    if not SRC.exists():
        raise FileNotFoundError(f"CSV de origem não encontrado: {SRC}")
    df = pd.read_csv(SRC, low_memory=False)
    if "label" not in df.columns:
        raise RuntimeError("Coluna 'label' não encontrada em data\\train_cic.csv")
    if len(df) > N:
        df = df.sample(n=N, random_state=42)
    DST.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DST, index=False)
    print(f"[OK] Avaliação salva em: {DST} (n={len(df):,})")

if __name__ == "__main__":
    main()
