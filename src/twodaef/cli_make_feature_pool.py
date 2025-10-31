import argparse
from pathlib import Path
import json
import pandas as pd
from loguru import logger

from twodaef.features.pools import PoolConfig, build_feature_pool

def main():
    ap = argparse.ArgumentParser(description="Gerar pool de feature sets (PSO/GWO/FFA-like).")
    ap.add_argument("--csv", type=Path, required=True, help="CSV de amostra (ex.: um chunk de CIC-IDS-2018).")
    ap.add_argument("--target_col", type=str, required=True, help="Nome da coluna alvo.")
    ap.add_argument("--max_features_per_set", type=int, default=20)
    ap.add_argument("--total_sets", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--feature_costs", type=Path, default=None, help="(Opcional) arquivo feature,cost")
    ap.add_argument("--out_json", type=Path, default=Path("artifacts/feature_pool.json"))
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if args.target_col not in df.columns:
        logger.error(f"Coluna alvo '{args.target_col}' não encontrada no CSV.")
        raise SystemExit(2)

    cfg = PoolConfig(
        target_col=args.target_col,
        max_features_per_set=args.max_features_per_set,
        total_sets=args.total_sets,
        seed=args.seed,
        feature_costs_path=str(args.feature_costs) if args.feature_costs else None
    )
    result = build_feature_pool(df, cfg)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    logger.success(f"Pool gerado: {args.out_json}")
    logger.info(f"Top-10 MI: {result['mi_top10']}")

if __name__ == "__main__":
    main()
