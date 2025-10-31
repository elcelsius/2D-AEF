import argparse
from pathlib import Path
from loguru import logger

from twodaef.specialists.train_specialists import TrainConfig, train_specialists

def main():
    ap = argparse.ArgumentParser(description="Treinar Matriz de Especialistas (2D-AEF).")
    ap.add_argument("--train_csv", type=Path, required=True)
    ap.add_argument("--target_col", type=str, required=True)
    ap.add_argument("--feature_pool_json", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, default=Path("artifacts/specialists"))
    ap.add_argument("--map_path", type=Path, default=Path("artifacts/specialist_map.json"))
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--models", type=str, default="auto", help="auto ou lista separada por vírgula (ex: lgbm,xgb,sk_hgb,sk_rf)")
    ap.add_argument("--max_features_per_set", type=int, default=None)
    args = ap.parse_args()

    models_list = None if args.models == "auto" else [m.strip() for m in args.models.split(",") if m.strip()]

    cfg = TrainConfig(
        train_csv=str(args.train_csv),
        target_col=args.target_col,
        feature_pool_json=str(args.feature_pool_json),
        out_dir=str(args.out_dir),
        map_path=str(args.map_path),
        test_size=args.test_size,
        seed=args.seed,
        models=models_list,
        max_features_per_set=args.max_features_per_set
    )
    res = train_specialists(cfg)
    logger.info(f"Resumo: {res.get('specialists', {})}")

if __name__ == "__main__":
    main()
