import argparse
from pathlib import Path
import pandas as pd
from loguru import logger
from twodaef.gatekeeper import GatekeeperModel, GatekeeperConfig
from twodaef.utils.io import ensure_dir, save_joblib

def read_feature_list(path: Path) -> list[str]:
    cols = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                cols.append(s)
    return cols

def main():
    parser = argparse.ArgumentParser(description="Treinar Gatekeeper (Decision Tree podada).")
    parser.add_argument("--train_csv", type=Path, required=True)
    parser.add_argument("--target_col", type=str, required=True)
    parser.add_argument("--features", type=Path, required=True, help="Arquivo texto com features (uma por linha).")
    parser.add_argument("--model_out", type=Path, default=Path("artifacts/gatekeeper.joblib"))
    parser.add_argument("--max_depth", type=int, default=6)
    parser.add_argument("--min_samples_leaf", type=int, default=10)
    args = parser.parse_args()

    df = pd.read_csv(args.train_csv)
    feat_list = read_feature_list(args.features)
    missing = [c for c in feat_list if c not in df.columns]
    if missing:
        logger.error(f"Features ausentes no CSV: {missing}")
        raise SystemExit(2)

    X = df[feat_list]
    y = df[args.target_col]

    cfg = GatekeeperConfig(max_depth=args.max_depth, min_samples_leaf=args.min_samples_leaf)
    model = GatekeeperModel(cfg)
    metrics = model.fit(X, y)
    logger.info(f"F1-macro (val): {metrics['f1_macro']:.4f}")
    logger.info("\n" + metrics["report"])

    ensure_dir(args.model_out.parent)
    save_joblib(model, args.model_out)
    logger.success(f"Modelo salvo em {args.model_out}")

if __name__ == "__main__":
    main()
