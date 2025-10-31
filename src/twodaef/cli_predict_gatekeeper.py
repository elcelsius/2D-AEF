import argparse
from pathlib import Path
import pandas as pd
from loguru import logger
from twodaef.utils.io import load_joblib

def main():
    parser = argparse.ArgumentParser(description="Predição com Gatekeeper.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input_csv", type=Path, required=True)
    parser.add_argument("--output_csv", type=Path, default=Path("outputs/preds.csv"))
    args = parser.parse_args()

    model = load_joblib(args.model)
    df = pd.read_csv(args.input_csv)

    y_pred, lat_ms = model.predict(df)
    logger.info(f"Latência média por amostra: {lat_ms:.3f} ms")
    out = df.copy()
    out["gatekeeper_pred"] = y_pred
    out["latency_ms_per_sample"] = lat_ms
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)
    logger.success(f"Predições salvas em {args.output_csv}")

if __name__ == "__main__":
    main()
