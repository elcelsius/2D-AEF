import argparse
from pathlib import Path
from loguru import logger

from twodaef.infer.two_stage import TwoStageConfig, TwoStageInferencer

def main():
    ap = argparse.ArgumentParser(description="Inferência 2 estágios (Gatekeeper -> Especialista).")
    ap.add_argument("--gatekeeper_model", type=Path, required=True)
    ap.add_argument("--gatekeeper_features", type=Path, required=True, help="arquivo .txt com uma feature por linha")
    ap.add_argument("--specialist_map", type=Path, required=True)
    ap.add_argument("--input_csv", type=Path, required=True)
    ap.add_argument("--output_csv", type=Path, required=True)
    ap.add_argument("--fill_missing", type=float, default=0.0)
    args = ap.parse_args()

    cfg = TwoStageConfig(
        gatekeeper_model=str(args.gatekeeper_model),
        gatekeeper_features_file=str(args.gatekeeper_features),
        specialist_map_json=str(args.specialist_map),
        input_csv=str(args.input_csv),
        output_csv=str(args.output_csv),
        fill_missing=args.fill_missing
    )
    inf = TwoStageInferencer(cfg)
    gk_ms, s2_ms, tot_ms = inf.predict_csv()
    logger.info(f"OK — {gk_ms:.3f} / {s2_ms:.3f} / {tot_ms:.3f} ms")

if __name__ == "__main__":
    main()
