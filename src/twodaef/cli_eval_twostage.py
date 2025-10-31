import argparse
from pathlib import Path
from loguru import logger

from twodaef.eval.evaluate import run_eval_twostage

def main():
    ap = argparse.ArgumentParser(description="Avaliação quantitativa do 2D-AEF (two-stage).")
    ap.add_argument("--gatekeeper_model", type=Path, required=True)
    ap.add_argument("--gatekeeper_features", type=Path, required=True)
    ap.add_argument("--specialist_map", type=Path, required=True)
    ap.add_argument("--input_csv", type=Path, required=True, help="CSV com coluna de rótulo")
    ap.add_argument("--label_col", type=str, required=True, help="Nome da coluna de rótulo (ex.: label)")
    ap.add_argument("--output_dir", type=Path, required=True)
    ap.add_argument("--fill_missing", type=float, default=0.0)
    args = ap.parse_args()

    res = run_eval_twostage(
        gatekeeper_model=str(args.gatekeeper_model),
        gatekeeper_features_file=str(args.gatekeeper_features),
        specialist_map_json=str(args.specialist_map),
        input_csv=str(args.input_csv),
        label_col=args.label_col,
        output_dir=str(args.output_dir),
        fill_missing=args.fill_missing,
    )
    logger.info(f"OK — F1-macro={res['f1_macro']:.6f} | Acc={res['accuracy']:.6f} | out={args.output_dir}")

if __name__ == "__main__":
    main()
