import argparse
from pathlib import Path
from loguru import logger

from twodaef.reports.plots_eval import make_eval_plots

def main():
    ap = argparse.ArgumentParser(description="Gera plots (confusion matrix, F1 por classe) a partir de preds.csv.")
    ap.add_argument("--preds_csv", type=Path, required=True)
    ap.add_argument("--label_col", type=str, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    args = ap.parse_args()

    res = make_eval_plots(
        preds_csv=str(args.preds_csv),
        label_col=args.label_col,
        out_dir=str(args.out_dir),
        class_labels=None,  # deixe None para inferir de y_true/y_pred (binário UNSW: ['0','1'])
    )
    logger.info(f"OK — plots em {args.out_dir} | F1-macro={res['f1_macro']:.6f}")

if __name__ == "__main__":
    main()
