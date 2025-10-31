import argparse
from twodaef.reports.plots_eval import make_eval_plots

def main():
    ap = argparse.ArgumentParser(description="Gera plots (confusion matrix, F1 por classe) a partir de preds.csv.")
    ap.add_argument("--preds_csv", required=False, help="Caminho para preds.csv. Se ausente, usar --dataset_tag.")
    ap.add_argument("--label_col", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--dataset_tag", required=False, help="Ex.: unsw, cic (ativa fallback automático do preds.csv).")
    args = ap.parse_args()

    res = make_eval_plots(
        preds_csv=args.preds_csv,
        label_col=args.label_col,
        out_dir=args.out_dir,
        dataset_tag=args.dataset_tag,
    )
    print(f"OK — plots em {args.out_dir} | F1-macro={res['f1_macro']:.6f}")

if __name__ == "__main__":
    main()
