# src/twodaef/cli_eval_twostage.py
from __future__ import annotations
import argparse

from twodaef.eval.evaluate import run_eval_twostage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gatekeeper_model", required=True, help="(ignorado aqui, apenas coerência com pipeline)")
    ap.add_argument("--gatekeeper_features", required=True, help="(ignorado aqui, apenas coerência com pipeline)")
    ap.add_argument("--specialist_map", required=True, help="Caminho para specialist_map_*.json")
    ap.add_argument("--input_csv", required=True, help="CSV rotulado para avaliação (ex.: data\\cic_eval.csv)")
    ap.add_argument("--label_col", required=True, help="Nome da coluna de rótulo (ex.: label)")
    ap.add_argument("--output_dir", required=True, help="Pasta para artefatos (ex.: outputs\\eval_cic)")
    ap.add_argument("--fill_missing", type=float, default=0.0, help="(não usado aqui)")
    args = ap.parse_args()

    # NOTA: este CLI assume que você já rodou o two_stage para gerar preds.csv em output_dir/preds.csv
    preds_csv = f"{args.output_dir}/preds.csv".replace("\\", "/")

    res = run_eval_twostage(
        preds_csv=preds_csv,
        label_col=args.label_col,
        specialist_map=args.specialist_map,
        out_dir=args.output_dir
    )
    print(f"OK — F1-macro={res['f1_macro']:.6f} | Acc={res['accuracy']:.6f} | out={args.output_dir}")


if __name__ == "__main__":
    main()
