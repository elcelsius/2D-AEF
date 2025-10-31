from __future__ import annotations
import argparse
from pathlib import Path
from loguru import logger

from twodaef.reports.plots_eval import make_eval_plots


def main() -> None:
    parser = argparse.ArgumentParser(description="Gera plots (confusion matrix, F1 por classe) a partir de preds.csv.")
    parser.add_argument("--preds_csv", required=True, help="Caminho para o preds.csv")
    parser.add_argument("--label_col", required=True, help="Nome da coluna de rótulo (ex.: label)")
    parser.add_argument("--out_dir", required=True, help="Diretório de saída para os plots")
    parser.add_argument(
        "--dataset_tag",
        required=False,
        default=None,
        help="Tag do dataset para nomear arquivos e habilitar fallback automático (ex.: unsw, cic)",
    )
    args = parser.parse_args()

    preds_csv_p = Path(args.preds_csv)

    # Fallback automático: se o caminho informado não existir, tenta outputs/eval_<tag>/preds.csv
    if not preds_csv_p.exists():
        if args.dataset_tag:
            fallback = Path("outputs") / f"eval_{args.dataset_tag}" / "preds.csv"
            if fallback.exists():
                logger.warning(
                    f"Arquivo não encontrado em {preds_csv_p}. Usando fallback automático: {fallback}"
                )
                preds_csv_p = fallback
            else:
                raise FileNotFoundError(
                    f"Nem {preds_csv_p} nem fallback {fallback} existem. "
                    f"Execute a avaliação para gerar o preds.csv primeiro."
                )
        else:
            raise FileNotFoundError(
                f"{preds_csv_p} não existe e nenhum --dataset_tag foi fornecido para fallback. "
                f"Informe um caminho válido ou passe --dataset_tag (ex.: unsw, cic)."
            )

    res = make_eval_plots(
        preds_csv=str(preds_csv_p),
        label_col=args.label_col,
        out_dir=args.out_dir,
        class_labels=None,
        dataset_tag=args.dataset_tag,  # garante nomes como *_unsw.png ou *_cic.png
    )
    logger.info(f"OK — plots em {args.out_dir} | F1-macro={res['f1_macro']:.6f}")


if __name__ == "__main__":
    main()
