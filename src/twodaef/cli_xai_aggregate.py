from __future__ import annotations
import argparse
from loguru import logger
from twodaef.reports.aggregate_xai import aggregate_xai


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Agrega explicações SHAP por classe e gera CSV/MD consolidados."
    )
    parser.add_argument("--xai_root", required=True, help="Pasta raiz que contém as pastas por classe (ex.: outputs/xai_unsw ou outputs/xai_cic).")
    parser.add_argument("--out_dir", required=True, help="Diretório de saída para o consolidado (CSV/MD).")
    args = parser.parse_args()

    # Chamada POSICIONAL (evita o erro de kwargs inesperado)
    res = aggregate_xai(args.xai_root, args.out_dir)

    logger.info(f"OK — CSV: {res.get('csv_path')} | MD: {res.get('md_path')}")


if __name__ == "__main__":
    main()
