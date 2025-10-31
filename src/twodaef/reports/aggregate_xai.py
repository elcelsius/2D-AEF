from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple
import json
import pandas as pd
from loguru import logger


def _read_summary_csv(dir_path: Path) -> pd.DataFrame:
    csv_path = dir_path / "summary_mean_abs_shap.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Não encontrado: {csv_path}")
    df = pd.read_csv(csv_path)
    # garante colunas
    assert {"feature", "mean_abs_shap"}.issubset(df.columns), f"Colunas esperadas não encontradas em {csv_path}"
    return df


def aggregate_xai(base_dir: str, out_dir: str, top_k: int = 10) -> Dict[str, str]:
    """
    Lê as pastas class_* dentro de base_dir e agrega:
      - CSV consolidado com top-K por classe
      - Markdown com tabelas por classe
    Retorna dict com paths dos artefatos gerados.
    """
    base = Path(base_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    class_dirs = sorted([p for p in base.glob("class_*") if p.is_dir()])
    if not class_dirs:
        raise RuntimeError(f"Nenhuma pasta class_* encontrada em {base_dir}")

    records: List[Dict[str, object]] = []
    md_parts: List[str] = ["# Relatório XAI Consolidado (SHAP)\n"]

    for cdir in class_dirs:
        class_key = cdir.name.replace("class_", "", 1)
        df = _read_summary_csv(cdir).sort_values("mean_abs_shap", ascending=False).head(top_k)
        df = df.reset_index(drop=True)

        # CSV consolidado (acumula)
        for i, row in df.iterrows():
            records.append({
                "class_key": class_key,
                "rank": i + 1,
                "feature": row["feature"],
                "mean_abs_shap": float(row["mean_abs_shap"]),
            })

        # Tabela em Markdown
        md_parts.append(f"## Classe {class_key}\n")
        md_parts.append("| Rank | Feature | |SHAP| médio |\n|---:|---|---:|\n")
        for i, row in df.iterrows():
            md_parts.append(f"| {i+1} | `{row['feature']}` | {row['mean_abs_shap']:.6f} |\n")
        md_parts.append("\n")

    # Salva CSV consolidado
    df_all = pd.DataFrame.from_records(records)
    csv_out = out / "xai_shap_consolidado.csv"
    df_all.to_csv(csv_out, index=False)

    # Salva Markdown
    md_out = out / "xai_shap_consolidado.md"
    md_out.write_text("".join(md_parts), encoding="utf-8")

    logger.success(f"Consolidado CSV: {csv_out}")
    logger.success(f"Consolidado MD:  {md_out}")
    return {"csv": str(csv_out), "md": str(md_out)}
