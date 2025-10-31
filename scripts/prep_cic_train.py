# scripts/prep_cic_train.py
from __future__ import annotations
import re
from pathlib import Path
from typing import Iterator, List
import pandas as pd
import numpy as np

RAW_DIR = Path(r"data\raw\cicids2018")
OUT_TRAIN = Path(r"data\train_cic.csv")
OUT_INFER = Path(r"data\cic_infer.csv")

# Limites para não estourar memória/tempo (ajuste se quiser)
CHUNKSIZE = 200_000
MAX_ROWS  = 300_000   # total alvo no train_cic
INFER_ROWS = 1_000    # amostra para inferência

def snake(s: str) -> str:
    s = s.strip()
    s = re.sub(r"[^0-9A-Za-z]+", "_", s)  # troca espaços e pontuação por _
    s = re.sub(r"_+", "_", s)
    return s.strip("_").lower()

def map_label(raw: str) -> str:
    """
    Converte rótulos do CIC-IDS2018 (ex.: 'Web Attack - XSS', 'DDoS attacks-LOIC-HTTP', 'BENIGN')
    em classes agregadas.
    """
    if raw is None:
        return "Others"
    s = str(raw).strip().lower()

    # benign
    if "benign" in s or s in {"normal"}:
        return "Benign"

    # ddos / dos (ordem importa: checar ddos antes)
    if "ddos" in s:
        return "DDoS"
    if "dos" in s:
        return "DoS"

    # web attacks
    if "web" in s or "sql" in s or "xss" in s or "brute force" in s and "web" in s:
        return "Web"

    # brute force (ftp/ssh patator)
    if "patator" in s or "brute force" in s and "web" not in s:
        return "BruteForce"

    # botnet
    if "bot" in s:
        return "Bot"

    # infiltration
    if "infiltration" in s:
        return "Infiltration"

    # portscan
    if "portscan" in s or "port scan" in s:
        return "PortScan"

    # heartbleed
    if "heartbleed" in s:
        return "Heartbleed"

    return "Others"

def iter_csv_files(dirpath: Path) -> Iterator[Path]:
    for p in sorted(dirpath.glob("*.csv")):
        yield p

def load_build_train() -> None:
    OUT_TRAIN.parent.mkdir(parents=True, exist_ok=True)
    OUT_INFER.parent.mkdir(parents=True, exist_ok=True)

    total_rows = 0
    collected: List[pd.DataFrame] = []

    files = list(iter_csv_files(RAW_DIR))
    if not files:
        raise FileNotFoundError(f"Nenhum CSV encontrado em {RAW_DIR}")

    for csv_path in files:
        # Leitura em chunks para não estourar RAM
        try:
            for chunk in pd.read_csv(
                csv_path,
                low_memory=False,
                chunksize=CHUNKSIZE,
                on_bad_lines="skip"
            ):
                # Normaliza nomes
                chunk.columns = [snake(c) for c in chunk.columns]

                # Campo "label" pode variar em nome; procure por uma coluna candidata
                label_col = None
                for cand in ("label", "labels", "attack_category", "attack_cat"):
                    if cand in chunk.columns:
                        label_col = cand
                        break
                if label_col is None:
                    # não há rótulo -> tenta deduzir por heurística (pouco comum nos CSVs do CIC)
                    continue

                # Mapeia rótulos
                chunk["label"] = chunk[label_col].apply(map_label)

                # Mantém só numéricas + label
                num = chunk.select_dtypes(include=[np.number])
                if num.empty:
                    continue
                df_keep = num.copy()
                df_keep["label"] = chunk["label"]

                collected.append(df_keep)
                total_rows += len(df_keep)
                if total_rows >= MAX_ROWS:
                    break
        except Exception as e:
            # Se algum CSV quebrar, apenas loga e segue com os outros
            print(f"[WARN] Falha lendo {csv_path.name}: {e}")

        if total_rows >= MAX_ROWS:
            break

    if not collected:
        raise RuntimeError("Não foi possível montar nenhum chunk válido para o train.")

    df = pd.concat(collected, axis=0, ignore_index=True)

    # Remove colunas NaN/Inf e mantém só numéricas + label
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(axis=1, how="any")

    # Garante que rótulo seja string
    df["label"] = df["label"].astype(str)

    # Salva train
    df.to_csv(OUT_TRAIN, index=False)
    print(f"[OK] Train salvo em: {OUT_TRAIN}  (n={len(df):,})")

    # Infer (pequena amostra)
    infer = df.sample(n=min(INFER_ROWS, len(df)), random_state=42).drop(columns=["label"])
    infer.to_csv(OUT_INFER, index=False)
    print(f"[OK] Infer salvo em: {OUT_INFER}  (n={len(infer):,})")

if __name__ == "__main__":
    load_build_train()
