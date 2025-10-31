# scripts/download_cicids2018.py
import os
from kaggle.api.kaggle_api_extended import KaggleApi

DEST = r"data\raw\cicids2018"

def main():
    os.makedirs(DEST, exist_ok=True)
    api = KaggleApi()
    api.authenticate()  # usa %USERPROFILE%\.kaggle\kaggle.json
    print("Autenticação OK. Baixando dataset…")
    api.dataset_download_files(
        dataset="solarmainframe/ids-intrusion-csv",
        path=DEST,
        unzip=True
    )
    print(f"Concluído. Arquivos em: {DEST}")

if __name__ == "__main__":
    main()
