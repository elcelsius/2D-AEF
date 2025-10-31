# data/

Este diretório **não** é versionado (ver `.gitignore`) para evitar subir datasets grandes/sensíveis.

## Estrutura sugerida
```
data/
├─ raw/           # dumps originais (UNSW, CIC, etc.)
├─ interim/       # arquivos intermediários/limpos
└─ processed/     # amostras finais p/ treino/avaliação
```

## Como popular
- **UNSW**: coloque `UNSW_NB15_training-set.csv` em `data/raw/unsw/` e gere subsets conforme os scripts do projeto.
- **CIC-IDS2018**: use os scripts em `scripts/` para baixar e preparar (`prep_cic_train.py`, `make_cic_eval.py`, etc.).

> Dica: crie `.gitkeep` vazios nas subpastas para manter a estrutura.
