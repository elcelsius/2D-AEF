# data/

Este diretório **não** é versionado (ver `.gitignore`) para evitar subir datasets grandes/sensíveis.
Use-o para armazenar **datasets locais** e **CSVs derivados** durante os experimentos.

## Estrutura sugerida
```
data/
├─ raw/           # dumps originais (UNSW, CIC, etc.)
├─ interim/       # arquivos intermediários/limpos
└─ processed/     # amostras finais p/ treino/avaliação
```

> Dica: crie um `.gitkeep` vazio dentro de cada subpasta para manter a estrutura no repositório sem subir dados.

## Como popular

### UNSW-NB15
- Coloque `UNSW_NB15_training-set.csv` em `data/raw/unsw/`.
- Gere subsets e arquivos de inferência/avaliação conforme os scripts e CLIs do projeto.

### CIC-IDS2018 (binário)
- Use os scripts em `scripts/` para **baixar e preparar** (ex.: `download_cicids2018.py`, `prep_cic_train.py`, `make_cic_eval.py`).

## Exemplos de arquivos usados no projeto
- UNSW: `UNSW_NB15_training-set.csv`, `UNSW_NB15_testing-set.csv`
- CIC:  `cic_train.csv`, `cic_eval.csv`, `cic_infer.csv`

## Importante
- **NÃO versionar** arquivos grandes/privados.
- Consulte `README.md` (raiz) para o passo-a-passo dos pipelines e CLIs.
