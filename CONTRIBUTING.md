# Contribuindo para 2D-AEF

Obrigado por contribuir! Este guia é curto e prático.

## Fluxo de trabalho (Git)

- **branch principal:** `main` (sempre estável).
- **branches de feature/experimento:**
  - `feat/cicids2018-pipeline`
  - `exp/specialists-lightgbm-vs-xgb`
  - `fix/windows-paths`
- Use _feature branches_ e **Pull Requests** (PRs).

## Convenção de commits

Formato (Conventional Commits):
```
<type>(escopo opcional): descrição breve

ex.: feat(xai): adicionar SHAP ao especialista classe 1
```
Tipos comuns: `feat`, `fix`, `docs`, `refactor`, `perf`, `test`, `build`, `chore`.

## Estilo de código

- Python >= 3.11
- `ruff` (lint/format) e `black` (format) recomendados.
- Tipagem sempre que possível.

## Testes rápidos

- Dados pequenos para _smoke test_:
  - `data/raw/unsw/UNSW_NB15_training-set.csv` (amostrar n=5k)
- Rodar:
  ```powershell
  python -m twodaef.cli_make_feature_pool `
    --csv data\raw\unsw\UNSW_NB15_training-set.csv `
    --target_col label `
    --max_features_per_set 20 `
    --total_sets 10 `
    --seed 42 `
    --out_json artifacts\feature_pool_unsw.json

  python -m twodaef.cli_train_specialists `
    --train_csv data\raw\unsw\UNSW_NB15_training-set.csv `
    --target_col label `
    --feature_pool_json artifacts\feature_pool_unsw.json `
    --out_dir artifacts\specialists `
    --map_path artifacts\specialist_map.json `
    --test_size 0.2 `
    --seed 42 `
    --models auto
  ```

## PR Checklist

- [ ] Código passa `ruff`/`black` (se usados).
- [ ] Nenhum arquivo grande em `data/`, `outputs/`, `artifacts/` (exceto `.json` de config/mapas).
- [ ] Atualizou `README.md` se necessário.
- [ ] Explicou resultados/impactos no PR.

## Roadmap curto

- [ ] Pipeline **CIC-IDS2018**
- [ ] Comparação completa com baselines (LCCDE-like, GA-FS-like, LGBM otimizado)
- [ ] Relatórios automáticos (plots + MD)
- [ ] CI básica (lint + unit tests opcionais)
