# 2D-AEF — Release Checklist (v0.1.0)

> Objetivo: checklist simples para fechar a **v0.1.0 (MVP)** com UNSW e CIC funcionando ponta‑a‑ponta (ingestão → treino → inferência → avaliação → XAI → docs).  
> Use os _checkboxes_ ✅/⬜ para marcar o que foi feito. Mantenha este arquivo no **raiz do repositório**.

---

## 0) Escopo da v0.1.0 (MVP)
- ✅ **UNSW-NB15**: pipeline completo (feature pool, gatekeeper, especialistas, inferência, avaliação, plots, XAI, relatórios).
- ✅ **CIC-IDS2018 (binário)**: pipeline completo análogo ao UNSW (inclui _label mapping_ automático nos plots).
- ✅ **CLIs estáveis** (entrypoints): `gatekeeper-train`, `train-specialists`, `infer-twostage`, `eval-twostage`, `plot-eval`, `explain-specialist`, `aggregate-xai`.
- [ ] **Docs mínimas**: `README.md`, relatórios em `reports/`, placeholders em `data/`, `artifacts/`, `outputs/`. _[faltam: `data/README.md` e `artifacts/README.md`]_ 
- ✅ **.gitignore** aplicado (sem arquivos grandes/binários acidentais).

---

## 1) Preparação do ambiente
- ✅ Python 3.11 instalado
- ✅ `pip install -r requirements.txt`
- ✅ Projeto instalado (modo dev opcional): `pip install -e .`
- ✅ Execução de sanity check:
  ```powershell
  gatekeeper-train --help
  train-specialists --help
  infer-twostage --help
  eval-twostage --help
  plot-eval --help
  explain-specialist --help
  aggregate-xai --help
  ```

---

## 2) Dados de exemplo / Reprodutibilidade
- [ ] `data/README.md` presente com instruções (UNSW/CIC)
- ✅ Scripts utilitários sob `scripts/` funcionam sem edição manual:
  - ✅ `download_cicids2018.py`
  - ✅ `prep_cic_train.py`
  - ✅ `make_cic_eval.py`
  - ✅ `make_gatekeeper_cols_from_csv.py`
- [ ] **Placeholders** presentes:
  - [ ] `data/README.md`
  - [ ] `artifacts/README.md`
  - ✅ `outputs/README.md`

---

## 3) UNSW-NB15 (pipeline)
- ✅ **Feature pool** gerado: `artifacts/feature_pool_unsw.json`
- ✅ **Especialistas** treinados: `artifacts/specialists/*` + `artifacts/specialist_map.json`
- ✅ **Gatekeeper** treinado: `artifacts/gatekeeper.joblib` + `cols.txt`
- ✅ **Inferência**: `infer-twostage` → `outputs/eval_unsw/preds.csv`
- ✅ **Avaliação**: `eval-twostage` (se aplicável) e/ou `plot-eval`
- ✅ **Plots**: `confusion_matrix.png`, `f1_per_class.png`
- ✅ **XAI (SHAP)** por classe: `outputs/xai_unsw/class_*/`
- ✅ **Agregado XAI**: `outputs/xai_unsw/_consolidado/(csv|md)`
- ✅ **Relatório**: `reports/UNSW/RELATORIO_UNSW.md`

---

## 4) CIC-IDS2018 (pipeline binário)
- ✅ **Download/Prep**: scripts executados com sucesso
- ✅ **Feature pool**: `artifacts/feature_pool_cic.json`
- ✅ **Gatekeeper**: `artifacts/gatekeeper_cic.joblib` + `gatekeeper_cic_cols.txt`
- ✅ **Especialistas**: `artifacts/specialists_cic/*` + `artifacts/specialist_map_cic.json`
- ✅ **Inferência de avaliação**: `outputs/eval_cic/preds.csv`
- ✅ **Avaliação**: `eval-twostage` → métricas em `outputs/eval_cic`
- ✅ **Plots** (com alinhamento automático “0↔Benign/1↔Others”): ok
- ✅ **XAI**: `outputs/xai_cic/class_Benign` e `class_Others`
- ✅ **Agregado XAI**: `outputs/xai_cic/_consolidado/(csv|md)`
- ✅ **Relatório**: `reports/cic/RELATORIO_CIC.md`

---

## 5) Qualidade / Robustez
- ✅ **Windows/Pathlib**: sem _hardcodes_ de separador; uso de `Path()` e `.as_posix()` quando serializar
- ✅ **Tratamento de NaN/inf**: sanitização antes de treinos/inferências
- ✅ **Label space**: coerência entre `label` dos CSVs, Gatekeeper e especialistas (ver `_try_align_spaces` do `plots_eval`)
- ✅ **Latência estável**: medição robusta (múltiplas repetições mínimas) no `two_stage`
- ✅ **Mensagens de log** claras (níveis INFO/SUCCESS/WARN)

---

## 6) Empacotamento / CLI
- ✅ `pyproject.toml` com `project.scripts` configurados
- ✅ Instalação local cria os comandos de console acima
- ✅ `requirements.txt` consistente com `pyproject.toml`
- ✅ Execução dos CLIs sem depender do diretório atual (path‑safe)

---

## 7) Documentação
- ✅ `README.md` atualizado (setup, pipelines UNSW/CIC, exemplos de comando)
- ✅ Relatórios de resultados anexados em `reports/`
- ✅ Changelog iniciado: `CHANGELOG.md`
- ✅ Guia de contribuição: `CONTRIBUTING.md` (opcionalmente mantido)

---

## 8) Higiene de repositório
- ✅ `.gitignore` atualizado (sem `data/`, `artifacts/` binários, `outputs/`, caches)
- [ ] Sem arquivos grandes/binarizados acidentais no versionamento (_revisar se **artifacts/** contém `.joblib` versionados; ideal é não versionar_)
- ✅ Commits limpos: `type(scope): mensagem`
- ✅ Issues abertas mapeadas para próximos passos:
  - ✅ #1 **CIC pipeline** (concluído para v0.1.0)
  - ✅ #2 **Experimento boosters por classe** (próxima versão)
  - ✅ #3 **Hardening Windows/Pathlib** (manter aberto até v0.1.1)

---

## 9) Fechamento da release
- [ ] Criar **tag** semântica `v0.1.0`
  ```powershell
  git pull
  git tag -a v0.1.0 -m "2D-AEF v0.1.0 — MVP (UNSW + CIC binário)"
  git push origin v0.1.0
  ```
- [ ] Criar **GitHub Release** usando a tag, anexando:
  - [ ] Resumo das métricas
  - [ ] Prints dos plots (UNSW/CIC)
  - [ ] Links para relatórios `reports/*`
- [ ] Marcar issues/PRs relevantes como fechados ou vinculados à release

---

## 10) Pós‑release (próximos milestones)
- [ ] v0.1.1: hardening Windows/Pathlib + pequenas correções
- [ ] v0.2.0: experimento LGBM vs XGB vs CAT por classe; logging de hardware/tempo; _seeds_ e tabelas de variação
- [ ] v0.3.0: suporte multi‑classe (CIC por ataque); integração Hydra/MLflow (opcional); empacotamento Docker

---

### Notas rápidas
- Gerou CSVs/PNGs? Não commitar os binários — apenas **relatórios .md** e **artefatos .json** essenciais.
- Para pipelines repetíveis, mantenha comandos usados no final dos relatórios.
- Qualquer ajuste em CLIs → lembrar de atualizar `README.md` + exemplos.
