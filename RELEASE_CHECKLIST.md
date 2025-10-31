# 2D-AEF — Release Checklist (v0.1.0)

> Objetivo: checklist simples para fechar a **v0.1.0 (MVP)** com UNSW e CIC funcionando ponta‑a‑ponta (ingestão → treino → inferência → avaliação → XAI → docs).  
> Use os _checkboxes_ ✅/⬜ para marcar o que foi feito. Mantenha este arquivo no **raiz do repositório**.

---

## 0) Escopo da v0.1.0 (MVP)
- [x] **UNSW-NB15**: pipeline completo (feature pool, gatekeeper, especialistas, inferência, avaliação, plots, XAI, relatórios).
- [x] **CIC-IDS2018 (binário)**: pipeline completo análogo ao UNSW (inclui _label mapping_ automático nos plots).
- [x] **CLIs estáveis** (entrypoints): `gatekeeper-train`, `train-specialists`, `infer-twostage`, `eval-twostage`, `plot-eval`, `explain-specialist`, `aggregate-xai`.
- [ ] **Docs mínimas**: `README.md`, relatórios em `reports/`, placeholders em `data/`, `artifacts/`, `outputs/`. _[faltam: `data/README.md` e `artifacts/README.md`]_ 
- [x] **.gitignore** aplicado (sem arquivos grandes/binários acidentais).

---

## 1) Preparação do ambiente
- [x] Python 3.11 instalado
- [x] `pip install -r requirements.txt`
- [x] Projeto instalado (modo dev opcional): `pip install -e .`
- [x] Execução de sanity check:
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
- [x] Scripts utilitários sob `scripts/` funcionam sem edição manual:
  - [x] `download_cicids2018.py`
  - [x] `prep_cic_train.py`
  - [x] `make_cic_eval.py`
  - [x] `make_gatekeeper_cols_from_csv.py`
- [ ] **Placeholders** presentes:
  - [ ] `data/README.md`
  - [ ] `artifacts/README.md`
  - [x] `outputs/README.md`

---

## 3) UNSW-NB15 (pipeline)
- [x] **Feature pool** gerado: `artifacts/feature_pool_unsw.json`
- [x] **Especialistas** treinados: `artifacts/specialists/*` + `artifacts/specialist_map.json`
- [x] **Gatekeeper** treinado: `artifacts/gatekeeper.joblib` + `cols.txt`
- [x] **Inferência**: `infer-twostage` → `outputs/eval_unsw/preds.csv`
- [x] **Avaliação**: `eval-twostage` (se aplicável) e/ou `plot-eval`
- [x] **Plots**: `confusion_matrix.png`, `f1_per_class.png`
- [x] **XAI (SHAP)** por classe: `outputs/xai_unsw/class_*/`
- [x] **Agregado XAI**: `outputs/xai_unsw/_consolidado/(csv|md)`
- [x] **Relatório**: `reports/UNSW/RELATORIO_UNSW.md`

---

## 4) CIC-IDS2018 (pipeline binário)
- [x] **Download/Prep**: scripts executados com sucesso
- [x] **Feature pool**: `artifacts/feature_pool_cic.json`
- [x] **Gatekeeper**: `artifacts/gatekeeper_cic.joblib` + `gatekeeper_cic_cols.txt`
- [x] **Especialistas**: `artifacts/specialists_cic/*` + `artifacts/specialist_map_cic.json`
- [x] **Inferência de avaliação**: `outputs/eval_cic/preds.csv`
- [x] **Avaliação**: `eval-twostage` → métricas em `outputs/eval_cic`
- [x] **Plots** (com alinhamento automático “0↔Benign/1↔Others”): ok
- [x] **XAI**: `outputs/xai_cic/class_Benign` e `class_Others`
- [x] **Agregado XAI**: `outputs/xai_cic/_consolidado/(csv|md)`
- [x] **Relatório**: `reports/cic/RELATORIO_CIC.md`

---

## 5) Qualidade / Robustez
- [x] **Windows/Pathlib**: sem _hardcodes_ de separador; uso de `Path()` e `.as_posix()` quando serializar
- [x] **Tratamento de NaN/inf**: sanitização antes de treinos/inferências
- [x] **Label space**: coerência entre `label` dos CSVs, Gatekeeper e especialistas (ver `_try_align_spaces` do `plots_eval`)
- [x] **Latência estável**: medição robusta (múltiplas repetições mínimas) no `two_stage`
- [x] **Mensagens de log** claras (níveis INFO/SUCCESS/WARN)

---

## 6) Empacotamento / CLI
- [x] `pyproject.toml` com `project.scripts` configurados
- [x] Instalação local cria os comandos de console acima
- [x] `requirements.txt` consistente com `pyproject.toml`
- [x] Execução dos CLIs sem depender do diretório atual (path‑safe)

---

## 7) Documentação
- [x] `README.md` atualizado (setup, pipelines UNSW/CIC, exemplos de comando)
- [x] Relatórios de resultados anexados em `reports/`
- [x] Changelog iniciado: `CHANGELOG.md`
- [x] Guia de contribuição: `CONTRIBUTING.md` (opcionalmente mantido)

---

## 8) Higiene de repositório
- [x] `.gitignore` atualizado (sem `data/`, `artifacts/` binários, `outputs/`, caches)
- [ ] Sem arquivos grandes/binarizados acidentais no versionamento (_revisar se **artifacts/** contém `.joblib` versionados; ideal é não versionar_)
- [x] Commits limpos: `type(scope): mensagem`
- [x] Issues abertas mapeadas para próximos passos:
  - [x] #1 **CIC pipeline** (concluído para v0.1.0)
  - [x] #2 **Experimento boosters por classe** (próxima versão)
  - [x] #3 **Hardening Windows/Pathlib** (manter aberto até v0.1.1)

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
