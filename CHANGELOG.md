# Changelog

Formato baseado em [Keep a Changelog](https://keepachangelog.com/) e **SemVer**.

## [0.1.0] - 2025-10-31
### Added
- MVP do **2D-AEF** com pipeline **UNSW-NB15** (gatekeeper, especialistas, inferência 2 estágios).
- Avaliação (F1-macro, Acc) e plots (matriz de confusão, F1 por classe).
- XAI (SHAP) por especialista + agregação.
- Documentação inicial (`README.md`).

### Notes
- Dados e artefatos pesados são ignorados por `.gitignore`.
- Próxima versão deve incluir pipeline **CIC-IDS2018** e baselines comparativos completos.
