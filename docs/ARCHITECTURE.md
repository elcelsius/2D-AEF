# 2D-AEF — Arquitetura do Projeto
**Versão:** v0.1.0

Repositório: [elcelsius/2D-AEF](https://github.com/elcelsius/2D-AEF)

---

## Visão Geral

O 2D-AEF organiza a detecção de intrusão em **dois estágios**:

- **Gatekeeper (filtro rápido):** classifica grosso modo o tráfego e direciona;
- **Especialista por classe:** modelo + conjunto de atributos otimizados para cada classe.

---

## Diagrama (alto nível)

> Dica: no GitHub, mantenha as crases do bloco *Mermaid* exatamente como abaixo.

```mermaid
flowchart LR
    A["Dados brutos<br/>(UNSW, CIC, ...)"] --> B["Ingestão / ETL<br/>(scripts/*)"]
    B --> C["Feature Pool<br/>(artifacts/feature_pool_*.json)"]
    B --> D["Treino Gatekeeper<br/>(artifacts/gatekeeper_*.joblib)"]
    C --> E["Treino Especialistas por Classe<br/>(artifacts/specialists_*/*/model.joblib)"]
    D --> F["Gatekeeper (.joblib)"]
    E --> G["Mapa de Especialistas<br/>(artifacts/specialist_map_*.json)"]
    I["CSV p/ inferência<br/>(data/*_infer.csv ou *_eval.csv)"] --> J["Inferência 2 estágios<br/>(infer-twostage)"]
    F --> J
    G --> J
    J --> K["Predições (preds.csv)"]
    K --> M["Plots / Relatórios<br/>(plot-eval) → confusion_matrix.png, f1_per_class.png"]
    J --> L["XAI (SHAP / LIME)"]
    L --> N["Consolidação XAI<br/>(aggregate-xai) → _consolidado/"]
```

---

## Componentes

- **ETL / Scripts** (`scripts/`): preparo dos CSVs de treino/avaliação e listas de colunas do Gatekeeper.
- **Gatekeeper** (`artifacts/gatekeeper_*.joblib`): classificador rápido para roteamento.
- **Especialistas** (`artifacts/specialists_*/*/model.joblib`): um modelo otimizado por classe.
- **Mapa de Especialistas** (`artifacts/specialist_map_*.json`): liga classe → (modelo, features).
- **Inferência 2 estágios**: combina Gatekeeper + Especialista para gerar `preds.csv`.
- **Relatórios/Plots** (`reports/` e `outputs/`): gráficos, métricas e XAI consolidados.

---

## CLI (pontos de entrada)

```mermaid
flowchart TB
    subgraph Treino
      T1["gatekeeper-train"] --> T2["train-specialists"]
    end
    subgraph Inferência & Avaliação
      I1["infer-twostage"] --> I2["eval-twostage"]
      I2 --> I3["plot-eval"]
      I2 --> I4["explain-specialist"]
      I4 --> I5["aggregate-xai"]
    end
```

---

## Layout de Pastas (essencial)

```
2D-AEF/
├─ scripts/                         # ETL/auxiliares
├─ src/twodaef/                     # código-fonte (CLI + libs)
│  ├─ eval/ reports/ xai/ ...
├─ artifacts/                       # modelos & mapas (pesados ignorados no git)
│  ├─ gatekeeper_*.joblib
│  ├─ specialists_*/*/model.joblib
│  └─ specialist_map_*.json
├─ data/                            # dados brutos/derivados (ignorado no git)
├─ outputs/                         # resultados (plots, preds, xai) (ignorado no git)
├─ reports/                         # relatórios em Markdown/figuras
├─ README.md  pyproject.toml  requirements.txt  .gitignore
```

---

## Notas de Renderização (Mermaid no GitHub)

- Use bloco cercado com **três crases** e a linguagem `mermaid` na primeira linha.
- Não cole linhas de Mermaid fora do bloco de código.
- Evite caracteres especiais fora de aspas nos rótulos; quando necessário, use `"` e quebras de linha com `<br/>`.
- Se aparecer *Unable to render rich display*, verifique se **toda** a seção do diagrama está dentro do bloco cercado.
