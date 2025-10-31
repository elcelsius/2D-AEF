# Pull Request Template

## Resumo
_O que mudou e por quê?_

## Issues Relacionadas
Closes #<número> (ou) Relates to #<número>

## Tipo de Mudança
- [ ] Bug fix
- [ ] Feature
- [ ] Breaking change
- [ ] Docs
- [ ] CI/CD

## Como Foi Testado?
- [ ] Execução local (comandos + prints)
- [ ] Métricas re-executadas (`eval-twostage`, `plot-eval`)
- [ ] XAI gerado/validado (`explain-specialist`, `aggregate-xai`)

## Capturas / Artefatos (opcional)
_Anexe imagens (matriz de confusão), CSVs, etc._

## Checklist
- [ ] Código segue o style (tipagem básica + logs claros)
- [ ] `requirements.txt`/`pyproject.toml` atualizados (se necessário)
- [ ] Documentação atualizada (README/ARCHITECTURE/RELATÓRIOS)
- [ ] Testado em Windows (PowerShell) e/ou WSL2
- [ ] Não inclui arquivos grandes/privados (checar `.gitignore`)
