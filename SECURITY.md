# Security Policy

Obrigado por ajudar a manter o 2D-AEF seguro.

## Versões suportadas

| Versão      | Suporte de segurança |
|------------:|:---------------------|
| v0.1.x      | ✅ Correções críticas |
| < v0.1.0    | ❌ Sem suporte        |

> Observação: este projeto é acadêmico/experimental. As pipelines (UNSW/CIC) são reprodutíveis, mas **não** devem ser usadas diretamente em produção sem revisão de segurança de dados e infraestrutura.

## Reportar vulnerabilidades

1. **Não** abra issue pública para falhas sensíveis.
2. Envie um email para **elcelsius+security@gmail.com** com:
   - Título/assunto: `[2D-AEF][Security] <resumo do problema>`
   - Descrição, impacto esperado, passos de reprodução.
   - Ambiente (SO, Python, versões das libs) e logs relevantes.
3. Prazo de resposta alvo: **7 dias**.
4. Divulgações públicas combinadas após correção ou em até **90 dias**.

## Escopo

- Scripts e CLIs em `src/twodaef/*`.
- Pipelines e scripts em `scripts/`.
- Exclusões: datasets de terceiros (UNSW/CIC), infra local do usuário, artefatos não versionados.

## Boas práticas recomendadas

- Use ambiente virtual dedicado (`.venv`) e **não** execute como administrador.
- Revise e fixe versões do `requirements.txt` quando for usar em produção.
- Mantenha arquivos de dados fora de diretórios sincronizados/publicados.
- Não faça upload de `*.joblib` com dados sensíveis.
