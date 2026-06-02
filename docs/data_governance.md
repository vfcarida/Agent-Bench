# Governança de Dados

## Ciclo de Vida dos Dados

```
draft → reviewed → approved → gold
```

| Estado | Significado |
|--------|-------------|
| `draft` | Criado, ainda não revisado. Pode ter erros. |
| `reviewed` | Revisado por pelo menos 1 anotador. Pode ter issues pendentes. |
| `approved` | Aprovado por 2+ anotadores com concordância. Pronto para uso. |
| `gold` | Promovido ao dataset oficial. Imutável após promoção (nova versão se precisar alterar). |

## Tipos de Fonte (source_type)

| Tipo | Descrição |
|------|-----------|
| `human_gold` | Criado e validado integralmente por humanos. Máxima confiança. |
| `synthetic_shadow` | Gerado por LLM, executado em shadow mode contra o agente real. Validado por amostragem. |
| `synthetic_candidate` | Gerado por LLM, aguardando validação humana para promoção. |
| `adversarial` | Casos de ataque: prompt injection, inputs malformados, boundary values. |
| `calibration` | Casos com score conhecido, usados para calibrar graders e judges. |

## Estratégia de Splits

| Split | Uso | % típico |
|-------|-----|----------|
| `dev` | Desenvolvimento e debugging. Vale iterar sobre estes. | 60% |
| `holdout` | Métricas oficiais. NUNCA tunar aqui. | 20% |
| `calibration` | Calibrar graders/judges. Scores pré-definidos. | 5% |
| `regression` | Casos que já falharam antes. Garantir não-regressão. | 10% |
| `smoke` | Subset mínimo para CI rápido (<30s). | 5% |

Splits são atribuídos no campo `split` de cada caso. A atribuição é determinística (hash do case_id).

## Política de Dados Sensíveis

**Proibido incluir dados reais de clientes.** Todo dado deve ser:

- Fictício (CPFs gerados, nomes inventados)
- Anonimizado irreversivelmente se derivado de produção
- Sem PII: sem CPF real, conta real, telefone real, endereço real

Validação automática: o pipeline rejeita casos com padrões de PII real (regex + validação de dígitos verificadores).

## Rastreabilidade (Lineage)

Cada caso deve conter no metadata:

- `created_by`: quem criou (pessoa ou pipeline)
- `created_at`: timestamp ISO-8601
- `source_type`: tipo de fonte (ver tabela acima)
- `generation_config`: se sintético, parâmetros usados (model, temperature, seed)
- `reviewed_by`: lista de revisores
- `promotion_history`: log de transições de estado

## Concordância Inter-Anotadores

- Mínimo 2 anotadores por caso gold
- Concordância mínima esperada: Cohen's Kappa >= 0.7
- Casos com discordância vão para adjudicação (ver `adjudication_guide.md`)
- Métricas de concordância reportadas por domínio no dataset card

## Critérios de Promoção: Synthetic → Gold

Um caso sintético pode ser promovido a gold quando:

1. Revisado por humano especialista no domínio
2. Campos `expected_outcome` e `expected_state_changes` validados manualmente
3. Executado contra o agente com resultado consistente (3+ runs)
4. Sem ambiguidade na avaliação (grader determinístico retorna mesmo score)
5. Aprovado por 2 revisores independentes
6. Documentado no lineage como `promoted_from: synthetic_candidate`
