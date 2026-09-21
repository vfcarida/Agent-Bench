# Governança de Dados

## Ciclo de Vida dos Dados

```
draft → reviewed → approved → gold
```

| Estado | Significado |
|--------|-------------|
| `draft` | Criado, ainda não revisado. Pode ter erros. |
| `reviewed` | Revisado por pelo menos 1 especialista/anotador. Pode ter issues pendentes. |
| `approved` | Aprovado após validação de schema, consistência lógica e revisão humana. Pronto para uso. |
| `gold` | Promovido ao dataset oficial canônico. Imutável após promoção (nova versão para alterações). |

## Tipos de Fonte (source_type)

| Tipo | Descrição |
|------|-----------|
| `human_gold` | Criado e validado integralmente por humanos especialistas. Máxima confiança. |
| `synthetic_shadow` | Gerado por LLM, executado em shadow mode contra o agente real. Validado por amostragem. |
| `synthetic_candidate` | Gerado por LLM, aguardando validação humana para promoção. |
| `adversarial` | Casos de ataque: prompt injection, inputs malformados, bypass de limites, engenharia social. |
| `calibration` | Casos com score conhecido, usados para calibrar graders determinísticos e LLM judges. |

## Estrutura Autoritativa de Datasets

O repositório consolida seus datasets na seguinte hierarquia autoritativa:

1. **`datasets/gold/<split>/<domain>.yaml` (Canônico Autoritativo)**:
   - Estrutura oficial padrão para benchmarking corporativo.
   - Contém casos no schema v2 (`EvalCase v2`) com particionamento estrito por split.
2. **`datasets/synthetic/` e `datasets/adversarial/`**:
   - Famílias sintéticas (`synthetic/shadow`, `synthetic/candidates`) e ataques de segurança.
3. **`data/fixtures/<domain>.yaml` (Legado Dev-Only)**:
   - Mantido exclusivamente para retrocompatibilidade em testes unitários legados de desenvolvimento.
   - **Regra inviolável**: é estritamente proibido carregar casos de `holdout` a partir de `data/fixtures`.
4. **`configs/domains/<domain>.yaml` (Configurações de Sistemas)**:
   - Define a arquitetura, modelos e ferramentas dos agentes sob teste (não contém casos de teste).

## Estratégia de Splits

| Split | Uso | % típico |
|-------|-----|----------|
| `dev` | Desenvolvimento, tuning e debugging diário. | 60% |
| `holdout` | Métricas oficiais finais. NUNCA tunar ou inspecionar aqui. | 20% |
| `calibration` | Calibração de graders/judges com labels conhecidos. | 5% |
| `regression` | Casos de falha histórica para prevenção de regressões. | 10% |
| `smoke` | Subset mínimo para CI rápido e conectividade (<30s). | 5% |

Splits são declarados no campo `split` de cada caso e organizados em diretórios físicos dedicados (`datasets/gold/<split>/`).

## Gate de Contaminação e Leakage de Holdout

Para garantir integridade metodológica e impedir vazamento de dados de teste (`holdout`) para conjuntos de desenvolvimento ou dados sintéticos de treinamento:

- **Check Verbatim**: Compara o hash criptográfico SHA-256 de `prompt | initial_state` e igualdade de prompts normalizados.
- **Check de Paráfrase**: Calcula a similaridade de Jaccard tokenizada sobre os textos de prompt, com threshold padrão de 0.80 e poda de comprimento.
- **Execução em CI**: `python scripts/check_contamination.py` e o comando CLI `bench check-contamination` rodam obrigatoriamente no pipeline de CI offline. Qualquer vazamento detectado falha o build imediatamente.

## Política de Dados Sensíveis

**Proibido incluir dados reais de clientes.** Todo dado deve ser:

- Fictício (CPFs gerados com dígitos válidos, nomes inventados)
- Anonimizado irreversivelmente se derivado de produção
- Sem PII: sem CPF real, conta bancária real, telefone real, endereço real

Validação automática: o pipeline de validação rejeita casos com padrões de PII real.

## Rastreabilidade (Lineage)

Cada caso contém no `metadata`:

- `created_by`: criador (autor humano, migração ou pipeline sintético)
- `reviewer`: revisor responsável pela aprovação
- `review_status`: estado de revisão (`draft`, `approved`, etc.)
- `generation_recipe_id`: identificador de receita de geração quando aplicável

## Concordância Inter-Anotadores e Validação

- **Status de Coleta Multi-Anotador**: No presente estágio do benchmark, os casos passam por validação automatizada de schema (`validate_eval_case`), verificação de consistência lógica de ferramentas/estados (`consistency_checker`), checagem de duplicatas (`dedup_checker`) e revisão individual por especialista.
- **Métricas Formais (Cohen's Kappa / Krippendorff's Alpha)**: O cálculo formal de concordância estatística inter-anotadores (Cohen's Kappa) está previsto para inclusão quando datasets anotados por múltiplos juízes independentes forem introduzidos. Alegações numéricas a priori (como $\kappa \ge 0.7$) foram removidas para preservar o rigor factual da documentação até que anotações redundantes estejam publicadas.

## Critérios de Promoção: Synthetic → Gold

Um caso sintético pode ser promovido a gold quando:

1. Revisado por humano especialista no domínio
2. Campos `expected_outcome` e `expected_state_changes` validados manualmente
3. Executado contra o agente com resultado consistente (3+ runs)
4. Sem ambiguidade na avaliação (grader determinístico retorna mesmo score)
5. Aprovado pelo time de segurança/revisão
6. Documentado no lineage como `promoted_from: synthetic_candidate`
