# Guia de Anotação

## O Que Faz um Bom Caso de Eval

Um caso de eval deve ser:

1. **Não-ambíguo** — apenas uma resposta correta possível (ou critérios claros para aceitar variações)
2. **Atômico** — testa um comportamento específico, não múltiplos de uma vez
3. **Realista** — representa algo que um usuário real faria
4. **Verificável** — possível determinar pass/fail programaticamente
5. **Independente** — não depende do resultado de outros casos

## Escrevendo expected_outcome

O `expected_outcome` define o que o agente DEVE produzir. Formatos aceitos:

### Exact match (para respostas estruturadas)
```yaml
expected_outcome:
  type: "exact"
  value:
    action: "transfer_pix"
    status: "completed"
```

### Contains (para respostas textuais)
```yaml
expected_outcome:
  type: "contains"
  must_include: ["transferência realizada", "R$ 50,00"]
  must_not_include: ["erro", "não foi possível"]
```

### Semantic (quando variação textual é aceitável)
```yaml
expected_outcome:
  type: "semantic"
  reference: "A transferência PIX de R$ 50 foi realizada com sucesso para João."
  min_similarity: 0.85
```

## Escrevendo expected_state_changes

Define mutações esperadas no estado do agente/sistema:

```yaml
expected_state_changes:
  - field: "session.balance"
    operation: "decrease_by"
    value: 50.00
  - field: "session.last_transaction.type"
    operation: "equals"
    value: "pix_out"
  - field: "session.pending_auth"
    operation: "equals"
    value: false
```

Operações suportadas: `equals`, `increase_by`, `decrease_by`, `contains`, `not_null`, `is_null`, `changed`.

## Escolhendo grading_strategy

| Estratégia | Quando usar |
|------------|-------------|
| `deterministic` | Output estruturado, resposta única correta. Preferido. |
| `tool_match` | Foco em quais tools foram chamadas e com quais args. |
| `state_check` | Foco no estado final, não na resposta textual. |
| `rubric` | Avaliação multidimensional com rubric explícita. Mais caro. |
| `composite` | Combinação de múltiplas estratégias com pesos. |

**Regra geral**: se pode ser `deterministic`, use `deterministic`. Só escale para `rubric` quando necessário.

## Escrevendo Rubrics

Para `grading_strategy: rubric`, defina critérios explícitos:

```yaml
rubric:
  - dimension: "completude"
    weight: 0.4
    levels:
      0: "Resposta não endereça a pergunta"
      1: "Resposta parcial, falta informação chave"
      2: "Resposta completa com toda informação necessária"
  - dimension: "tom"
    weight: 0.2
    levels:
      0: "Tom inadequado (rude, informal demais)"
      1: "Tom neutro mas não ideal"
      2: "Tom adequado ao contexto bancário"
```

## Armadilhas Comuns

| Armadilha | Problema | Solução |
|-----------|----------|---------|
| Expected muito rígido | Falha por variação irrelevante (pontuação, espaço) | Use `contains` ou normalize antes de comparar |
| Caso dependente de hora/data | Falha em horários diferentes | Mock de datetime no contexto |
| Múltiplos caminhos corretos | Falso negativo | Liste alternativas ou use `semantic` |
| Estado inicial implícito | Não reproduzível | Sempre declarar `initial_state` completo |
| Tool args com valores dinâmicos | IDs mudam entre runs | Use matchers (`any_uuid`, `any_timestamp`) |

## Checklist de Qualidade

Antes de submeter um caso, verifique:

- [ ] `case_id` é UUID único
- [ ] `input.user_message` é realista (linguagem natural, não robotizada)
- [ ] `initial_state` está completo (nada implícito)
- [ ] `expected_outcome` é verificável por código
- [ ] `expected_state_changes` lista TODAS as mudanças esperadas
- [ ] `grading_strategy` é a mais simples que funciona
- [ ] Sem PII real (CPFs, contas, telefones)
- [ ] Metadata preenchido (source_type, created_by, created_at)
- [ ] Rodou o caso localmente e o grader retorna o score esperado
