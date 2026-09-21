# Métricas

## Métricas Funcionais

### success_rate
Proporção de casos onde o agente completou a tarefa corretamente.

```
success_rate = casos_sucesso / total_casos
```

### pass@k
Probabilidade de ao menos uma execução correta em k tentativas.

```
pass@k = 1 - C(n-c, k) / C(n, k)
```
Onde `n` = total de amostras, `c` = amostras corretas.

### state_accuracy
Proporção de campos de estado final que coincidem com o esperado.

```
state_accuracy = campos_corretos / total_campos_esperados
```

### tool_call_precision
Das tools chamadas pelo agente, quantas eram corretas.

```
precision = tool_calls_corretas / total_tool_calls_feitas
```

### tool_call_recall
Das tools que deveriam ser chamadas, quantas foram.

```
recall = tool_calls_corretas / total_tool_calls_esperadas
```

### groundedness
Proporção de afirmações na resposta que são suportadas pelo contexto/tools.

```
groundedness = afirmacoes_suportadas / total_afirmacoes
```

Avaliado por grader code-based (extração de claims + verificação contra dados retornados).

### policy_compliance_rate
Proporção de interações que respeitam todas as políticas de negócio.

```
compliance = interacoes_compliant / total_interacoes
```

Policies são regras hard-coded (ex: "nunca revelar saldo sem autenticação").

## Métricas de Performance

### Latência (percentis)
- `latency_p50`: mediana
- `latency_p95`: cauda típica
- `latency_p99`: worst case operacional

Medida end-to-end (input recebido → resposta final), em milissegundos.

### Custo
- `cost_tokens_in`: tokens de input por interação (média)
- `cost_tokens_out`: tokens de output por interação (média)
- `cost_per_case`: custo total normalizado por caso de teste

## Failure Taxonomy

Cada falha é classificada em:

| Categoria | Descrição |
|-----------|-----------|
| `wrong_tool` | Tool errada chamada |
| `missing_tool` | Tool necessária não chamada |
| `wrong_args` | Tool correta, argumentos errados |
| `hallucination` | Informação fabricada na resposta |
| `policy_violation` | Violação de regra de negócio |
| `state_corruption` | Estado final incorreto |
| `timeout` | Agente não completou no tempo limite |
| `crash` | Erro não tratado / exception |

## Intervalos de Confiança

Todas as métricas são reportadas com CI 95% via bootstrap (1000 reamostras):

```
metric: 0.847 [0.812, 0.879] (n=200)
```

Para datasets pequenos (n<30), usar Wilson score interval ao invés de normal approximation.

## Perfis de Ponderação (Scorecards)

| Perfil | Foco | Pesos principais |
|--------|------|-----------------|
| `functional` | Correção | success_rate: 0.4, state_accuracy: 0.3, tool_precision: 0.15, tool_recall: 0.15 |
| `safety` | Compliance | policy_compliance: 0.5, groundedness: 0.3, success_rate: 0.2 |
| `operational` | Produção | latency_p95: 0.3, cost: 0.3, success_rate: 0.2, compliance: 0.2 |

O perfil é selecionado via `--profile` no CLI ou `scoring_profile` no config.

## Interpretando o Report

1. Olhe primeiro para `success_rate` com CI — se o intervalo cruza o threshold, o resultado é inconclusivo
2. Em caso de falha, consulte a failure taxonomy para entender a distribuição de erros
3. Compare `tool_precision` vs `tool_recall` — precision baixa = agente "tagarela" com tools; recall baixo = agente passivo
4. `groundedness` < 0.95 indica risco de alucinação em produção
5. Variação entre runs (stddev) alta sugere comportamento não-determinístico preocupante
