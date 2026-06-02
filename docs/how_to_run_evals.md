# Como Executar Avaliações

## Execução Local

### Rodar uma suite completa

```bash
bench run-suite --domain pix_whatsapp --split dev
```

Opções úteis:
- `--split dev|holdout|smoke|regression` — qual split usar
- `--profile functional|safety|operational` — perfil de scoring
- `--parallel 4` — execuções paralelas
- `--report` — gera report HTML ao final
- `--seed 42` — seed para reprodutibilidade em casos não-determinísticos

### Rodar um caso específico

```bash
bench run-case --case-id "abc123-def456" --verbose
```

Útil para debugging. Com `--verbose`, mostra cada turn do agente, tools chamadas e scores parciais.

### Rodar múltiplos domínios

```bash
bench run-suite --all-domains --split smoke
```

## Execução em CI

### PR Check (smoke suite)

Roda automaticamente em cada PR. Configuração típica:

```yaml
# No workflow de CI
- name: Eval Smoke
  run: bench run-suite --all-domains --split smoke --gate
  timeout-minutes: 5
```

O flag `--gate` faz o comando retornar exit code 1 se algum threshold for violado.

### Nightly Full Eval

Roda toda noite sobre o split `dev` completo + `holdout`:

```bash
bench run-suite --all-domains --split dev --report --output reports/nightly/
bench run-suite --all-domains --split holdout --report --output reports/holdout/
```

## Gerando Reports

```bash
bench report --input results/latest/ --format html --output reports/
bench report --input results/latest/ --format json  # Para consumo programático
```

O report inclui:
- Scorecard por domínio
- Failure taxonomy breakdown
- Comparação com baseline (se disponível)
- Intervalos de confiança

## Gate Command (CI Thresholds)

```bash
bench gate --results results/latest/ --config domains/pix_whatsapp/domain_config.yaml
```

Lê os thresholds do `domain_config.yaml` e compara com os resultados:
- Exit 0: todos thresholds atendidos
- Exit 1: algum threshold violado (detalhes no stderr)

Exemplo de output em falha:
```
GATE FAILED:
  success_rate: 0.82 < 0.85 (threshold)
  policy_compliance_rate: 0.93 < 0.95 (threshold)
```

## Artefatos de Output

Após execução, a pasta `results/` contém:

```
results/
├── <run_id>/
│   ├── summary.json          # Métricas agregadas
│   ├── cases/                # Resultado individual por caso
│   │   ├── <case_id>.json    # Input, output, scores, traces
│   │   └── ...
│   ├── failures/             # Apenas casos que falharam
│   │   └── <case_id>.json
│   ├── report.html           # Report visual (se --report)
│   └── metadata.json         # Config usado, timestamps, versões
```

### Lendo um resultado individual

Cada `<case_id>.json` contém:

```json
{
  "case_id": "abc123",
  "status": "pass|fail|error",
  "scores": {
    "success": 1.0,
    "state_accuracy": 0.875,
    "tool_precision": 1.0,
    "tool_recall": 0.67
  },
  "failure_category": null,
  "agent_trace": [...],
  "latency_ms": 2340,
  "tokens": {"in": 1200, "out": 450}
}
```

## Dicas

- Use `--split smoke` durante desenvolvimento para feedback rápido
- Rode `--split regression` antes de merge para garantir não-regressão
- Nunca rode `--split holdout` localmente para "ver como está" — reservado para CI oficial
- Use `--verbose` + `--case-id` para debugar falhas específicas
