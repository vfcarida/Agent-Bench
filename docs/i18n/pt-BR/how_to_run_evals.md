# Como Executar Avaliações (Arquivo Histórico pt-BR)

> **Nota**: Este documento foi arquivado para preservação histórica. Para a documentação oficial e atualizada da CLI do Agent-Bench, consulte [`docs/how_to_run_evals.md`](../how_to_run_evals.md).

## Execução Local

### Rodar uma suite completa

```bash
bench --config-dir configs run-suite pix_basic_v1 --split dev
```

### Rodar um caso específico

```bash
bench --config-dir configs run-case PIX_001 --system tool_calling_reactive_gpt4 --domain pix_assist
```

### Executar Gate de CI

```bash
bench gate <run_id> --min-global 0.60 --min-functional 0.50 --min-risk 0.70
```

### Gerar Relatórios

```bash
bench generate-report <run_id> --format html
bench generate-report <run_id> --format markdown
```
