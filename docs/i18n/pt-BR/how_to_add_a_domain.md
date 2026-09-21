# Como Adicionar um Domínio (Arquivo Histórico pt-BR)

> **Nota**: Este documento foi arquivado para preservação histórica. Para a documentação oficial e atualizada em inglês, consulte [`docs/how_to_add_a_domain.md`](../how_to_add_a_domain.md).

## Visão Geral

Um domínio representa uma família de agentes e fluxos de tarefas com comportamentos e restrições regulatórias similares (ex: `pix_assist`, `investment_advisor`, `sme_business_advisor`, `cyber_sandbox`).

Para adicionar um novo domínio na estrutura oficial:
1. Criar a especificação de sistemas em `configs/domains/<domain_id>.yaml`.
2. Criar os casos canônicos em `datasets/gold/dev/<domain_id>.yaml` no schema `EvalCase v2`.
3. Adicionar o domínio em uma suíte em `configs/suites/<suite_id>.yaml`.
4. Validar com `bench --config-dir configs validate-config`.
