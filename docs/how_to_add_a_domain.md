# Como Adicionar um Domínio

## Visão Geral

Um domínio representa uma família de agentes com comportamentos similares (ex: `pix`, `seguros`, `renegociacao`). Cada domínio tem seu próprio config, dataset e graders.

## Passo a Passo

### 1. Criar o Domain Config

```bash
mkdir -p domains/<nome_dominio>
```

Criar `domains/<nome_dominio>/domain_config.yaml`:

```yaml
domain:
  name: "pix_whatsapp"
  display_name: "PIX WhatsApp"
  version: "1.0.0"
  owner: "squad-pix"

agent:
  entry_point: "agents.pix.main:run"
  timeout_ms: 30000
  max_turns: 10

graders:
  - type: "state_accuracy"
  - type: "tool_call_match"
  - type: "policy_compliance"
    policies_file: "domains/pix_whatsapp/policies.yaml"

scoring_profile: "functional"

thresholds:
  success_rate: 0.85
  policy_compliance_rate: 0.95
  latency_p95_ms: 5000
```

### 2. Escrever Seed Cases (mínimo 10)

Criar `domains/<nome_dominio>/cases/seed/` com casos YAML manuais. Estes são a base para geração sintética e devem cobrir:

- Happy path principal (3-4 casos)
- Variações de input (2-3 casos)
- Edge cases conhecidos (2-3 casos)
- Ao menos 1 caso adversarial

### 3. Gerar Casos Sintéticos

```bash
bench generate --domain <nome_dominio> --count 50 --seed 42
```

Saída em `domains/<nome_dominio>/cases/synthetic/`. Cada caso gerado inclui `source_type: synthetic_candidate`.

### 4. Validar Casos Gerados

```bash
bench validate --domain <nome_dominio> --split dev
```

Verificações automáticas:
- Schema válido (todos campos obrigatórios presentes)
- Sem PII detectável
- Tools referenciadas existem no agent config
- Expected state fields são válidos

### 5. Executar e Revisar

```bash
bench run-suite --domain <nome_dominio> --split dev --report
```

Revisar o report. Ajustar casos com resultados ambíguos.

### 6. Promover para Gold

Após revisão humana (2 anotadores, concordância >= 0.7):

```bash
bench promote --domain <nome_dominio> --cases <ids> --reviewer "fulano"
```

## Estrutura de Arquivos

```
domains/
└── <nome_dominio>/
    ├── domain_config.yaml
    ├── policies.yaml
    ├── dataset_card.md
    ├── cases/
    │   ├── seed/          # Casos iniciais manuais
    │   ├── synthetic/     # Gerados automaticamente
    │   ├── gold/          # Promovidos e validados
    │   └── adversarial/   # Red team
    └── graders/           # Graders custom (opcional)
        └── custom_policy_grader.py
```

## Campos Obrigatórios em Cada Caso

| Campo | Tipo | Descrição |
|-------|------|-----------|
| `case_id` | string (UUID) | Identificador único |
| `version` | string | Versão do schema (ex: "2.0") |
| `domain` | string | Nome do domínio |
| `split` | string | dev/holdout/smoke/regression/calibration |
| `input` | object | Mensagem do usuário + contexto |
| `expected_outcome` | object | Resultado esperado |
| `grading_strategy` | string | Estratégia de avaliação |
| `metadata` | object | Lineage, timestamps, source_type |

## Checklist Final

- [ ] domain_config.yaml criado e válido
- [ ] Mínimo 10 seed cases escritos
- [ ] `bench validate` passa sem erros
- [ ] Pelo menos 5 casos gold promovidos
- [ ] dataset_card.md preenchido
- [ ] Thresholds definidos no config
- [ ] Smoke suite funciona em < 30s
