# Dataset Card: [NOME_DO_DATASET]

## Identificação

| Campo | Valor |
|-------|-------|
| Nome | `[nome_do_dataset]` |
| Versão | `[x.y.z]` |
| Família | `[gold \| synthetic \| adversarial \| mixed]` |
| Domínio | `[nome_dominio]` |
| Owner | `[squad/pessoa responsável]` |
| Contato | `[email ou canal Slack]` |
| Data de criação | `[YYYY-MM-DD]` |
| Última atualização | `[YYYY-MM-DD]` |

## Tamanho e Distribuição

| Split | Quantidade | % |
|-------|-----------|---|
| dev | [N] | [X%] |
| holdout | [N] | [X%] |
| smoke | [N] | [X%] |
| regression | [N] | [X%] |
| calibration | [N] | [X%] |
| **Total** | **[N]** | **100%** |

### Distribuição por source_type

| Tipo | Quantidade |
|------|-----------|
| human_gold | [N] |
| synthetic_shadow | [N] |
| synthetic_candidate | [N] |
| adversarial | [N] |

## Metodologia de Criação

### Casos Gold
[Descrever como foram criados. Ex: "Escritos manualmente por 3 especialistas do squad PIX com base em logs anonimizados de produção."]

### Casos Sintéticos
[Descrever pipeline. Ex: "Gerados via GPT-4o com temperature=0.7, seed=42, a partir dos 20 seed cases como few-shot. Validados por amostragem (30% revisados manualmente)."]

### Casos Adversariais
[Descrever abordagem. Ex: "Criados em sessão de red-teaming focada em prompt injection e boundary values de valores monetários."]

## Cobertura

### Cenários cobertos
- [Cenário 1: ex. "PIX por chave CPF - happy path"]
- [Cenário 2: ex. "PIX com saldo insuficiente"]
- [Cenário 3: ...]

### Cenários NÃO cobertos (known gaps)
- [Gap 1: ex. "PIX agendado para data futura"]
- [Gap 2: ...]

## Limitações Conhecidas

- [Limitação 1: ex. "Casos sintéticos tendem a ser mais 'limpos' que inputs reais"]
- [Limitação 2: ex. "Não cobre variações regionais de linguagem"]
- [Limitação 3: ...]

## Uso Pretendido

- **Usar para**: [ex. "Avaliar agente PIX WhatsApp em CI e nightly evals"]
- **NÃO usar para**: [ex. "Treinar modelos, benchmarking de modelos isolados"]

## Integridade

| Verificação | Valor |
|-------------|-------|
| SHA-256 (arquivo principal) | `[hash]` |
| Inter-annotator agreement (Kappa) | `[valor]` |
| Validação de schema | `[pass/fail + data]` |
| Última execução de PII scan | `[data + resultado]` |

## Histórico de Versões

| Versão | Data | Mudança |
|--------|------|---------|
| [x.y.z] | [YYYY-MM-DD] | [Descrição breve] |
