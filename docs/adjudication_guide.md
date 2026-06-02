# Guia de Adjudicação

## Quando Adjudicar

A adjudicação é necessária quando:

- 2 anotadores discordam sobre o `expected_outcome` de um caso
- Discordância sobre se um comportamento do agente é "correto" ou "aceitável"
- Ambiguidade na política de negócio aplicável
- Novo edge case não coberto pelas diretrizes existentes

## Critérios de Decisão

### 1. Consultar a fonte de verdade

Ordem de precedência:
1. Documentação oficial do produto (regras de negócio)
2. Políticas definidas em `policies.yaml` do domínio
3. Comportamento atual em produção (se consistente)
4. Consenso do squad responsável

### 2. Princípio da menor surpresa

Se ambas interpretações são defensáveis, escolha a que o **usuário final** esperaria. O agente deve ser previsível.

### 3. Princípio da segurança

Em caso de dúvida entre uma resposta mais permissiva e uma mais conservadora, preferir a conservadora — especialmente em:
- Transações financeiras
- Dados sensíveis
- Ações irreversíveis

### 4. Teste do determinismo

Prefira a interpretação que permite avaliação determinística. Se uma opção requer LLM-as-judge e outra permite grader code-based, preferir a segunda.

## Como Documentar a Decisão

Cada adjudicação deve ser registrada no caso:

```yaml
adjudication:
  date: "2026-05-15"
  adjudicator: "nome.sobrenome"
  original_disagreement:
    annotator_a: "Transfer deve falhar por saldo insuficiente"
    annotator_b: "Transfer deve pedir confirmação antes de falhar"
  decision: "Transfer deve falhar com mensagem de saldo insuficiente"
  rationale: |
    Regra PIX-003 no policies.yaml define que validação de saldo
    ocorre antes de qualquer confirmação. Sem saldo, não há o que confirmar.
  reference: "policies.yaml#PIX-003"
  precedent: true  # Serve como referência para casos futuros similares
```

## Registro de Precedentes

Decisões marcadas como `precedent: true` devem ser adicionadas ao arquivo:

```
domains/<dominio>/precedents.yaml
```

Formato:

```yaml
- id: "PREC-001"
  domain: "pix_whatsapp"
  summary: "Validação de saldo ocorre antes de confirmação do usuário"
  date: "2026-05-15"
  case_ids: ["abc123"]
  rationale: "Regra PIX-003 no policies.yaml"
```

Anotadores devem consultar precedentes ANTES de escalar para adjudicação.

## Processo de Escalação

```
1. Anotadores discordam
   ↓
2. Consultam precedents.yaml — resolve?
   → Sim: aplicar precedente, documentar
   → Não: continuar
   ↓
3. Discutem entre si (max 10 min) — resolve?
   → Sim: documentar decisão
   → Não: continuar
   ↓
4. Escalar para adjudicador (tech lead do domínio)
   ↓
5. Adjudicador decide em até 48h
   ↓
6. Decisão documentada + registrada como precedente se aplicável
```

## Métricas de Adjudicação

Monitorar mensalmente:
- Taxa de adjudicação (% de casos que precisaram) — target: < 10%
- Tempo médio de resolução — target: < 48h
- Categorias mais frequentes de discordância (indica gaps nas guidelines)

Se taxa > 15%, revisar o annotation guide e adicionar exemplos para os casos mais comuns.
