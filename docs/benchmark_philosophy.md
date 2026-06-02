# Filosofia do Benchmark

## Benchmark de SISTEMA, não de modelo

Este benchmark avalia o **sistema agêntico completo**: orquestração, tool calling, gerenciamento de estado, políticas de negócio e recuperação de erros. O modelo LLM é apenas um componente. Um agente pode falhar com um modelo excelente se a orquestração for ruim — e vice-versa.

Implicação prática: os casos de teste exercitam fluxos end-to-end, não apenas geração de texto.

## Separação Gold / Synthetic / Adversarial

| Tipo | Propósito | Quem cria |
|------|-----------|-----------|
| **Gold** | Ground truth validada por humanos. Fonte de verdade para métricas oficiais. | Especialistas de domínio |
| **Synthetic** | Volume para cobertura estatística. Gerado programaticamente ou via LLM, validado por amostragem. | Pipeline automatizado |
| **Adversarial** | Testa robustez: edge cases, injeções, entradas malformadas. | Red team / fuzzing |

A separação impede contaminação: gold nunca é diluído por dados sintéticos de menor confiança.

## Graders code-based > LLM-as-judge

Preferimos graders determinísticos (código) porque:

1. **Reprodutibilidade** — mesmo input sempre gera mesmo score
2. **Velocidade** — ordens de magnitude mais rápido que chamar LLM
3. **Custo** — zero custo incremental por avaliação
4. **Auditabilidade** — lógica explícita, sem prompt frágil

LLM-as-judge é aceito apenas para dimensões subjetivas (fluência, tom) com rubrics explícitas e calibração documentada.

## Princípios de Reprodutibilidade

- **Seeds fixos** em toda geração sintética (`seed` campo obrigatório no metadata)
- **Versionamento semântico** dos datasets (v1.0.0, v1.1.0...)
- **Hash SHA-256** de cada arquivo de dataset registrado no dataset card
- **Pinagem de dependências** — versão exata do framework e providers no lockfile
- **Timestamp ISO-8601** em todo artefato gerado

## Princípio do Holdout

O split `holdout` é sagrado:

- NUNCA usar para tuning, debugging ou desenvolvimento
- Usado exclusivamente para reportar métricas oficiais
- Acesso restrito (idealmente só CI gera reports sobre holdout)
- Se contaminado, deve ser regenerado integralmente

## Design Provider-Agnostic

O benchmark não assume nenhum provider LLM específico:

- Interface abstrata `LLMProvider` com método `complete(messages, **kwargs)`
- Configuração de provider via variável de ambiente ou config YAML
- Métricas de custo normalizadas (tokens in/out, não currency)
- Nenhum prompt hardcoded com instruções provider-specific (ex: "You are ChatGPT")
- Testes devem passar independente de OpenAI, Anthropic, Azure, Bedrock, etc.
