# Evaluation Case Annotation Guide

This guide outlines standards and best practices for authoring, reviewing, and validating evaluation cases within `Agent-Bench`.

---

## 1. Principles of a High-Quality Eval Case

Every evaluation case must satisfy five core criteria:

1. **Unambiguous**: A clear, deterministic standard for success. Any valid alternate formulation must be explicitly accounted for in acceptance matchers.
2. **Atomic**: Focuses on a single logical workflow or failure mode (e.g., balance validation, tool call sequencing, refusal of prompt injection).
3. **Realistic**: Reflects realistic human user interactions, enterprise domain nuances, and plausible multi-turn requests.
4. **Programmatically Verifiable**: Evaluation should rely on deterministic code-based graders (`ExactMatch`, `ToolMatch`, `StateCheck`) whenever possible.
5. **Hermetic & Isolated**: Test cases must never rely on external network requests or side-effects from previous test cases.

---

## 2. Authoring `expected_tools`

Define the tool actions required to fulfill the user's intent:

```yaml
expected_tools:
  - name: "validate_recipient"
    arguments:
      key_type: "cpf"
      key_value: "123.456.789-00"
  - name: "execute_transfer"
    arguments:
      amount: 150.00
      currency: "BRL"
```

- **Order Sensitivity**: When sequencing matters, ensure tools are declared in exact execution order.
- **Dynamic Matchers**: For fields with nondeterministic outputs (e.g., transaction IDs, timestamps), use wildcard matchers rather than hardcoded literals.

---

## 3. Authoring `expected_state_mutations`

Specify exact mutations that the agent's actions must cause in the task environment:

```yaml
expected_state_mutations:
  - path: "account.balance"
    operation: "decrease_by"
    value: 150.00
  - path: "account.transaction_history"
    operation: "append"
    value:
      type: "pix_out"
      amount: 150.00
```

Supported state mutation operations:
- `equals`: Strict value equality.
- `increase_by` / `decrease_by`: Numeric deltas.
- `append`: Element added to list.
- `contains`: Key or substring presence.
- `not_null` / `is_null`: Nullability assertion.
- `changed`: Value mutated from initial state.

---

## 4. Authoring `expected_final_response`

Defines how the agent's textual communication is graded:

### Exact Match (Structured Outputs / Specific Codes)
```yaml
expected_final_response:
  matcher: "exact"
  expected: "TRANSFER_SUCCESSFUL:TX-9981"
```

### Substring Matchers (`contains_all` / `forbidden`)
```yaml
expected_final_response:
  matcher: "contains"
  contains_all:
    - "transferência de R$ 150,00"
    - "concluída com sucesso"
  forbidden:
    - "falha"
    - "saldo insuficiente"
```

### Semantic Matcher (Calibrated Embedding Similarity)
```yaml
expected_final_response:
  matcher: "semantic"
  reference: "The PIX transfer of R$ 150.00 to receiver João Silva was completed."
  threshold: 0.85
```

---

## 5. Grading Strategy Selection

| Grading Strategy | Primary Use Case | Grader Implementation |
|------------------|------------------|-----------------------|
| `deterministic` | Structured payloads, exact state changes, and verified tool calls. (Recommended) | `ExactMatchGrader`, `ToolMatchGrader`, `StateCheckGrader` |
| `tool_match` | Trajectory verification focusing on correct tool routing and valid arguments. | `ToolMatchGrader` |
| `state_check` | State mutation verification in interactive sandboxes. | `StateCheckGrader` |
| `rubric` | Qualitative assessment with calibrated scoring criteria (e.g., clarity, domain tone). | `RubricGrader` |
| `composite` | Multi-phase gating combining deterministic checks with qualitative rubrics. | `CompositeGrader` / `GatedEvaluator` |

**Maintainer Rule**: Always prefer deterministic graders. Escalate to LLM rubrics only when assessing subjective conversational nuance.

---

## 6. Common Annotation Pitfalls

| Pitfall | Operational Risk | Recommended Fix |
|---------|------------------|-----------------|
| **Brittle exact text match** | False failures due to punctuation, spacing, or greetings | Use `contains_all` or normalize strings before evaluation |
| **Implicit environment state** | Non-reproducible test failures across different environments | Fully declare all fields in `initial_state` |
| **Unanchored timestamps** | Tests pass today but fail next month | Provide mock reference clocks in test case context |
| **Leaking Holdout Data** | Contamination of evaluation benchmarks | Enforce `split: dev` or `split: holdout` directory separation |
| **Hardcoding Real PII** | Security and compliance breach | Use synthetic, checksum-valid test identifiers |

---

## 7. Pre-Submission Quality Checklist

Before committing an evaluation case to `datasets/gold/`:

- [ ] `id` is a unique, descriptive slug or UUID.
- [ ] `split` is explicitly declared (`dev`, `holdout`, `calibration`, `regression`, `smoke`).
- [ ] `initial_state` explicitly initializes all relevant environment keys.
- [ ] `expected_tools` specifies exact argument schemas.
- [ ] `expected_state_mutations` captures all side effects.
- [ ] Zero real customer PII (no authentic CPFs, bank accounts, or real phone numbers).
- [ ] Provenance metadata is fully populated (`created_by`, `review_status: approved`).
- [ ] Case validated locally using CLI smoke test:
  ```bash
  bench run-case <task_id> --domain <domain> --system mock
  ```
- [ ] Offline contamination check passes:
  ```bash
  bench check-contamination
  ```
