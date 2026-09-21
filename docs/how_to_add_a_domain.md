# How to Add a Benchmark Domain

This guide details the procedure for authoring, configuring, validating, and registering a new evaluation domain within Agent-Bench.

---

## 1. Domain Concept

A **domain** represents an operational category of agent tasks sharing common tool sets, regulatory guardrails, and scoring profiles. Existing domains include:
- `pix_assist`: Instant banking payment flows, Central Bank regulations, and transaction limits.
- `investment_advisor`: Financial product suitability, fixed income calculations, and risk disclosures.
- `sme_business_advisor`: Small business tax calculations, DRE statements, and cash flow projections.
- `cyber_sandbox`: Defensive and offensive security, privilege escalation, and prompt injection refusal.

---

## 2. Step-by-Step Domain Addition

### Step 1: Define Systems Configuration (`configs/domains/<domain_id>.yaml`)

Create `configs/domains/<domain_id>.yaml` to define candidate agent architectures and their allowed toolsets:

```yaml
systems:
  - system_id: prompt_only_gpt4
    architecture: prompt_only
    model: gpt-4o
    tools: []
    retrieval: null
    memory: false
    max_steps: 1

  - system_id: tool_calling_reactive_gpt4
    architecture: tool_calling_reactive
    model: gpt-4o
    tools:
      - check_balance
      - validate_pix_key
      - execute_pix_transfer
    retrieval: null
    memory: false
    max_steps: 10
```

### Step 2: Author Golden Evaluation Cases (`datasets/gold/dev/<domain_id>.yaml`)

Author task cases following the typed **EvalCase v2** schema:

```yaml
family: transactional_tools
domain: my_domain
version: 1.0.0
source_type: human_gold
split: dev
cases:
  - id: MY_001
    version: 1.0.0
    name: Standard Happy Path Flow
    description: User requests valid operation
    family: transactional_tools
    domain: my_domain
    locale: pt-BR
    difficulty: medium
    risk_level: medium
    source_type: human_gold
    split: dev
    prompt_or_user_goal: Execute operation with key 123
    input_messages:
      - role: user
        content: Execute operation with key 123
    initial_state:
      balance: 1000.0
    allowed_tools:
      - validate_key
      - execute_action
    forbidden_tools: []
    expected_outcome:
      state_changes:
        operation_done: true
      refusal_expected: false
    expected_state_changes:
      operation_done: true
    evidence_requirements:
      - tool_calling
    grading_strategy: state_based
    rubric: {}
    tags:
      - happy_path
    severity: medium
    business_criticality: operational
    expected_refusal_mode: none
```

### Step 3: Register in Benchmark Suite (`configs/suites/<suite_id>.yaml`)

Include your new domain in an existing suite or create a new suite manifest:

```yaml
suite_id: my_domain_basic_v1
name: My Domain Basic Evaluation Suite
version: 1.0.0
domains:
  - my_domain
systems:
  - prompt_only_gpt4
  - tool_calling_reactive_gpt4
weighting_profile: transactional_high_risk
repeat_n: 3
seed: 42
```

### Step 4: Validate and Verify

Run validation checks to verify schema and configuration correctness:

```bash
# 1. Validate configuration files
bench --config-dir configs validate-config

# 2. Check gold dataset integrity
python scripts/check_gold_integrity.py

# 3. Check for contamination leakage
bench check-contamination

# 4. Run single case smoke test
bench --config-dir configs run-case MY_001 --system tool_calling_reactive_gpt4 --domain my_domain
```

---

## 3. Dataset Lifecycle & Splits

Agent-Bench enforces strict split isolation:
- `dev`: Active development and prompt optimization.
- `holdout`: Held-out evaluation tasks reserved strictly for final benchmark scoring. Never loaded by development runs.
- `calibration`: Labeled samples used for calibrating LLM judges against human annotators.
- `regression`: Static golden cases run in CI to detect model regressions.
