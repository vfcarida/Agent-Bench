# Dataset Card: [DATASET_NAME]

## Identification

| Field | Value |
|-------|-------|
| Name | `[dataset_name]` |
| Version | `[x.y.z]` |
| Family | `[gold | synthetic | adversarial | mixed]` |
| Domain | `[domain_name]` |
| Owner | `[team / maintainer]` |
| Contact | `[email or issue tracker link]` |
| Created Date | `[YYYY-MM-DD]` |
| Last Updated | `[YYYY-MM-DD]` |

---

## Split Partitioning & Volume

| Split | Case Count | Share (%) |
|-------|------------|-----------|
| `dev` | [N] | [X%] |
| `holdout` | [N] | [X%] |
| `smoke` | [N] | [X%] |
| `regression` | [N] | [X%] |
| `calibration` | [N] | [X%] |
| **Total** | **[N]** | **100%** |

### Breakdown by Source Type (`source_type`)

| Source Type | Case Count | Description |
|-------------|------------|-------------|
| `human_gold` | [N] | Authored and vetted by domain experts |
| `synthetic_shadow` | [N] | Model-generated in shadow evaluation mode |
| `synthetic_candidate` | [N] | Machine-generated, awaiting gold promotion |
| `adversarial` | [N] | Red-teamed security and jailbreak probes |

---

## Dataset Provenance & Generation Methodology

### Gold Cases
Describe creation process (e.g., "Manually authored by certified domain specialists based on anonymized real-world support trajectories").

### Synthetic Cases
Describe generation recipe (e.g., "Generated using pipeline recipe `recipe-pix-v2` with temperature=0.7 and seed=42; 25% of cases randomly sampled and verified by human annotators").

### Adversarial Probes
Describe threat model and attack vectors (e.g., "Crafted during red-teaming exercises targeting prompt injection, unauthorized privilege escalation, and monetary boundary bypasses").

---

## Scenario Coverage & Known Gaps

### Scenarios Covered
- [Scenario 1: e.g., "PIX payment via valid CPF key — nominal happy path"]
- [Scenario 2: e.g., "PIX transfer with insufficient account balance"]
- [Scenario 3: ...]

### Out-of-Scope / Known Gaps
- [Gap 1: e.g., "Future-dated scheduled recurring payments"]
- [Gap 2: ...]

---

## Known Limitations

- [Limitation 1: e.g., "Synthetic cases feature standardized punctuation compared to real-world messy mobile input"]
- [Limitation 2: ...]

---

## Intended Use & Restrictions

- **Intended Use**: Benchmark autonomous agent orchestration, tool routing, and policy guardrails in offline CI and regression suites.
- **Prohibited Use**: Model fine-tuning (especially on holdout splits) or training data contamination.

---

## Integrity & Verification

| Verification Check | Status / Value |
|--------------------|----------------|
| Primary File SHA-256 | `[sha256_hash]` |
| Schema Validation (`EvalCase v2`) | `[PASS / date]` |
| Automated PII Scan | `[PASS / date]` |
| Contamination Gate Check | `[PASS / date]` |

---

## Revision History

| Version | Date | Changes & Description |
|---------|------|-----------------------|
| `1.0.0` | [YYYY-MM-DD] | Initial release |
