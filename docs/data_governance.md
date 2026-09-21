# Data Governance

## Dataset Lifecycle

```
draft → reviewed → approved → gold
```

| Lifecycle State | Description |
|-----------------|-------------|
| `draft` | Newly created case, pending peer review. May contain formatting or logic discrepancies. |
| `reviewed` | Reviewed by at least one domain expert or peer annotator. Action items may be pending. |
| `approved` | Passed automated schema validation, logical consistency checks, and human sign-off. Ready for evaluation suites. |
| `gold` | Promoted to the authoritative canonical benchmark. Immutable once released (modifications require a new version identifier). |

## Source Types (`source_type`)

| Source Type | Description |
|-------------|-------------|
| `human_gold` | Authored and verified end-to-end by human domain specialists. Highest confidence tier. |
| `synthetic_shadow` | Generated via automated LLM pipelines and executed in shadow mode against production agents. Validated by statistical sampling. |
| `synthetic_candidate` | Machine-generated test case awaiting human expert validation before promotion. |
| `adversarial` | Targeted attack cases: prompt injection, jailbreak attempts, malformed payload injections, and security boundary bypasses. |
| `calibration` | Benchmark cases with verified ground-truth labels used to calibrate deterministic rubrics and LLM judges. |

## Authoritative Dataset Hierarchy

`Agent-Bench` consolidates evaluation datasets under a single authoritative structure:

1. **`datasets/gold/<split>/<domain>.yaml` (Authoritative Canonical Gold)**:
   - Primary standard for production benchmarking and official leaderboard scoring.
   - Enforces `EvalCase v2` schema with physical directory partitioning by split (`dev`, `holdout`, `calibration`, `regression`, `smoke`).
2. **`datasets/synthetic/` and `datasets/adversarial/`**:
   - High-volume automated test cases and dedicated red-team adversarial probes.
3. **`configs/domains/<domain>.yaml` (System Configurations)**:
   - Defines agent system architectures, tool declarations, mock environment parameters, and scoring profile weights (does not contain test cases).

## Split Strategy

| Split | Intended Usage | Typical Share |
|-------|----------------|---------------|
| `dev` | Day-to-day prompt engineering, system iteration, and local debugging. | 60% |
| `holdout` | Official metric reporting and final release gating. **NEVER** tune prompts or systems against holdout cases. | 20% |
| `calibration` | Empirical calibration of code-based rubrics and LLM judge agreement metrics. | 5% |
| `regression` | Historical failure cases retained to prevent regression of fixed bugs. | 10% |
| `smoke` | Fast connectivity and smoke test suite for rapid CI validation (<30s runtime). | 5% |

Splits are explicitly declared in each test case via the `split` field and organized in corresponding filesystem directories (`datasets/gold/<split>/`).

## Contamination & Holdout Leakage Gate

To preserve scientific rigor and prevent benchmark test sets (`holdout`) from leaking into training corpora or development sets:

- **Verbatim Detection**: Compares SHA-256 cryptographic hashes of `prompt | initial_state` and checks normalized exact-match string equality.
- **Paraphrase Detection**: Tokenized Jaccard similarity across prompt texts, enforcing a standard threshold (default: 0.80) with length filtering.
- **CI Automated Enforcement**: `bench check-contamination` and `scripts/check_contamination.py` execute automatically in the offline CI lane. Any detected leakage immediately fails the build.

## Sensitive Data & PII Policy

**Real customer data is strictly forbidden.** All evaluation cases must adhere to the following:

- **Synthetic Identifiers**: Use valid algorithmically generated identifiers (e.g., checksum-valid test tax IDs / CPFs, synthetic bank accounts) that belong to reserved test ranges.
- **Irreversible Anonymization**: If test cases are inspired by real-world interaction patterns, all names, accounts, timestamps, and locations must be completely synthesized.
- **Automated PII Scanning**: Schema validation tools reject cases containing patterns of real customer personally identifiable information (PII).

## Provenance & Lineage Tracking

Each case records mandatory provenance metadata within its `metadata` block:

- `created_by`: Author identifier (human specialist handle, pipeline name, or migration run).
- `reviewer`: Domain reviewer or lead engineer responsible for sign-off.
- `review_status`: Explicit state (`draft`, `reviewed`, `approved`).
- `generation_recipe_id`: Recipe or pipeline ID for synthetic cases, detailing generator model and temperature.

## Inter-Annotator Agreement & Validation Quality

- **Current Validation Pipeline**: Test cases pass schema validation (`validate_eval_case`), automated tool/state consistency verification (`consistency_checker`), duplicate detection (`dedup_checker`), and single-specialist peer review.
- **Statistical Agreement (Cohen's Kappa / Krippendorff's Alpha)**: Formal multi-annotator agreement metrics will be calculated and reported as redundant annotations are introduced for subjective quality tiers. Unsubstantiated prior claims have been removed in favor of documented, reproducible validation scripts.

## Promotion Criteria: Synthetic → Gold

A candidate case from the synthetic pipeline may only be promoted to canonical `gold` status when:

1. A human domain expert reviews and approves the full prompt and context.
2. `expected_outcome` and `expected_state_changes` are manually validated against domain specifications.
3. The case executes consistently against reference agents across multiple iterations (3+ runs).
4. Deterministic code-based graders evaluate the case unambiguously with deterministic scores.
5. Signed off by the lead evaluator or domain owner.
6. Provenance is updated with `promoted_from: synthetic_candidate` and a reference to the review log.
