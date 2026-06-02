# Benchmark Governance

## Principles

1. **No real customer data** — all fixtures are synthetic
2. **Versioned** — every suite, dataset, and rubric has a version
3. **Auditable** — every run produces a manifest with config hashes
4. **Reproducible** — fixed seeds, deterministic configs, artifact persistence
5. **Approved changes** — dataset/rubric changes require review

## Dataset Provenance

Each dataset file must include:
- `version`: semantic version
- `domain`: owning domain
- Description of synthetic data generation method

## Change Management

- Dataset changes require PR review from domain owner
- Rubric changes require review from evaluation team
- Suite version must be bumped on any config change
- Config hash in artifacts proves what config was used

## Redaction

- `governance.redaction_hooks` can strip sensitive patterns before storage
- Denylist of field patterns (CPF, card numbers, etc.)
- Pre-storage validation ensures no PII in artifacts

## Ownership

Each benchmark suite must declare:
- `owner`: team responsible
- `reviewers`: who can approve changes
- `schedule`: cadence for re-evaluation (e.g., weekly, per-release)
