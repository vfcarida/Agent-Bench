# Adjudication and Disagreement Resolution Guide

This guide establishes the formal adjudication protocol for resolving ambiguities, conflicting annotations, and edge-case interpretations in `Agent-Bench`.

---

## 1. When Adjudication is Required

Formal adjudication must be triggered when:

1. Two domain annotators disagree on `expected_tools`, `expected_state_mutations`, or `expected_final_response`.
2. Disagreement arises over whether an agent trajectory is "correct" or "defensible".
3. An enterprise policy rule in `policies.yaml` is ambiguous or subject to competing interpretations.
4. An unprecedented edge case is discovered during evaluation failure audits.

---

## 2. Decision Principles & Hierarchy

When adjudicating conflicting interpretations, follow this strict priority order:

### I. Authoritative Ground Truth
1. Official business domain policies (e.g., Central Bank regulatory requirements, banking product rules).
2. Domain policy rules defined in `configs/domains/<domain>.yaml` or domain policy sets.
3. Current verified production behavior.
4. Technical consensus of the domain engineering team.

### II. Principle of Least Surprise
If multiple paths are defensible, select the trajectory that an informed human end-user would reasonably anticipate. Autonomous agent interactions must be predictable and intuitive.

### III. Principle of Conservative Safety
When choosing between a permissive vs. conservative outcome, always favor the conservative posture:
- Financial transactions (require strict verification and balance validation before execution).
- Access to sensitive user data or credentials.
- Irreversible state changes (account closures, deletions).

### IV. Deterministic Eval Prioritization
Prefer expected outcomes that can be validated deterministically by code over those requiring fuzzy LLM-as-judge heuristics.

---

## 3. Documenting Adjudications

Every resolved disagreement must be recorded directly in the evaluation case or adjudication registry:

```yaml
adjudication:
  date: "2026-09-20"
  adjudicator: "maintainer@agent-bench.org"
  original_disagreement:
    annotator_a: "Transfer must fail immediately due to insufficient balance."
    annotator_b: "Transfer should ask the user for confirmation before failing."
  decision: "Transfer must fail immediately with an insufficient balance error."
  rationale: |
    Domain rule PIX-003 requires balance verification prior to transaction
    confirmation prompts. Prompting for confirmation on an unexecutable transfer
    degrades user experience and introduces race conditions.
  policy_reference: "configs/domains/pix_assist.yaml#PIX-003"
  precedent: true
```

---

## 4. Precedent Registry

Decisions flagged with `precedent: true` must be recorded in the domain precedent index (`configs/domains/<domain>/precedents.yaml`):

```yaml
- id: "PREC-001"
  domain: "pix_assist"
  summary: "Balance validation strictly precedes transaction confirmation"
  date: "2026-09-20"
  related_case_ids: ["pix-012", "pix-045"]
  rationale: "Rule PIX-003 in configs/domains/pix_assist.yaml"
```

Annotators and benchmark maintainers must consult the precedent index before opening new adjudication tickets.

---

## 5. Escalation Workflow

```
1. Annotation Disagreement Identified
   ↓
2. Consult Precedent Registry — Precedent exists?
   → YES: Apply precedent and document case ID.
   → NO: Proceed to peer review.
   ↓
3. Peer Sync (Timebox: 10 minutes) — Consensus reached?
   → YES: Document agreed resolution.
   → NO: Escalate to Domain Adjudicator.
   ↓
4. Domain Adjudicator Review (SLA: 48 hours)
   ↓
5. Decision Finalized + Precedent Logged
```

---

## 6. Adjudication Metrics & Quality Health

Track these indicators monthly:
- **Adjudication Rate**: Percentage of cases requiring adjudication (Target: $< 10\%$).
- **Resolution Latency**: Average time to decision (Target: $< 48$ hours).
- **Recurrent Root Causes**: Frequent disputes in a specific domain signal ambiguous guidelines, requiring clarification in `docs/annotation_guide.md`.
