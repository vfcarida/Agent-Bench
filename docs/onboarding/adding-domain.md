# Adding a New Domain

## 1. Generate Template

```bash
bench export-dataset-template <domain_id>
```

## 2. Define Tasks

Edit `data/fixtures/<domain_id>.yaml`:

- Define policy (limits, rules, constraints)
- List available tools
- Define success criteria
- Create tasks with:
  - `happy_path` cases (expected to succeed)
  - `edge_case` cases (boundary conditions)
  - `refusal` cases (policy violations the system must refuse)
  - `policy_conflict` cases (user request vs domain policy)

## 3. Configure Systems

Create `configs/domains/<domain_id>.yaml` with system definitions that reference models and tools.

## 4. Add to Suite

Reference the domain in `configs/suites/<suite>.yaml`.

## 5. (Optional) Custom Tools

Implement `ToolAdapter` subclasses in `src/agent_bench/tools/<domain>_tools.py`.

## 6. (Optional) Custom Judges

Implement domain-specific judges for specialized evaluation criteria.

## 7. Run

```bash
bench --config-dir configs run-suite <suite_id>
```
