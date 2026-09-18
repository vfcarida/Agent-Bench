# AB-T01 — Preflight: Reproducible Environment + Baseline Capture

- **Task ID**: AB-T01
- **Repository**: [Agent-Bench](https://github.com/vfcarida/Agent-Bench)
- **Baseline SHA**: `66bcf95d366b31a7e7e2d67992a68e7d67c76d32` (branch `main`)
- **Date / Timestamp**: 2026-09-18T16:08:43-03:00 / 2026-09-18T19:08:43Z
- **Execution Mode**: Local, offline only, no paid APIs, no network dependencies.

---

## 1. Environment & Preflight Checks

### Git Preflight
- `git rev-parse HEAD`: `66bcf95d366b31a7e7e2d67992a68e7d67c76d32` (matches expected baseline SHA exactly)
- `git status`: Clean working tree (`nothing to commit, working tree clean`)

### Host System & Python
- **OS**: Windows NT 10.0.26200 (Microsoft Windows 11 / Windows NT 10.0.26200.0 amd64)
- **Host Python**: Python 3.12.10 (`C:\Users\vinicius\AppData\Local\Programs\Python\Python312\python.exe`)
- **Virtual Environment**: `.venv` created via `python3.12 -m venv .venv`

### Package Installation
- **Command**: `.\.venv\Scripts\python.exe -m pip install -e ".[dev]"`
- **Exit Code**: `0`
- **Scope Compliance**: Installed only `.[dev]` (did NOT install `.[all]`, `.[vllm]`, or `.[huggingface]`).

### Environment Inventory (`pip freeze`)
```text
-e git+https://github.com/vfcarida/Agent-Bench.git@66bcf95d366b31a7e7e2d67992a68e7d67c76d32#egg=agent_bench
annotated-types==0.8.0
anyio==4.15.1
ast_serialize==0.11.2
certifi==2026.7.22
click==8.5.0
colorama==0.4.6
coverage==7.16.1
h11==0.16.0
httpcore==1.0.9
httpx==0.28.1
idna==3.20
iniconfig==2.3.0
Jinja2==3.1.6
librt==0.15.0
markdown-it-py==4.2.0
MarkupSafe==3.0.3
mdurl==0.1.2
mypy==2.3.1
mypy_extensions==1.1.0
packaging==26.3
pathspec==1.1.1
pluggy==1.6.0
pyarrow==25.0.1
pydantic==2.13.5
pydantic-settings==2.15.0
pydantic_core==2.46.5
Pygments==2.21.0
pytest==9.1.1
pytest-asyncio==1.4.0
pytest-cov==7.1.0
python-dotenv==1.2.3
PyYAML==6.0.3
rich==15.0.0
ruff==0.16.8
structlog==26.1.0
typing-inspection==0.4.4
typing_extensions==4.16.0
```

---

## 2. CI Discrepancy Observation

In `.github/workflows/bench.yml` line 31:
```yaml
      - name: Code Quality - Ruff Linting
        run: |
          ruff check src/ agent_bench tests/
```
- **Discrepancy**: The CI workflow specifies `ruff check src/ agent_bench tests/`. However, the repository source layout places the package under `src/agent_bench`. There is no top-level `agent_bench/` directory.
- Running `ruff check agent_bench` directly produces:
  ```text
  E902 The system cannot find the file specified. (os error 2)
  --> agent_bench:1:1
  Found 1 error.
  ```
- Because of this, running CI's exact command adds a path error (`E902`) on top of source lint issues.

---

## 3. Offline Checks Baseline Results

### 3.1. Ruff Linting (`ruff check src/ tests/`)
- **Command**: `.\.venv\Scripts\ruff.exe check src/ tests/`
- **Exit Code**: `1`
- **Total Errors**: 194 errors (141 fixable with `--fix`, 18 hidden fixes with `--unsafe-fixes`)
- **Note on CI command**: When run with CI's arguments `src/ agent_bench tests/`, total errors = 195 (194 lint errors + 1 E902 missing path error).
- **Rule Breakdown (`--statistics`)**:
  ```text
  54	F401   	[-] unused-import
  49	I001   	[*] unsorted-imports
  19	UP017  	[*] datetime-timezone-utc
  12	BLE001 	[ ] blind-except
   9	S110   	[ ] try-except-pass
   7	UP035  	[-] deprecated-import
   7	UP006  	[*] non-pep585-annotation
   6	F841   	[ ] unused-variable
   5	PLR0124	[ ] comparison-with-itself
   5	C401   	[ ] unnecessary-generator-set
   4	RUF012 	[ ] mutable-class-default
   4	RUF059 	[ ] unused-unpacked-variable
   2	SIM102 	[ ] collapsible-if
   2	F541   	[*] f-string-missing-placeholders
   2	SIM117 	[*] multiple-with-statements
   2	RUF046 	[ ] unnecessary-cast-to-int
   1	B017   	[ ] assert-raises-exception
   1	FURB122	[*] for-loop-writes
   1	UP037  	[*] quoted-annotation
   1	RUF022 	[*] unsorted-dunder-all
   1	RUF100 	[*] unused-noqa
  ```

---

### 3.2. Type Checking (`mypy src/agent_bench`)
- **Command**: `.\.venv\Scripts\mypy.exe src/agent_bench`
- **Exit Code**: `1`
- **Total Errors**: 45 errors in 20 files (checked 97 source files)
- **Log Excerpt**:
  ```text
  src\agent_bench\storage\parquet.py:7: error: Skipping analyzing "pyarrow": module is installed, but missing library stubs or py.typed marker  [import-untyped]
  src\agent_bench\storage\parquet.py:8: error: Skipping analyzing "pyarrow.parquet": module is installed, but missing library stubs or py.typed marker  [import-untyped]
  src\agent_bench\storage\parquet.py:39: error: Unused "type: ignore" comment  [unused-ignore]
  src\agent_bench\storage\parquet.py:77: error: Unused "type: ignore" comment  [unused-ignore]
  src\agent_bench\storage\parquet.py:91: error: Unused "type: ignore" comment  [unused-ignore]
  src\agent_bench\storage\parquet.py:129: error: Unused "type: ignore" comment  [unused-ignore]
  src\agent_bench\storage\analytics.py:6: error: Skipping analyzing "pyarrow.parquet": module is installed, but missing library stubs or py.typed marker  [import-untyped]
  src\agent_bench\storage\analytics.py:6: error: Skipping analyzing "pyarrow": module is installed, but missing library stubs or py.typed marker  [import-untyped]
  src\agent_bench\storage\analytics.py:7: error: Skipping analyzing "pyarrow.compute": module is installed, but missing library stubs or py.typed marker  [import-untyped]
  src\agent_bench\storage\analytics.py:24: error: Unused "type: ignore" comment  [unused-ignore]
  src\agent_bench\metrics\transfer_analysis.py:161: error: Returning Any from function declared to return "float"  [no-any-return]
  src\agent_bench\metrics\difficulty_analysis.py:202: error: Returning Any from function declared to return "float"  [no-any-return]
  src\agent_bench\datasets\exporter.py:5: error: Library stubs not installed for "yaml"  [import-untyped]
  src\agent_bench\core\schema_migration.py:6: error: Library stubs not installed for "yaml"  [import-untyped]
  src\agent_bench\graders\thinking_parser.py:173: error: Returning Any from function declared to return "GradeResult"  [no-any-return]
  src\agent_bench\judges\cross_artifact.py:62: error: Need type annotation for "artifact_facts" (hint: "artifact_facts: list[<type>] = ...")  [var-annotated]
  src\agent_bench\core\config.py:8: error: Library stubs not installed for "yaml"  [import-untyped]
  src\agent_bench\models\vllm_adapter.py:20: error: Cannot find implementation or library stub for module named "vllm"  [import-not-found]
  src\agent_bench\models\peft_adapter.py:89: error: Returning Any from function declared to return "ModelResponse"  [no-any-return]
  src\agent_bench\models\huggingface_adapter.py:21: error: Cannot find implementation or library stub for module named "torch"  [import-not-found]
  src\agent_bench\models\huggingface_adapter.py:22: error: Cannot find implementation or library stub for module named "transformers"  [import-not-found]
  src\agent_bench\models\huggingface_adapter.py:162: error: Returning Any from function declared to return "str"  [no-any-return]
  src\agent_bench\models\adapter_manager.py:25: error: Cannot find implementation or library stub for module named "torch"  [import-not-found]
  src\agent_bench\models\adapter_manager.py:26: error: Cannot find implementation or library stub for module named "peft"  [import-not-found]
  src\agent_bench\models\adapter_manager.py:27: error: Cannot find implementation or library stub for module named "transformers"  [import-not-found]
  src\agent_bench\models\adapter_manager.py:255: error: Returning Any from function declared to return "str"  [no-any-return]
  src\agent_bench\storage\trace_logger.py:7: error: Skipping analyzing "pyarrow": module is installed, but missing library stubs or py.typed marker  [import-untyped]
  src\agent_bench\storage\trace_logger.py:8: error: Skipping analyzing "pyarrow.parquet": module is installed, but missing library stubs or py.typed marker  [import-untyped]
  src\agent_bench\storage\trace_logger.py:218: error: Unused "type: ignore" comment  [unused-ignore]
  Found 45 errors in 20 files (checked 97 source files)
  ```

---

### 3.3. Test Suite (`pytest tests/ -q`)
- **Command**: `.\.venv\Scripts\pytest.exe tests/ -q`
- **Exit Code**: `1`
- **Summary**: `1 failed, 279 passed, 3 skipped in 7.27s`
- **Failure Details**:
  ```text
  ================================== FAILURES ===================================
  _______________ TestDtypeResolution.test_resolve_unknown_dtype ________________

  self = <tests.unit.test_athena_adapters.TestDtypeResolution object at 0x0000017421AE1E80>

      def test_resolve_unknown_dtype(self):
          """Unknown dtype strings should fall back to 'auto'."""
          from agent_bench.models.huggingface_adapter import _resolve_torch_dtype
      
          result = _resolve_torch_dtype("unknown_dtype")
  >       assert result == "auto"
  E       AssertionError: assert None == 'auto'

  tests\unit\test_athena_adapters.py:143: AssertionError
  =========================== short test summary info ===========================
  FAILED tests/unit/test_athena_adapters.py::TestDtypeResolution::test_resolve_unknown_dtype
  1 failed, 279 passed, 3 skipped in 7.27s
  ```

---

### 3.4. Configuration Validation (`bench validate-config`)
- **Command**: `.\.venv\Scripts\bench.exe --config-dir configs validate-config`
- **Exit Code**: `0`
- **Output**:
  ```text
  Config valid.
    Models: 8
    Systems: 10
    Suites: 4
    Hash: bf17520eb586
  ```

---

### 3.5. Suite Golden Run (`bench run-suite pix_basic_v1`) — Vacuous Baseline (Finding F1)
- **Command**: `.\.venv\Scripts\bench.exe --config-dir configs run-suite pix_basic_v1`
- **Exit Code**: `0`
- **Run ID**: `226a72cd-aca2-4d3d-9ff8-37387fbed829`
- **Output**:
  ```text
  Suite completed. Run ID: 226a72cd-aca2-4d3d-9ff8-37387fbed829
    Passed: 20/60
  ```
- **Analysis of Pass**: 2 systems (`prompt_only_gpt4`, `tool_calling_reactive_gpt4`) evaluated against 10 suite tasks = 20 task executions. All 20 passed.
- **Root Cause (Finding F1 - Circular Baseline)**:
  In `src/agent_bench/runners/case_runner.py` lines 245-289, the `_stub_execute` function copies the answer key directly into the result:
  ```python
  def _stub_execute(task: Task, system_id: str) -> dict[str, Any]:
      """Stub execution — simulates system behavior based on task metadata."""
      if "happy_path" in task.tags:
          result: dict[str, Any] = {
              "response": _generate_stub_response(task),
              "final_state": task.expected_final_state,
              "tools_called": task.allowed_tools,
          }
  ```
  Because the stub unconditionally returns `task.expected_final_state` and `task.allowed_tools`, the grader awards full credit regardless of model behavior or quality.

#### Verbatim Scorecards (`data/runs/226a72cd-aca2-4d3d-9ff8-37387fbed829.json`)
```json
{
  "scorecards": [
    {
      "system_id": "prompt_only_gpt4",
      "domain": "pix_assist",
      "functional_score": 1.0,
      "risk_score": 0.8,
      "cost_score": 0.999,
      "latency_score": 1.0,
      "reliability_score": 1.0,
      "global_score": 0.9199,
      "weighting_profile": "transactional_high_risk"
    },
    {
      "system_id": "tool_calling_reactive_gpt4",
      "domain": "pix_assist",
      "functional_score": 1.0,
      "risk_score": 0.8,
      "cost_score": 0.999,
      "latency_score": 1.0,
      "reliability_score": 1.0,
      "global_score": 0.9199,
      "weighting_profile": "transactional_high_risk"
    }
  ],
  "pass_k_results": [
    {
      "system_id": "prompt_only_gpt4",
      "domain": "pix_assist",
      "pass_1": 1.0,
      "pass_3": 1.0,
      "pass_5": 1.0,
      "repeat_n": 3,
      "total_tasks": 10
    },
    {
      "system_id": "tool_calling_reactive_gpt4",
      "domain": "pix_assist",
      "pass_1": 1.0,
      "pass_3": 1.0,
      "pass_5": 1.0,
      "repeat_n": 3,
      "total_tasks": 10
    }
  ]
}
```

---

## 4. Summary & Handoff

| Check | Target / Command | Exit Code | Result Summary |
|---|---|---|---|
| **Environment Setup** | Python 3.12.10 venv + `pip install -e ".[dev]"` | `0` | Success |
| **Linting (Ruff)** | `ruff check src/ tests/` | `1` | 194 errors (141 fixable) |
| **CI Discrepancy** | `ruff check agent_bench` | `1` | E902 path not found |
| **Type Checking (Mypy)** | `mypy src/agent_bench` | `1` | 45 errors in 20 files |
| **Unit Tests (Pytest)** | `pytest tests/ -q` | `1` | 1 failed, 279 passed, 3 skipped |
| **Config Validation** | `bench --config-dir configs validate-config` | `0` | Config valid (Hash: `bf17520eb586`) |
| **Suite Run** | `bench --config-dir configs run-suite pix_basic_v1` | `0` | 20/20 passed; vacuous 1.0 functional score (F1 baseline captured) |

- **Changed Files**: `docs/baseline/AB-T01-baseline.md` (new baseline document).
- **Source/CI Modifications**: None.
- **Next Eligible Tasks**: AB-T02, AB-T03, AB-T07.
