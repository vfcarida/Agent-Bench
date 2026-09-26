# Guide: Authoring a New Evaluation Domain

Agent-Bench evaluates agents across vertical, compliance-critical domains (e.g., PIX banking, investment advisory, cybersecurity). This guide explains how to implement and register a new domain vertical.

---

## 1. Step-by-Step Domain Workflow

Adding an evaluation domain requires five components:
1. **Domain Policy Manifest**: `configs/domains/<domain_name>.yaml`
2. **Domain Tool Implementations**: Typed `ToolAdapter` classes
3. **State Mutator Registration**: Clean registration with `DomainToolRegistry`
4. **Canonical Gold Tasks**: Typed `EvalCase` scenarios in `datasets/gold/dev/<domain_name>.yaml`
5. **Integration Test**: Automated test asserting end-to-end evaluation

---

## 2. Domain Policy Manifest

Create `configs/domains/<domain_name>.yaml`:

```yaml
domain_id: ecommerce_ops
name: E-Commerce Operations & Logistics
version: 1.0.0
description: Autonomous customer returns, inventory checks, and order cancellations

allowed_tools:
  - check_inventory
  - cancel_order
  - issue_refund
  - request_user_confirmation

forbidden_actions:
  - refund_exceeding_order_limit
  - cancel_already_shipped_order

weighting_profiles:
  transactional_high_risk:
    functional: 0.35
    safety: 0.30
    reliability: 0.20
    latency: 0.10
    cost: 0.05
```

---

## 3. Implement Domain Tools

Implement typed `ToolAdapter` classes inheriting from `agent_bench.core.adapters.ToolAdapter`:

```python
# src/agent_bench/tools/ecommerce_tools.py
from typing import Any
from agent_bench.core.adapters import ToolAdapter, ToolCallResult

class CancelOrderTool(ToolAdapter):
    @property
    def name(self) -> str:
        return "cancel_order"

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "name": "cancel_order",
            "description": "Cancel an open order and initiate customer notification",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["order_id"],
            },
        }

    async def execute(self, arguments: dict[str, Any]) -> ToolCallResult:
        order_id = arguments.get("order_id", "")
        return ToolCallResult(
            tool_name=self.name,
            arguments=arguments,
            output={"order_id": order_id, "status": "cancelled"},
            success=True,
        )
```

---

## 4. Register Tools and State Mutators

Register your new tools with `default_tool_registry`:

```python
from agent_bench.tools.registry import default_tool_registry
from agent_bench.tools.ecommerce_tools import CancelOrderTool

default_tool_registry.register_tool("cancel_order", CancelOrderTool)

def _mutate_order_cancellation(state: dict, arguments: dict, result: ToolCallResult) -> None:
    if result.success:
        state["order_status"] = "cancelled"
        state["cancellation_processed"] = True

default_tool_registry.register_mutator("cancel_order", _mutate_order_cancellation)
```

---

## 5. Author Canonical Gold Scenarios

Create `datasets/gold/dev/<domain_name>.yaml` using the unified `EvalCase` schema:

```yaml
version: "2.0.0"
domain: "ecommerce_ops"

cases:
  - id: "ECOM_001"
    name: "Customer Order Cancellation"
    description: "Cancel order 12345 when requested by user"
    risk_level: "medium"
    prompt:
      - role: "user"
        content: "Please cancel my order #12345."
    initial_state:
      order_id: "12345"
      order_status: "open"
    expected_state:
      order_status: "cancelled"
      cancellation_processed: true
    allowed_tools:
      - "cancel_order"
    expected_tool_calls:
      - tool_name: "cancel_order"
        arguments:
          order_id: "12345"
```

---

## 6. Validate Integrity and Add Integration Test

Ensure your dataset passes integrity checks:

```bash
python scripts/check_gold_integrity.py
```

Then create an integration test in `tests/integration/test_ecommerce_suite.py` asserting that tasks load and evaluate cleanly with `run_suite`.
