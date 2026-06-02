# Adding a New Model Provider

## 1. Implement ModelAdapter

Create `src/agent_bench/models/<provider>.py`:

```python
from agent_bench.core.adapters import ModelAdapter, ModelResponse

class MyProviderAdapter(ModelAdapter):
    @property
    def model_id(self) -> str:
        return "my-model-v1"

    @property
    def provider(self) -> str:
        return "my_provider"

    async def generate(self, messages, *, tools=None, temperature=0.0, max_tokens=4096, seed=None):
        # Call your API here
        ...
        return ModelResponse(content=response_text, tokens_in=..., tokens_out=...)
```

## 2. Register in Config

Add to `configs/models/<provider>.yaml`:

```yaml
models:
  - model_id: my-model-v1
    provider: my_provider
    endpoint: "https://api.myprovider.com/v1"
    api_key_env: MY_PROVIDER_API_KEY
    parameters:
      temperature: 0.0
      max_tokens: 4096
```

## 3. Reference in Systems

```yaml
systems:
  - system_id: tool_calling_my_model
    architecture: tool_calling_reactive
    model: my-model-v1
    tools: [...]
```

## 4. Set Environment Variable

Add `MY_PROVIDER_API_KEY=...` to your `.env`.
