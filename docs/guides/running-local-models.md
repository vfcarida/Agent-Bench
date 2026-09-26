# Guide: Running Local Open-Weights Models (vLLM & HuggingFace)

Agent-Bench provides native support for evaluating open-weights LLMs (such as LLaMA-3, Qwen-2.5, Mistral, and DeepSeek) locally or on private GPU infrastructure using **vLLM** and **HuggingFace Transformers**.

---

## 1. Structured Tool-Calling for Open-Weights Models

Standard open-weights models often lack native proprietary function-calling APIs. Agent-Bench automatically handles this via `agent_bench.models.tool_formatter`:

1. **System Prompt Injection**: Tools and schemas are dynamically injected into the system prompt with strict schema definitions and examples.
2. **Tool Call Syntax**: Models are prompted to format tool invocations using XML or JSON blocks:
   ```xml
   <tool_call>
   {"name": "check_balance", "arguments": {}}
   </tool_call>
   ```
3. **Robust Output Extraction**: Generated responses are automatically parsed across XML tags, markdown blocks, and JSON blocks, returning structured `ModelResponse.tool_calls`.

---

## 2. High-Throughput vLLM Setup

Install vLLM dependencies:
```bash
pip install -e ".[vllm]"
```

Configure your model manifest in `configs/models/vllm_llama.yaml`:

```yaml
model_id: vllm_llama3_8b
provider: vllm
model_name: meta-llama/Meta-Llama-3-8B-Instruct
temperature: 0.0
max_tokens: 4096
gpu_memory_utilization: 0.85
price_per_1k_input: 0.0002
price_per_1k_output: 0.0002
```

Configure your benchmark system in `configs/systems/llama_reactive.yaml`:
```yaml
system_id: llama_reactive
model: vllm_llama3_8b
architecture: tool_calling_reactive
system_prompt: "You are a helpful banking and operations assistant. Call tools when necessary."
```

Run evaluation:
```bash
bench --config-dir configs run-suite pix_basic_v1 --system llama_reactive
```

---

## 3. Local HuggingFace Pipeline Setup

For single-GPU or developer workstation testing:

Install HuggingFace dependencies:
```bash
pip install -e ".[huggingface]"
```

Configure `configs/models/hf_qwen.yaml`:

```yaml
model_id: hf_qwen_7b
provider: huggingface
model_name: Qwen/Qwen2.5-7B-Instruct
device_map: auto
torch_dtype: bfloat16
price_per_1k_input: 0.0001
price_per_1k_output: 0.0001
```

---

## 4. Offline Verification & Regression Testing

In CI environments without GPU hardware, Agent-Bench provides deterministic, zero-cost reference evaluation:

```bash
# Evaluate scripted policy without GPU or API keys
bench --config-dir configs run-suite pix_basic_v1 --runner scripted
```
