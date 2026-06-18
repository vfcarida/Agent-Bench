"""Tests for Athena local model adapters (vLLM, HuggingFace, PEFT)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_bench.core.adapters import ModelResponse


# ──────────────────────────────────────────────────────────────────
# vLLM Adapter Tests
# ──────────────────────────────────────────────────────────────────


class TestVLLMAdapterImportGuard:
    """Test graceful failure when vLLM is not installed."""

    def test_import_error_without_vllm(self):
        """VLLMModelAdapter should raise ImportError with instructions."""
        with patch.dict("sys.modules", {"vllm": None}):
            # The check happens at instantiation time
            from agent_bench.models.vllm_adapter import VLLM_AVAILABLE

            if not VLLM_AVAILABLE:
                from agent_bench.models.vllm_adapter import _check_vllm_available

                with pytest.raises(ImportError, match="vLLM is not installed"):
                    _check_vllm_available()


class TestVLLMAdapterProperties:
    """Test VLLMModelAdapter when vLLM is mocked as available."""

    def test_model_id_default(self):
        """model_id defaults to model_path when not overridden."""
        from agent_bench.models.vllm_adapter import VLLMModelAdapter

        adapter = VLLMModelAdapter.__new__(VLLMModelAdapter)
        adapter._model_path = "/path/to/model"
        adapter._model_id_override = "/path/to/model"
        adapter._engine = None
        adapter._request_counter = 0

        assert adapter.model_id == "/path/to/model"
        assert adapter.provider == "vllm"

    def test_model_id_override(self):
        """model_id can be overridden via constructor."""
        from agent_bench.models.vllm_adapter import VLLMModelAdapter

        adapter = VLLMModelAdapter.__new__(VLLMModelAdapter)
        adapter._model_path = "/path/to/model"
        adapter._model_id_override = "my-custom-id"
        adapter._engine = None

        assert adapter.model_id == "my-custom-id"

    def test_format_prompt(self):
        """_format_prompt should concatenate messages."""
        with patch("agent_bench.models.vllm_adapter.VLLM_AVAILABLE", True):
            from agent_bench.models.vllm_adapter import VLLMModelAdapter

            adapter = VLLMModelAdapter.__new__(VLLMModelAdapter)
            messages = [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello!"},
            ]
            prompt = adapter._format_prompt(messages)
            assert "system: You are helpful." in prompt
            assert "user: Hello!" in prompt


# ──────────────────────────────────────────────────────────────────
# HuggingFace Adapter Tests
# ──────────────────────────────────────────────────────────────────


class TestHuggingFaceAdapterImportGuard:
    """Test graceful failure when transformers is not installed."""

    def test_import_error_without_transformers(self):
        """Should raise ImportError with install instructions."""
        from agent_bench.models.huggingface_adapter import HF_AVAILABLE

        if not HF_AVAILABLE:
            from agent_bench.models.huggingface_adapter import _check_hf_available

            with pytest.raises(ImportError, match="Hugging Face transformers"):
                _check_hf_available()


class TestHuggingFaceAdapterProperties:
    """Test HuggingFacePipelineAdapter properties."""

    def test_properties(self):
        """Verify model_id and provider properties."""
        with patch("agent_bench.models.huggingface_adapter.HF_AVAILABLE", True):
            from agent_bench.models.huggingface_adapter import HuggingFacePipelineAdapter

            adapter = HuggingFacePipelineAdapter.__new__(HuggingFacePipelineAdapter)
            adapter._model_path = "/path/to/model"
            adapter._model_id_override = "athena-7b"
            adapter._loaded = False

            assert adapter.model_id == "athena-7b"
            assert adapter.provider == "huggingface"

    def test_default_model_id(self):
        """model_id defaults to model_path."""
        with patch("agent_bench.models.huggingface_adapter.HF_AVAILABLE", True):
            from agent_bench.models.huggingface_adapter import HuggingFacePipelineAdapter

            adapter = HuggingFacePipelineAdapter.__new__(HuggingFacePipelineAdapter)
            adapter._model_path = "meta-llama/Llama-2-7b-hf"
            adapter._model_id_override = "meta-llama/Llama-2-7b-hf"

            assert adapter.model_id == "meta-llama/Llama-2-7b-hf"


class TestDtypeResolution:
    """Test torch dtype string resolution."""

    def test_resolve_known_dtypes(self):
        """Known dtype strings should resolve correctly."""
        from agent_bench.models.huggingface_adapter import HF_AVAILABLE

        if HF_AVAILABLE:
            import torch
            from agent_bench.models.huggingface_adapter import _resolve_torch_dtype

            assert _resolve_torch_dtype("float16") == torch.float16
            assert _resolve_torch_dtype("bfloat16") == torch.bfloat16
            assert _resolve_torch_dtype("float32") == torch.float32
            assert _resolve_torch_dtype("auto") == "auto"

    def test_resolve_unknown_dtype(self):
        """Unknown dtype strings should fall back to 'auto'."""
        from agent_bench.models.huggingface_adapter import _resolve_torch_dtype

        result = _resolve_torch_dtype("unknown_dtype")
        assert result == "auto"


# ──────────────────────────────────────────────────────────────────
# PEFT / AdapterManager Tests
# ──────────────────────────────────────────────────────────────────


class TestAdapterManagerImportGuard:
    """Test graceful failure when PEFT is not installed."""

    def test_import_error_without_peft(self):
        """Should raise ImportError with install instructions."""
        from agent_bench.models.adapter_manager import PEFT_AVAILABLE

        if not PEFT_AVAILABLE:
            from agent_bench.models.adapter_manager import _check_peft_available

            with pytest.raises(ImportError, match="PEFT is not installed"):
                _check_peft_available()


class TestAdapterManagerLifecycle:
    """Test AdapterManager load/swap/unload lifecycle (mocked)."""

    def _make_manager(self):
        """Create a mocked AdapterManager."""
        from agent_bench.models.adapter_manager import AdapterManager, PEFT_AVAILABLE

        if not PEFT_AVAILABLE:
            pytest.skip("PEFT not installed")

        mgr = AdapterManager.__new__(AdapterManager)
        mgr._base_model_path = "/fake/base"
        mgr._device = "cpu"
        mgr._torch_dtype = "auto"
        mgr._trust_remote_code = False
        mgr._model = MagicMock()
        mgr._tokenizer = MagicMock()
        mgr._adapters = {}
        mgr._active_adapter = None
        mgr._loaded = True
        return mgr

    def test_list_adapters_empty(self):
        """New manager should have no adapters."""
        mgr = self._make_manager()
        assert mgr.list_adapters() == []
        assert mgr.get_active_adapter() is None

    def test_set_adapter_not_loaded_raises(self):
        """Setting a non-existent adapter should raise ValueError."""
        mgr = self._make_manager()
        with pytest.raises(ValueError, match="not loaded"):
            mgr.set_active_adapter("nonexistent")

    def test_unload_nonexistent(self):
        """Unloading a non-existent adapter should be a no-op."""
        mgr = self._make_manager()
        mgr.unload_adapter("nonexistent")  # should not raise


class TestPEFTModelAdapter:
    """Test PEFTModelAdapter delegation."""

    def test_model_id_format(self):
        """model_id should be 'base:adapter' by default."""
        from agent_bench.models.peft_adapter import PEFTModelAdapter

        mock_manager = MagicMock()
        mock_manager.base_model_path = "/path/to/base"

        adapter = PEFTModelAdapter(mock_manager, "math-lora")
        assert adapter.model_id == "/path/to/base:math-lora"
        assert adapter.provider == "peft"
        assert adapter.adapter_name == "math-lora"

    def test_model_id_override(self):
        """model_id can be overridden."""
        from agent_bench.models.peft_adapter import PEFTModelAdapter

        mock_manager = MagicMock()
        mock_manager.base_model_path = "/path/to/base"

        adapter = PEFTModelAdapter(mock_manager, "math-lora", model_id="custom-id")
        assert adapter.model_id == "custom-id"

    @pytest.mark.asyncio
    async def test_generate_delegates_to_manager(self):
        """generate() should call manager.set_active_adapter and generate_async."""
        from agent_bench.models.peft_adapter import PEFTModelAdapter

        mock_manager = MagicMock()
        mock_manager.base_model_path = "/path/to/base"
        mock_manager.set_active_adapter = MagicMock()
        mock_manager.generate_async = AsyncMock(return_value=ModelResponse(
            content="test output",
            tokens_in=10,
            tokens_out=5,
            latency_ms=100.0,
        ))

        adapter = PEFTModelAdapter(mock_manager, "math-lora")
        messages = [{"role": "user", "content": "test"}]

        result = await adapter.generate(messages)

        mock_manager.set_active_adapter.assert_called_once_with("math-lora")
        mock_manager.generate_async.assert_called_once()
        assert result.content == "test output"
