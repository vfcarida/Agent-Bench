"""Tests for Phase D: online eval, governance, observability, plugins, analytics."""

import json
from pathlib import Path

import pytest

from agent_bench.governance.redaction import RedactionEngine, default_denylist
from agent_bench.governance.provenance import ProvenanceRegistry
from agent_bench.governance.versioning import BenchmarkVersioning
from agent_bench.utils.observability import Span, SpanCollector, bench_span
from agent_bench.utils.plugins import PluginRegistry, get_registry


# === Redaction Engine ===

class TestRedactionEngine:
    def test_redact_cpf(self):
        engine = RedactionEngine(default_denylist())
        text = "O CPF do cliente é 123.456.789-00 e ele mora em SP."
        result = engine.redact_text(text)
        assert "123.456.789-00" not in result
        assert "[CPF_REDACTED]" in result

    def test_redact_cnpj(self):
        engine = RedactionEngine(default_denylist())
        text = "CNPJ: 12.345.678/0001-90"
        result = engine.redact_text(text)
        assert "12.345.678/0001-90" not in result
        assert "[CNPJ_REDACTED]" in result

    def test_redact_email(self):
        engine = RedactionEngine(default_denylist())
        text = "Envie para joao.silva@empresa.com.br"
        result = engine.redact_text(text)
        assert "joao.silva@empresa.com.br" not in result
        assert "[EMAIL_REDACTED]" in result

    def test_redact_card_number(self):
        engine = RedactionEngine(default_denylist())
        text = "Cartão 4111 1111 1111 1111"
        result = engine.redact_text(text)
        assert "4111 1111 1111 1111" not in result
        assert "[CARD_REDACTED]" in result

    def test_redact_phone(self):
        engine = RedactionEngine(default_denylist())
        text = "Telefone: (11) 99999-8888"
        result = engine.redact_text(text)
        assert "(11) 99999-8888" not in result
        assert "[PHONE_REDACTED]" in result

    def test_no_redaction_clean_text(self):
        engine = RedactionEngine(default_denylist())
        text = "Este é um texto normal sem dados sensíveis."
        result = engine.redact_text(text)
        assert result == text

    def test_multiple_redactions(self):
        engine = RedactionEngine(default_denylist())
        text = "CPF 123.456.789-00 email test@mail.com"
        result = engine.redact_text(text)
        assert "123.456.789-00" not in result
        assert "test@mail.com" not in result

    def test_redact_dict(self):
        engine = RedactionEngine(default_denylist())
        data = {"nome": "João", "cpf": "123.456.789-00", "nested": {"email": "a@b.com"}}
        result = engine.redact_dict(data)
        assert "123.456.789-00" not in result["cpf"]
        assert "a@b.com" not in result["nested"]["email"]

    def test_scan_finds_patterns(self):
        engine = RedactionEngine(default_denylist())
        text = "CPF 123.456.789-00 e CNPJ 12.345.678/0001-90"
        findings = engine.scan(text)
        assert len(findings) >= 2
        rule_names = [f["rule"] for f in findings]
        assert "cpf" in rule_names
        assert "cnpj" in rule_names

    def test_add_custom_rule(self):
        from agent_bench.governance.redaction import RedactionRule
        engine = RedactionEngine([])
        engine.add_rule(RedactionRule(name="custom", pattern=r"SECRET_\w+", replacement="[CUSTOM]"))
        result = engine.redact_text("My SECRET_TOKEN is here")
        assert "SECRET_TOKEN" not in result
        assert "[CUSTOM]" in result


# === Provenance Registry ===

class TestProvenanceRegistry:
    def test_register_and_verify(self, tmp_path):
        registry_file = tmp_path / "provenance.json"
        registry = ProvenanceRegistry(registry_file)
        test_file = tmp_path / "test_dataset.yaml"
        test_file.write_text("tasks:\n  - id: task_1\n")

        registry.register(test_file, domain="test", version="1.0.0")
        assert registry.check_integrity(test_file, "test", "1.0.0") is True

    def test_verify_modified_file(self, tmp_path):
        registry_file = tmp_path / "provenance.json"
        registry = ProvenanceRegistry(registry_file)
        test_file = tmp_path / "test_dataset.yaml"
        test_file.write_text("tasks:\n  - id: original\n")
        registry.register(test_file, domain="test", version="1.0.0")

        test_file.write_text("tasks:\n  - id: modified\n")
        assert registry.check_integrity(test_file, "test", "1.0.0") is False

    def test_list_registered(self, tmp_path):
        registry_file = tmp_path / "provenance.json"
        registry = ProvenanceRegistry(registry_file)
        for i in range(3):
            f = tmp_path / f"dataset_{i}.yaml"
            f.write_text(f"tasks:\n  - id: t{i}\n")
            registry.register(f, domain=f"domain_{i}", version="1.0.0")

        entries = registry.list_all()
        assert len(entries) == 3

    def test_has_changed(self, tmp_path):
        registry_file = tmp_path / "provenance.json"
        registry = ProvenanceRegistry(registry_file)
        test_file = tmp_path / "ds.yaml"
        test_file.write_text("tasks: []\n")
        registry.register(test_file, domain="d", version="1.0")

        assert registry.has_changed(test_file, "d", "1.0") is False
        test_file.write_text("tasks:\n  - new\n")
        assert registry.has_changed(test_file, "d", "1.0") is True

    def test_persistence(self, tmp_path):
        registry_file = tmp_path / "provenance.json"
        test_file = tmp_path / "ds.yaml"
        test_file.write_text("tasks: []\n")

        registry1 = ProvenanceRegistry(registry_file)
        registry1.register(test_file, domain="x", version="1.0")

        # Reload from disk
        registry2 = ProvenanceRegistry(registry_file)
        assert len(registry2.list_all()) == 1


# === Benchmark Versioning ===

class TestBenchmarkVersioning:
    def test_initial_version(self, tmp_path):
        versioning = BenchmarkVersioning(tmp_path / "versions.json")
        assert versioning.current_version == "0.0.0"

    def test_record_version(self, tmp_path):
        versioning = BenchmarkVersioning(tmp_path / "versions.json")
        entry = versioning.record_version("1.0.0", config_hash="abc", changes=["Initial release"])
        assert entry.version == "1.0.0"
        assert versioning.current_version == "1.0.0"

    def test_multiple_versions(self, tmp_path):
        versioning = BenchmarkVersioning(tmp_path / "versions.json")
        versioning.record_version("0.1.0", config_hash="a", changes=["First"])
        versioning.record_version("0.2.0", config_hash="b", changes=["Second"])
        versioning.record_version("1.0.0", config_hash="c", changes=["Major"])
        assert versioning.current_version == "1.0.0"

    def test_get_changelog(self, tmp_path):
        versioning = BenchmarkVersioning(tmp_path / "versions.json")
        versioning.record_version("0.1.0", config_hash="a", changes=["Feature A"])
        versioning.record_version("0.2.0", config_hash="b", changes=["Feature B"])
        versioning.record_version("1.0.0", config_hash="c", changes=["Breaking change"])

        log = versioning.get_changelog(since_version="0.1.0")
        assert len(log) == 2
        assert log[0].version == "0.2.0"
        assert log[1].version == "1.0.0"

    def test_render_changelog_markdown(self, tmp_path):
        versioning = BenchmarkVersioning(tmp_path / "versions.json")
        versioning.record_version("1.0.0", config_hash="x", changes=["Added PIX domain"])
        md = versioning.render_changelog_markdown()
        assert "1.0.0" in md
        assert "Added PIX domain" in md

    def test_persistence(self, tmp_path):
        vfile = tmp_path / "versions.json"
        v1 = BenchmarkVersioning(vfile)
        v1.record_version("1.0.0", config_hash="h", changes=["init"])

        v2 = BenchmarkVersioning(vfile)
        assert v2.current_version == "1.0.0"

    def test_compute_datasets_hash(self, tmp_path):
        versioning = BenchmarkVersioning(tmp_path / "v.json")
        fixtures = tmp_path / "fixtures"
        fixtures.mkdir()
        (fixtures / "a.yaml").write_text("tasks: []")
        (fixtures / "b.yaml").write_text("tasks: [x]")

        h = versioning.compute_datasets_hash(fixtures)
        assert len(h) == 16


# === Observability ===

class TestSpanCollector:
    def test_basic_span(self):
        collector = SpanCollector()
        with collector.trace("test_op", key="value") as span:
            span.set_attribute("result", "ok")

        assert len(collector.spans) == 1
        assert collector.spans[0].name == "test_op"
        assert collector.spans[0].attributes["key"] == "value"
        assert collector.spans[0].attributes["result"] == "ok"
        assert collector.spans[0].status == "ok"

    def test_nested_spans(self):
        collector = SpanCollector()
        with collector.trace("parent") as parent:
            with collector.trace("child") as child:
                pass

        assert len(collector.spans) == 2
        child_span = collector.spans[0]
        parent_span = collector.spans[1]
        assert child_span.parent_id == parent_span.span_id

    def test_error_span(self):
        collector = SpanCollector()
        with pytest.raises(ValueError):
            with collector.trace("failing_op") as span:
                raise ValueError("test error")

        assert len(collector.spans) == 1
        assert collector.spans[0].status == "error"
        assert "test error" in collector.spans[0].attributes["error"]

    def test_span_duration(self):
        collector = SpanCollector()
        with collector.trace("timed_op"):
            pass

        assert collector.spans[0].duration_ms is not None
        assert collector.spans[0].duration_ms >= 0

    def test_export_otel_format(self):
        collector = SpanCollector()
        with collector.trace("op1"):
            pass

        exported = collector.export_otel_format()
        assert len(exported) == 1
        assert exported[0]["operationName"] == "op1"
        assert "spanId" in exported[0]
        assert "traceId" in exported[0]

    def test_summary(self):
        collector = SpanCollector()
        with collector.trace("a"):
            pass
        with collector.trace("b"):
            pass

        summary = collector.summary()
        assert summary["total_spans"] == 2
        assert summary["error_count"] == 0

    def test_reset(self):
        collector = SpanCollector()
        with collector.trace("x"):
            pass
        collector.reset()
        assert len(collector.spans) == 0

    def test_span_add_event(self):
        collector = SpanCollector()
        with collector.trace("op") as span:
            span.add_event("checkpoint", {"step": 1})

        assert len(collector.spans[0].events) == 1
        assert collector.spans[0].events[0]["name"] == "checkpoint"

    def test_bench_span_convenience(self):
        with bench_span("convenience_op", foo="bar") as span:
            span.add_event("checkpoint")

        assert span.status == "ok"
        assert len(span.events) == 1


# === Plugin Registry ===

class TestPluginRegistry:
    def test_register_and_get(self):
        registry = PluginRegistry()
        registry.register("test_model", "model", "some.module", "TestClass")
        info = registry.get("model", "test_model")
        assert info is not None
        assert info.class_name == "TestClass"

    def test_invalid_plugin_type(self):
        registry = PluginRegistry()
        with pytest.raises(ValueError):
            registry.register("x", "invalid_type", "m", "C")

    def test_list_plugins(self):
        registry = PluginRegistry()
        registry.register("a", "model", "m.a", "A")
        registry.register("b", "judge", "m.b", "B")
        assert len(registry.list_plugins()) == 2
        assert len(registry.list_plugins("model")) == 1

    def test_global_registry_has_builtins(self):
        registry = get_registry()
        assert registry.get("model", "stub") is not None
        assert registry.get("model", "openai") is not None
        assert registry.get("model", "anthropic") is not None
        assert registry.get("judge", "deterministic") is not None
        assert registry.get("judge", "grounding") is not None
        assert registry.get("judge", "numeric") is not None
        assert registry.get("judge", "composite") is not None
        assert registry.get("retrieval", "stub_retrieval") is not None

    def test_load_class_stub(self):
        registry = get_registry()
        cls = registry.load_class("model", "stub")
        assert cls is not None
        assert cls.__name__ == "StubModelAdapter"

    def test_load_class_nonexistent(self):
        registry = get_registry()
        cls = registry.load_class("model", "nonexistent")
        assert cls is None

    def test_create_instance(self):
        registry = get_registry()
        instance = registry.create_instance("model", "stub")
        assert instance is not None

    def test_create_instance_not_found(self):
        registry = PluginRegistry()
        with pytest.raises(ValueError):
            registry.create_instance("model", "missing")

    def test_discover_entry_points(self):
        registry = PluginRegistry()
        count = registry.discover_entry_points()
        assert count >= 0


# === Online Eval ===

class TestOnlineEval:
    def test_load_traces_from_jsonl(self, tmp_path):
        from agent_bench.runners.online_eval import load_traces_from_jsonl

        traces_file = tmp_path / "traces.jsonl"
        traces = [
            {"task_id": "t1", "system_id": "s1", "domain": "pix_assist", "response": "ok"},
            {"task_id": "t2", "system_id": "s1", "domain": "pix_assist", "response": "no"},
        ]
        traces_file.write_text("\n".join(json.dumps(t) for t in traces))

        parsed = load_traces_from_jsonl(traces_file)
        assert len(parsed) == 2
        assert parsed[0]["task_id"] == "t1"

    def test_load_empty_file(self, tmp_path):
        from agent_bench.runners.online_eval import load_traces_from_jsonl

        traces_file = tmp_path / "empty.jsonl"
        traces_file.write_text("")
        parsed = load_traces_from_jsonl(traces_file)
        assert parsed == []

    def test_online_eval_no_matching_tasks(self, tmp_path):
        """Online eval with traces that don't match any registered task."""
        from agent_bench.runners.online_eval import run_online_eval
        from agent_bench.core.config import BenchConfig

        traces_file = tmp_path / "traces.jsonl"
        traces = [
            {"task_id": "nonexistent_task", "system_id": "s1", "domain": "fake", "response": "x"},
        ]
        traces_file.write_text(json.dumps(traces[0]))

        config = BenchConfig(models=[], systems=[], suites=[], judges={})
        import asyncio
        result = asyncio.run(run_online_eval(traces_file, config, tmp_path / "out", domain="fake"))
        assert result.tasks_total == 0


# === Analytics (basic - no parquet files needed) ===

class TestAnalytics:
    def test_aggregate_empty_dir(self, tmp_path):
        from agent_bench.storage.analytics import aggregate_by_system
        results = aggregate_by_system(tmp_path)
        assert results == []

    def test_aggregate_by_domain_empty(self, tmp_path):
        from agent_bench.storage.analytics import aggregate_by_domain
        results = aggregate_by_domain(tmp_path)
        assert results == []

    def test_trend_empty(self, tmp_path):
        from agent_bench.storage.analytics import trend_over_runs
        results = trend_over_runs(tmp_path)
        assert results == []

    def test_cross_system_empty(self, tmp_path):
        from agent_bench.storage.analytics import cross_system_comparison
        results = cross_system_comparison(tmp_path, "pix_assist")
        assert results == []
