"""Plugin system: dynamic discovery of adapters, judges, and tools."""

import importlib
import importlib.metadata
from dataclasses import dataclass, field
from typing import Any, Type

import structlog

from agent_bench.core.adapters import (
    JudgeAdapter,
    ModelAdapter,
    RetrievalAdapter,
    ToolAdapter,
)

logger = structlog.get_logger()


@dataclass
class PluginInfo:
    name: str
    plugin_type: str  # model | judge | tool | retrieval
    module_path: str
    class_name: str
    metadata: dict[str, Any] = field(default_factory=dict)


class PluginRegistry:
    """Registry for dynamically discovered plugins.

    Plugins can be registered via:
    1. Explicit registration (in code)
    2. Entry points (via pyproject.toml [project.entry-points])
    3. Plugin directories (auto-scan)
    """

    ENTRY_POINT_GROUP = "agent_bench"
    PLUGIN_TYPES = {"model", "judge", "tool", "retrieval"}

    def __init__(self) -> None:
        self._plugins: dict[str, dict[str, PluginInfo]] = {
            t: {} for t in self.PLUGIN_TYPES
        }

    def register(
        self,
        name: str,
        plugin_type: str,
        module_path: str,
        class_name: str,
        **metadata: Any,
    ) -> None:
        """Explicitly register a plugin."""
        if plugin_type not in self.PLUGIN_TYPES:
            raise ValueError(f"Invalid plugin type: {plugin_type}. Must be one of {self.PLUGIN_TYPES}")

        info = PluginInfo(
            name=name,
            plugin_type=plugin_type,
            module_path=module_path,
            class_name=class_name,
            metadata=metadata,
        )
        self._plugins[plugin_type][name] = info
        logger.debug("plugin_registered", name=name, type=plugin_type)

    def discover_entry_points(self) -> int:
        """Discover plugins from installed packages via entry_points.

        Entry points should be in format:
        [project.entry-points."agent_bench.models"]
        my_model = "my_package.models:MyModelAdapter"
        """
        count = 0
        for suffix in self.PLUGIN_TYPES:
            group = f"{self.ENTRY_POINT_GROUP}.{suffix}s"
            try:
                eps = importlib.metadata.entry_points(group=group)
            except TypeError:
                # Python 3.12 compat
                eps = importlib.metadata.entry_points().get(group, [])  # type: ignore[attr-defined]

            for ep in eps:
                self.register(
                    name=ep.name,
                    plugin_type=suffix,
                    module_path=ep.value.split(":")[0],
                    class_name=ep.value.split(":")[1] if ":" in ep.value else ep.name,
                    source="entry_point",
                )
                count += 1

        if count:
            logger.info("plugins_discovered", count=count)
        return count

    def get(self, plugin_type: str, name: str) -> PluginInfo | None:
        """Get a plugin by type and name."""
        return self._plugins.get(plugin_type, {}).get(name)

    def list_plugins(self, plugin_type: str | None = None) -> list[PluginInfo]:
        """List all registered plugins, optionally filtered by type."""
        if plugin_type:
            return list(self._plugins.get(plugin_type, {}).values())
        return [
            info
            for type_plugins in self._plugins.values()
            for info in type_plugins.values()
        ]

    def load_class(self, plugin_type: str, name: str) -> Type[Any] | None:
        """Load and return the actual class for a plugin."""
        info = self.get(plugin_type, name)
        if not info:
            return None

        try:
            module = importlib.import_module(info.module_path)
            return getattr(module, info.class_name)  # type: ignore[no-any-return]
        except (ImportError, AttributeError) as e:
            logger.error("plugin_load_failed", name=name, error=str(e))
            return None

    def create_instance(self, plugin_type: str, name: str, **kwargs: Any) -> Any:
        """Load a plugin class and instantiate it."""
        cls = self.load_class(plugin_type, name)
        if cls is None:
            raise ValueError(f"Plugin not found: {plugin_type}/{name}")
        return cls(**kwargs)


# Global registry with built-in plugins pre-registered
_global_registry = PluginRegistry()

# Register built-in plugins
_global_registry.register(
    "stub", "model",
    "agent_bench.models.stub", "StubModelAdapter",
    builtin=True,
)
_global_registry.register(
    "openai", "model",
    "agent_bench.models.openai_adapter", "OpenAIModelAdapter",
    builtin=True,
)
_global_registry.register(
    "anthropic", "model",
    "agent_bench.models.anthropic_adapter", "AnthropicModelAdapter",
    builtin=True,
)
_global_registry.register(
    "deterministic", "judge",
    "agent_bench.judges.deterministic", "DeterministicJudge",
    builtin=True,
)
_global_registry.register(
    "grounding", "judge",
    "agent_bench.judges.grounding", "GroundingJudge",
    builtin=True,
)
_global_registry.register(
    "numeric", "judge",
    "agent_bench.judges.numeric", "NumericCorrectnessJudge",
    builtin=True,
)
_global_registry.register(
    "composite", "judge",
    "agent_bench.judges.composite", "CompositeJudge",
    builtin=True,
)
_global_registry.register(
    "stub_retrieval", "retrieval",
    "agent_bench.retrieval.stub", "StubRetrievalAdapter",
    builtin=True,
)


def get_registry() -> PluginRegistry:
    return _global_registry
