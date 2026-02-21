"""
Per-analysis in-memory tool result cache.

Shared across all sub-agents within a single deep analysis run.
Eliminates redundant API calls when multiple sub-agents
(research -> debate -> rebuttal) query the same data.

Usage:
    cache = AnalysisToolCache()
    wrapped_tools = cache.wrap_tools(original_tools)
    # Pass wrapped_tools to sub-agent factory
"""

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import wraps
from typing import Any

import structlog

logger = structlog.get_logger()


@dataclass
class AnalysisToolCache:
    """In-memory cache for tool results, scoped to one analysis run."""

    _cache: dict[str, str] = field(default_factory=dict, repr=False)
    _hits: int = field(default=0, repr=False)
    _misses: int = field(default=0, repr=False)

    @property
    def stats(self) -> dict[str, Any]:
        """Cache statistics for observability."""
        total = self._hits + self._misses
        rate = f"{self._hits / total * 100:.1f}%" if total > 0 else "0.0%"
        return {"hits": self._hits, "misses": self._misses, "hit_rate": rate}

    def _make_key(self, tool_name: str, kwargs: dict[str, Any]) -> str:
        """Generate a canonical cache key from tool name and inputs."""
        return f"{tool_name}:{json.dumps(kwargs, sort_keys=True, default=str)}"

    def wrap_tools(self, tools: list[Any]) -> list[Any]:
        """Wrap a list of LangChain tools with caching.

        Returns new tool objects with the same name/description/schema
        but with cached async invocation.
        """
        return [self._wrap_single(tool) for tool in tools]

    def _wrap_single(self, tool: Any) -> Any:
        """Wrap a single LangChain tool with caching."""
        original_fn: Callable = tool.coroutine or tool.func
        cache = self

        @wraps(original_fn)
        async def cached_invoke(**kwargs: Any) -> str:
            key = cache._make_key(tool.name, kwargs)
            if key in cache._cache:
                cache._hits += 1
                return cache._cache[key]
            cache._misses += 1
            result = await original_fn(**kwargs)
            cache._cache[key] = result
            return result

        # Build a lightweight wrapper preserving tool metadata
        wrapped = _CachedToolWrapper(
            name=tool.name,
            description=tool.description,
            args_schema=tool.args_schema,
            coroutine=cached_invoke,
        )
        return wrapped

    def log_stats(self) -> None:
        """Log cache stats at end of analysis."""
        logger.info("Analysis tool cache stats", **self.stats)


class _CachedToolWrapper:
    """Minimal tool wrapper that satisfies deepagents' tool interface."""

    def __init__(
        self,
        name: str,
        description: str,
        args_schema: Any,
        coroutine: Callable,
    ):
        self.name = name
        self.description = description
        self.args_schema = args_schema
        self.coroutine = coroutine
        self.func = None  # deepagents checks this attribute
