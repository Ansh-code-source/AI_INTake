"""Routing logic between PersonaEngine and RAGEngine.

Provides an explicit Router that returns a mode-specific engine instance.
"""
from typing import Any


class Router:
    """Route validated IntakeConfig to a mode-specific engine instance."""

    def route(self, config: Any) -> Any:
        # Import lazily to avoid cycles
        from .engine import PersonaEngine, RAGEngine

        mode = getattr(config, "mode", None)
        if mode == "persona":
            return PersonaEngine(config)
        if mode == "rag":
            return RAGEngine(config)
        raise ValueError("Unknown mode: must be 'persona' or 'rag'")
