"""A bilingual (English/Spanish) general-purpose AI agent.

Un agente de IA bilingüe (español/inglés) de propósito general.
"""

from __future__ import annotations

__version__ = "0.1.0"
__all__ = ["Agent", "Config", "__version__"]


def __getattr__(name: str):  # lazy so `import agent` doesn't require the SDK
    if name == "Agent":
        from .core import Agent

        return Agent
    if name == "Config":
        from .config import Config

        return Config
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
