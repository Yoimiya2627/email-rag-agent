"""Tool visibility policy for MCP-facing tool registration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import config.settings as cfg
from agents.tool_registry import ToolSpec


@dataclass(frozen=True)
class ToolPolicy:
    """Decide which tools an MCP server should expose."""

    allowed_tools: set[str] | None = None
    read_only: bool = False

    @classmethod
    def from_settings(cls) -> "ToolPolicy":
        allowed = set(cfg.MCP_ALLOWED_TOOLS) if cfg.MCP_ALLOWED_TOOLS else None
        return cls(allowed_tools=allowed, read_only=cfg.MCP_READ_ONLY_MODE)

    def visible_specs(self, registry: Mapping[str, ToolSpec]) -> dict[str, ToolSpec]:
        """Return registry entries visible under this policy."""
        visible: dict[str, ToolSpec] = {}
        for name, spec in registry.items():
            if self.allowed_tools is not None and name not in self.allowed_tools:
                continue
            if self.read_only and (spec.requires_approval or spec.risk_level != "low"):
                continue
            visible[name] = spec
        return visible
