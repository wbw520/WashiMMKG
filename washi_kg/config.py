"""Configuration loading for the WashiMMKG pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Config:
    raw: dict[str, Any]
    base_dir: Path

    # ---- convenience accessors -------------------------------------------------
    def get(self, *keys: str, default: Any = None) -> Any:
        node: Any = self.raw
        for k in keys:
            if not isinstance(node, dict) or k not in node:
                return default
            node = node[k]
        return node

    @property
    def raw_dir(self) -> Path:
        return self.base_dir / self.get("paths", "raw_dir", default="raw")

    @property
    def output_dir(self) -> Path:
        return self.base_dir / self.get("paths", "output_dir", default="output")

    @property
    def graph_path(self) -> Path:
        return self.output_dir / self.get("paths", "graph_file", default="graph.json")

    @property
    def manifest_path(self) -> Path:
        return self.output_dir / self.get("paths", "manifest_file", default="manifest.json")


def load_config(path: str | os.PathLike | None = None) -> Config:
    """Load ``config.yaml``. ``path`` defaults to the file next to ``code/``."""
    if path is None:
        path = Path(__file__).resolve().parent.parent / "config.yaml"
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return Config(raw=raw, base_dir=path.resolve().parent)
