"""Processed-file manifest — the mechanism behind incremental auto-expansion.

Every source file in ``raw/`` is hashed; only files whose hash is new or changed are
re-processed on a subsequent run, and their results are merged into the existing graph.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class Manifest:
    """Maps relative file path -> {hash, n_triples, processed_at}."""

    def __init__(self, path: Path):
        self.path = path
        self.entries: dict[str, dict] = {}
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                self.entries = json.load(f).get("files", {})

    def is_processed(self, rel_path: str, digest: str) -> bool:
        entry = self.entries.get(rel_path)
        return entry is not None and entry.get("hash") == digest

    def record(self, rel_path: str, digest: str, n_triples: int, processed_at: str) -> None:
        self.entries[rel_path] = {
            "hash": digest,
            "n_triples": n_triples,
            "processed_at": processed_at,
        }

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump({"files": self.entries}, f, ensure_ascii=False, indent=2)
