"""The multimodal knowledge graph G = (E, R, T, A).

Stored as a single JSON document so it is easy to inspect, diff, and reload for
incremental expansion.

    entities : canonical name -> {name, type, aliases, attributes:{text, img}, degree}
    relations: sorted list of relation types R
    triples  : list of {h, r, u, sources:[file...]}   (the directed triple set T)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

# sentence boundaries for both Latin-script and Japanese text
_SENT_SPLIT = re.compile(r"(?<=[.!?。！？])\s+|(?<=[。！？])")


def compress_text(text: str, keywords: list[str], max_chars: int) -> str:
    """Compress a paragraph to at most ``max_chars`` characters (entity-aware).

    Extractive, no LLM needed: sentences that mention one of ``keywords`` (the
    entity name + its aliases) are preferred, then remaining sentences in order,
    greedily packed until the budget is reached. A single oversized sentence is
    hard-truncated with an ellipsis so the cap is always honoured.
    """
    text = " ".join(text.split())  # collapse whitespace/newlines
    if len(text) <= max_chars:
        return text
    sents = [s.strip() for s in _SENT_SPLIT.split(text) if s and s.strip()]
    kw = [k.lower() for k in keywords if k]
    hit = [s for s in sents if any(k in s.lower() for k in kw)]
    rest = [s for s in sents if s not in hit]

    out = ""
    for s in hit + rest:
        cand = (out + " " + s).strip() if out else s
        if len(cand) <= max_chars:
            out = cand
        elif not out:  # first sentence already too long -> hard truncate
            return s[: max_chars - 1].rstrip() + "…"
    return out or (text[: max_chars - 1].rstrip() + "…")


@dataclass
class KnowledgeGraph:
    entities: dict[str, dict] = field(default_factory=dict)
    relations: set[str] = field(default_factory=set)
    triples: list[dict] = field(default_factory=list)
    meta: dict = field(default_factory=lambda: {"version": 1})

    # index of (h, r, u) -> position in self.triples, for O(1) merge
    _triple_index: dict[tuple, int] = field(default_factory=dict, repr=False)

    # ---------------------------------------------------------------- load/save
    @classmethod
    def load(cls, path: Path) -> "KnowledgeGraph":
        if not path.exists():
            return cls()
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        kg = cls(
            entities=data.get("entities", {}),
            relations=set(data.get("relations", [])),
            triples=data.get("triples", []),
            meta=data.get("meta", {"version": 1}),
        )
        kg._reindex()
        return kg

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._recompute_degrees()
        data = {
            "meta": self.meta,
            "stats": self.stats(),
            "entities": self.entities,
            "relations": sorted(self.relations),
            "triples": self.triples,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------ entities
    def ensure_entity(self, name: str) -> dict:
        ent = self.entities.get(name)
        if ent is None:
            ent = {
                "name": name,
                "type": None,
                "aliases": [],
                "attributes": {"text": [], "img": []},
                "degree": 0,
            }
            self.entities[name] = ent
        return ent

    def add_alias(self, canonical: str, alias: str) -> None:
        if alias and alias != canonical:
            ent = self.ensure_entity(canonical)
            if alias not in ent["aliases"]:
                ent["aliases"].append(alias)

    def add_text_attr(self, name: str, text: str, cap: int, max_chars: int | None = None) -> None:
        ent = self.ensure_entity(name)
        text = text.strip()
        if not text:
            return
        if max_chars:
            text = compress_text(text, [name] + ent.get("aliases", []), max_chars)
        if text not in ent["attributes"]["text"]:
            ent["attributes"]["text"].append(text)
            del ent["attributes"]["text"][:-cap]  # keep at most `cap` most-recent

    def add_image_attr(
        self,
        name: str,
        image: str,
        caption: str | None = None,
        source: str | None = None,
        page: int | None = None,
        cap: int | None = None,
    ) -> bool:
        """Attach an image to A_img(e). Returns True if it was stored.

        ``cap`` bounds how many images one entity may accumulate; unlike A_text the
        oldest are *kept*, because the first images aligned to an entity are the ones
        a human curated. Returns False when the image is a duplicate or the cap is full.
        """
        ent = self.ensure_entity(name)
        imgs = ent["attributes"]["img"]
        if any(i.get("path") == image for i in imgs):
            return False
        if cap is not None and len(imgs) >= cap:
            return False
        rec: dict = {"path": image, "caption": caption}
        if source is not None:
            rec["source"] = source   # the file the image came from (e.g. a PDF)
        if page is not None:
            rec["page"] = page        # 1-based page number within that file
        imgs.append(rec)
        return True

    @property
    def multimodal_entities(self) -> list[str]:
        """E_mm: entities that carry at least one image attribute."""
        return [n for n, e in self.entities.items() if e["attributes"]["img"]]

    # ------------------------------------------------------------------- triples
    def add_triple(self, h: str, r: str, u: str, source: str | None = None) -> bool:
        """Merge a canonical triple into T. Returns True if it was new."""
        self.ensure_entity(h)
        self.ensure_entity(u)
        self.relations.add(r)
        key = (h, r, u)
        idx = self._triple_index.get(key)
        if idx is None:
            self._triple_index[key] = len(self.triples)
            self.triples.append({"h": h, "r": r, "u": u, "sources": [source] if source else []})
            return True
        if source and source not in self.triples[idx]["sources"]:
            self.triples[idx]["sources"].append(source)
        return False

    def has_triple(self, h: str, r: str, u: str) -> bool:
        return (h, r, u) in self._triple_index

    def neighbors(self, entity: str) -> list[tuple[str, str]]:
        """Outgoing (relation, tail) pairs — used by downstream RAG traversal."""
        return [(t["r"], t["u"]) for t in self.triples if t["h"] == entity]

    # -------------------------------------------------------------------- maint.
    def _reindex(self) -> None:
        self._triple_index = {(t["h"], t["r"], t["u"]): i for i, t in enumerate(self.triples)}

    def _recompute_degrees(self) -> None:
        for e in self.entities.values():
            e["degree"] = 0
        for t in self.triples:
            if t["h"] in self.entities:
                self.entities[t["h"]]["degree"] += 1
            if t["u"] in self.entities:
                self.entities[t["u"]]["degree"] += 1

    def stats(self) -> dict:
        n_img = sum(len(e["attributes"]["img"]) for e in self.entities.values())
        n_txt = sum(len(e["attributes"]["text"]) for e in self.entities.values())
        return {
            "entities": len(self.entities),
            "relations": len(self.relations),
            "triples": len(self.triples),
            "multimodal_entities": len(self.multimodal_entities),
            "images": n_img,
            "text_attributes": n_txt,
        }
