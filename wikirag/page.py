"""The wiki page — the unit an agent observes when standing on an entity.

AD-RAG as originally formulated expands a frontier: at each step it enumerates
$\\mathcal{N}(v)$ for every $v$ in the frontier and hands the union to an LLM. On this
graph that is the whole cost. `washi` has degree 326, so one visit produces a candidate
pool larger than most prompts should carry, and the pool grows with every accepted node.
The measured consequence in the original setup was ~92 s of retrieval per question.

Here the observation is one page. A page shows what a reader of an encyclopedia entry
sees — the name and its aliases, what is known about it, its pictures, and its outgoing
links grouped under the relation that produced them. Its size is a property of the
rendering, not of the graph: links are grouped by relation, each group is capped, and a
hub is summarised ("326 links, showing 24") rather than dumped. So the prompt stays
bounded whether the walker is standing on `chiritori` or on `washi`.

Grouping by relation is not only for size. A flat neighbour list makes every step look
alike; grouped links let the navigator reason at the level of *kind* of connection
("the question asks what it is made of, so look under made_from") before committing to
a specific neighbour, which is the decision the walk actually needs to make.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Link:
    relation: str
    target: str
    incoming: bool = False   # rendered as "<- X" so the walker knows the direction


@dataclass
class Page:
    name: str
    aliases: list[str] = field(default_factory=list)
    types: list[str] = field(default_factory=list)
    text: list[str] = field(default_factory=list)
    images: list[dict] = field(default_factory=list)
    links: list[Link] = field(default_factory=list)
    truncated: dict[str, int] = field(default_factory=dict)  # relation -> links hidden

    def render(self, with_text: bool = True, with_links: bool = True,
               hide_captions: frozenset = frozenset()) -> str:
        """Render the page. `hide_captions` names image paths whose caption is the
        gold answer for the item being evaluated: the picture stays, its wording
        goes. Without this a caption travels with the entity, and a question whose
        answer is what a photograph shows is settled by reaching the entity rather
        than by looking -- which is the one thing the multimodal items exist to
        measure."""
        """Plain-text rendering handed to the model."""
        out = [f"# {self.name}"]
        if self.aliases:
            out.append(f"also known as: {', '.join(self.aliases)}")
        if self.types:
            out.append(f"kind: {', '.join(self.types)}")
        if with_text and self.text:
            out.append("")
            out.extend(self.text)
        if self.images:
            caps = [("(uncaptioned)" if i.get("path") in hide_captions
                     else i.get("caption") or "(uncaptioned)")
                    for i in self.images]
            out.append("")
            out.append(f"[{len(self.images)} image(s)] " + " | ".join(c[:90] for c in caps[:3]))
        if with_links and self.links:
            # Two sections, not one list with arrows. What this entity asserts and what
            # merely points at it answer different questions -- "what is known about X" is
            # answered by X's own facts, while the back-links are other pages' facts that
            # happen to mention X. Marking the difference with a "<-" prefix inside one
            # list was not enough: asked for an entity's history the navigator returned
            # "Heisei Asakusa washi evolved_from Asakusa-gami" instead of its origin year.
            # A real encyclopedia separates the article from what links to it.
            for title, incoming in (("## Facts", False), ("## Mentioned by", True)):
                grouped: dict[str, list[Link]] = {}
                for l in self.links:
                    if l.incoming is incoming:
                        grouped.setdefault(l.relation, []).append(l)
                if not grouped:
                    continue
                out.append("")
                out.append(title)
                for rel in sorted(grouped):
                    line = f"- {rel}: " + ", ".join(l.target for l in grouped[rel])
                    hidden = self.truncated.get(rel, 0)
                    if hidden:
                        line += f"  (+{hidden} more)"
                    out.append(line)
        return "\n".join(out)

    def link_targets(self) -> list[str]:
        return [l.target for l in self.links]


class WikiGraph:
    """Renders pages from the cleaned graph, and nothing else.

    Holds no search state: the walker owns that. This is deliberately a read-only view
    so a page can be rendered identically no matter which walker asks for it.
    """

    def __init__(
        self,
        path: str | Path,
        max_links_per_relation: int = 8,
        max_total_links: int = 40,
        include_incoming: bool = True,
        max_text: int = 3,
        text_only: bool = False,
        drop_edges: float = 0.0,
        drop_seed: int = 0,
    ):
        g = json.loads(Path(path).read_text())
        self.E: dict = g["entities"]
        if text_only:
            # The text-only ablation asks what the graph is worth without A_img, so the
            # images have to be gone from the graph itself -- not merely skipped at root
            # initialisation. Stripping them here means no page shows a caption, no
            # walker can be drawn to a picture, and no answerer is handed one.
            for e in self.E.values():
                e["attributes"]["img"] = []
        self.text_only = text_only
        self.T: list = g["triples"]
        if drop_edges > 0:
            # A curated heritage graph is never finished, so the question is how the walk
            # degrades when facts are simply absent. Edges go uniformly at random, gold
            # ones included: protecting them would measure a graph nobody has. The share
            # of gold that survives is reported alongside accuracy so the two causes --
            # evidence that is gone, and search that cannot find what remains -- stay
            # separable. Seeded, so a rate is the same graph on every backbone.
            import random as _random
            rng = _random.Random(drop_seed)
            self.T = [t for t in self.T if rng.random() >= drop_edges]
        self.dropped_edges = drop_edges
        self.meta: dict = g.get("meta") or {}
        self.rel_class: dict = self.meta.get("relation_class") or {}
        self.max_links_per_relation = max_links_per_relation
        self.max_total_links = max_total_links
        self.include_incoming = include_incoming
        self.max_text = max_text

        self.out: dict[str, list[tuple[str, str]]] = {}
        self.inn: dict[str, list[tuple[str, str]]] = {}
        for t in self.T:
            self.out.setdefault(t["h"], []).append((t["r"], t["u"]))
            self.inn.setdefault(t["u"], []).append((t["r"], t["h"]))

        self.alias_index: dict[str, str] = {}
        for name, e in self.E.items():
            self.alias_index[name.lower()] = name
            for a in e.get("aliases", []):
                self.alias_index.setdefault(a.lower(), name)

    def resolve(self, mention: str) -> str | None:
        """Map a surface form (including a Japanese alias) to a canonical entity."""
        if mention in self.E:
            return mention
        return self.alias_index.get(mention.strip().lower())

    def page(self, name: str) -> Page | None:
        ent = self.E.get(name)
        if ent is None:
            return None

        # Order relations by how much they narrow the graph: a relation used once on this
        # page is far more informative than one used twenty times, and when the budget
        # forces a cut it should fall on the generic bulk, not on the distinctive edge.
        by_rel: dict[str, list[Link]] = {}
        for r, u in self.out.get(name, []):
            by_rel.setdefault(r, []).append(Link(r, u))
        if self.include_incoming:
            for r, h in self.inn.get(name, []):
                by_rel.setdefault(r, []).append(Link(r, h, incoming=True))

        links: list[Link] = []
        truncated: dict[str, int] = {}
        budget = self.max_total_links
        for rel in sorted(by_rel, key=lambda r: (len(by_rel[r]), r)):
            group = by_rel[rel]
            take = min(len(group), self.max_links_per_relation, max(budget, 0))
            if take < len(group):
                truncated[rel] = len(group) - take
            links.extend(group[:take])
            budget -= take

        return Page(
            name=name,
            aliases=ent.get("aliases", [])[:6],
            types=(ent.get("types") or ([ent["type"]] if ent.get("type") else []))[:3],
            text=ent["attributes"]["text"][: self.max_text],
            images=ent["attributes"]["img"],
            links=links,
            truncated=truncated,
        )

    def has_edge(self, h: str, r: str, u: str) -> bool:
        return (r, u) in self.out.get(h, [])

    def degree(self, name: str) -> int:
        return self.E.get(name, {}).get("degree", 0)

    def images_of(self, name: str) -> list[dict]:
        return self.E.get(name, {}).get("attributes", {}).get("img", [])
