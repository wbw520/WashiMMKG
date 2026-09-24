"""Graph cleaning — the hygiene pass that precedes QA benchmark construction.

The constructed graph carries three artefacts that are harmless for browsing but
actively damage a graph-derived QA benchmark:

1. **Ontology classes leaked into E.** The Extractor emits `(Iwano Ichibei, is_a,
   person)`, so `person`, `papermaker`, `museum` and ~140 other *classes* sit in E
   as if they were entities. A sampled chain that steps through `person` asserts
   nothing, and the class nodes inflate |E| with non-entities.

2. **Relations of unequal semantic weight.** `is_a` and `associated_with` together
   account for 45% of T. Both are fine as the *last* edge of a reasoning chain but
   vacuous in the interior ("A is associated with B, which is associated with C"),
   which is exactly where a naive K-hop sampler puts them.

3. **One concept split across two spellings.** The Detector left `gasenshi` beside
   `Gasenshi (画仙紙)`: appending a Japanese gloss moves the embedding far enough that
   the pair never reaches the LLM judge. The split halves a concept's degree and
   scatters its images.

This pass merges (3), folds (1) into entity attributes, and annotates (2) so the
chain sampler can act on it. It deliberately does **not** delete low-degree entities: a degree-1
node such as `chiritori` is a real, specific fact and makes a better answer than a
hub does. Nothing is removed except the class nodes and the triples that only ever
existed to point at them.

`graph.json` is never modified; the result is written to `clean_graph_file`. Because
the result is *derived*, re-running would otherwise discard work done on it later —
image alignment in particular runs against the cleaned graph, since aligning against
the uncleaned one would bind images to class nodes about to be deleted. `carry_over_images`
keeps both steps independently re-runnable.

Run (from code/, conda `washi`):
    python tools/clean_graph.py            # write output/graph.v2.json
    python tools/clean_graph.py --dry-run  # report only, write nothing
"""

from __future__ import annotations

import argparse
import collections
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from washi_kg.config import load_config  # noqa: E402
from washi_kg.graph import KnowledgeGraph  # noqa: E402


def carry_over_images(graph: KnowledgeGraph, previous: Path) -> dict:
    """Re-attach A_img from a previous cleaning output onto a freshly cleaned graph.

    Cleaning derives graph.v2.json from graph.json, so on its own a re-run silently
    discards everything added to v2 downstream — and image alignment (tools/align_images.py)
    runs against v2 by design, because aligning against the uncleaned graph would bind
    images to class nodes that cleaning is about to delete. Without this the two steps
    cannot both be re-runnable. Images whose entity no longer exists are reported, not
    dropped in silence.
    """
    if not previous.exists():
        return {"restored": 0, "orphaned": 0}
    old = KnowledgeGraph.load(previous)
    restored = orphaned = 0
    for name, ent in old.entities.items():
        imgs = ent["attributes"]["img"]
        if not imgs:
            continue
        target = name if name in graph.entities else None
        if target is None:
            # the name may have been merged away as a duplicate spelling
            for cand, e in graph.entities.items():
                if name in e.get("aliases", []):
                    target = cand
                    break
        if target is None:
            orphaned += len(imgs)
            continue
        for img in imgs:
            if graph.add_image_attr(
                target, img["path"], img.get("caption"),
                source=img.get("source"), page=img.get("page"),
            ):
                restored += 1
    return {"restored": restored, "orphaned": orphaned}


# ------------------------------------------------------------ surface duplicates
_PAREN = re.compile(r"\([^)]*\)")
_NOISE = re.compile(r"[^0-9a-z\u3040-\u30ff\u4e00-\u9fff]+")


def _surface_key(name: str) -> str:
    """Collapse a display name to the identity the Detector should have matched on."""
    return _NOISE.sub("", _PAREN.sub("", name).lower())


def merge_surface_duplicates(graph: KnowledgeGraph) -> dict:
    """Merge entities that differ only by a parenthetical gloss or casing.

    The Detector left 24 such pairs behind — `gasenshi` beside `Gasenshi (画仙紙)`,
    `hikkake` beside `Hikkake (ひっかけ)`. Embedding pre-filtering is what misses them:
    appending a Japanese gloss moves the vector far enough that the pair never reaches
    the LLM judge. They are one concept split across two nodes, which halves their
    degree, scatters their images, and lets a benchmark pair a node against itself.

    The survivor is the higher-degree spelling; the other becomes an alias, along with
    whatever was inside its parentheses, so the Japanese form stays searchable.
    """
    groups: dict[str, list[str]] = collections.defaultdict(list)
    for name in graph.entities:
        key = _surface_key(name)
        if key:
            groups[key].append(name)
    dups = {k: v for k, v in groups.items() if len(v) > 1}

    remap: dict[str, str] = {}
    for names in dups.values():
        keep = sorted(names, key=lambda n: (-graph.entities[n].get("degree", 0), len(n)))[0]
        for other in names:
            if other == keep:
                continue
            remap[other] = keep
            src, dst = graph.entities[other], graph.entities[keep]
            for alias in [other, *src.get("aliases", [])]:
                graph.add_alias(keep, alias)
            for inner in _PAREN.findall(other):
                graph.add_alias(keep, inner.strip("() "))
            for txt in src["attributes"]["text"]:
                if txt not in dst["attributes"]["text"]:
                    dst["attributes"]["text"].append(txt)
            have = {i.get("path") for i in dst["attributes"]["img"]}
            for img in src["attributes"]["img"]:
                if img.get("path") not in have:
                    dst["attributes"]["img"].append(img)
            if not dst.get("type") and src.get("type"):
                dst["type"] = src["type"]
            del graph.entities[other]

    if remap:
        rebuilt: list[dict] = []
        seen: set[tuple] = set()
        for t in graph.triples:
            h = remap.get(t["h"], t["h"])
            u = remap.get(t["u"], t["u"])
            if h == u:
                continue  # the merge turned this edge into a self-loop
            key = (h, t["r"], u)
            if key in seen:
                continue
            seen.add(key)
            rebuilt.append({**t, "h": h, "u": u})
        graph.triples = rebuilt
        graph.relations = {t["r"] for t in graph.triples}
        graph._reindex()

    return {"groups": len(dups), "merged_away": len(remap),
            "examples": sorted(remap.items())[:8]}


# --------------------------------------------------------------------- detection
def find_type_nodes(
    graph: KnowledgeGraph,
    min_isa_in: int,
    min_isa_in_ratio: float,
    max_out_degree: int,
    protect_attributed: bool = True,
) -> dict[str, int]:
    """Return {class node -> number of entities it classifies}.

    Structural test, so it needs no LLM and no hand-written stop list: a class is
    something many entities point *at* with is_a and which points nowhere itself.

    Structure alone over-fires, though. `senkashi` ("a tough washi used as lining
    and printing paper") and `bark cloth` look identical to `person` in the graph:
    several is_a heads, no outgoing edges. What separates them is `protect_attributed`
    — whether the corpus ever *described* the node. A node with A_text or A_img was
    written about and is an entity; a bare label the Extractor emitted only to type
    something else has no attributes at all. Class-membership is still recorded on
    the subjects either way; protected nodes simply also survive in E.
    """
    in_deg: collections.Counter = collections.Counter()
    out_deg: collections.Counter = collections.Counter()
    isa_in: collections.Counter = collections.Counter()
    for t in graph.triples:
        out_deg[t["h"]] += 1
        in_deg[t["u"]] += 1
        if t["r"] == "is_a":
            isa_in[t["u"]] += 1

    found: dict[str, int] = {}
    for name, n_isa in isa_in.items():
        if n_isa < min_isa_in:
            continue
        if n_isa / max(in_deg[name], 1) < min_isa_in_ratio:
            continue
        if out_deg[name] > max_out_degree:
            continue
        if protect_attributed:
            attrs = graph.entities.get(name, {}).get("attributes", {})
            if attrs.get("text") or attrs.get("img"):
                continue
        found[name] = n_isa
    return found


# ------------------------------------------------------------------------ folding
def fold_type_nodes(graph: KnowledgeGraph, type_nodes: dict[str, int]) -> dict:
    """Move class membership from T into the subject's `type` / `types` attribute.

    An entity keeps its existing `type` string (the Obsidian export reads it); every
    class it belongs to — including that original one — is collected into `types`.
    Triples that merely point at a folded class are dropped; any *other* triple
    touching the class node is kept, and the node with it, so a misfire cannot lose
    a fact.
    """
    # class -> subjects, so each subject learns every class it was asserted to be in
    classes_of: dict[str, list[str]] = collections.defaultdict(list)
    for t in graph.triples:
        if t["r"] == "is_a" and t["u"] in type_nodes:
            classes_of[t["h"]].append(t["u"])

    for subject, classes in classes_of.items():
        ent = graph.entities.get(subject)
        if ent is None:
            continue
        merged: list[str] = []
        for c in ([ent["type"]] if ent.get("type") else []) + classes:
            if c not in merged:
                merged.append(c)
        ent["types"] = merged
        ent["type"] = merged[0]

    kept: list[dict] = []
    dropped = 0
    for t in graph.triples:
        if t["r"] == "is_a" and t["u"] in type_nodes:
            dropped += 1
            continue
        kept.append(t)
    graph.triples = kept

    # A class node survives only if something other than a folded is_a still uses it.
    still_used = {t["h"] for t in kept} | {t["u"] for t in kept}
    removed = [n for n in type_nodes if n not in still_used and n in graph.entities]
    for n in removed:
        del graph.entities[n]

    graph.relations = {t["r"] for t in graph.triples}
    graph._reindex()
    return {
        "type_nodes_detected": len(type_nodes),
        "type_nodes_removed": len(removed),
        "isa_triples_folded": dropped,
        "entities_typed": len(classes_of),
        "removed_names": sorted(removed),
    }


# ---------------------------------------------------------------------- annotation
def annotate(graph: KnowledgeGraph, relation_class: dict, hub_degree: int) -> dict:
    """Record the relation typology and flag hub entities.

    Both are advisory metadata for the chain sampler — no triple is touched.
    """
    taxonomic = set(relation_class.get("taxonomic") or [])
    weak = set(relation_class.get("weak") or [])
    classified = {
        r: ("taxonomic" if r in taxonomic else "weak" if r in weak else "core")
        for r in sorted(graph.relations)
    }

    graph._recompute_degrees()
    hubs = []
    for name, ent in graph.entities.items():
        is_hub = ent.get("degree", 0) >= hub_degree
        if is_hub:
            ent["hub"] = True
            hubs.append(name)
        else:
            ent.pop("hub", None)

    graph.meta["relation_class"] = classified
    graph.meta["hub_degree"] = hub_degree
    n_core = sum(1 for v in classified.values() if v == "core")
    core_triples = sum(1 for t in graph.triples if classified[t["r"]] == "core")
    return {
        "relations_core": n_core,
        "relations_taxonomic": len(taxonomic & graph.relations),
        "relations_weak": len(weak & graph.relations),
        "core_triples": core_triples,
        "hubs": sorted(hubs, key=lambda n: -graph.entities[n]["degree"]),
    }


# ----------------------------------------------------------------------------- cli
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None, help="path to config.yaml")
    ap.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    ap.add_argument("--no-carry-images", action="store_true",
                    help="do not re-attach A_img from the existing cleaning output")
    args = ap.parse_args()

    cfg = load_config(args.config)
    c = cfg.get("cleaning", default={}) or {}
    tn = c.get("type_node", {}) or {}
    out_path = cfg.output_dir / cfg.get(
        "paths", "clean_graph_file", default="graph.v2.json"
    )

    graph = KnowledgeGraph.load(cfg.graph_path)
    before = graph.stats()

    dup = merge_surface_duplicates(graph)
    carried = ({"restored": 0, "orphaned": 0} if args.no_carry_images
               else carry_over_images(graph, out_path))

    type_nodes = find_type_nodes(
        graph,
        min_isa_in=int(tn.get("min_isa_in", 2)),
        min_isa_in_ratio=float(tn.get("min_isa_in_ratio", 0.90)),
        max_out_degree=int(tn.get("max_out_degree", 2)),
        protect_attributed=bool(tn.get("protect_attributed", True)),
    )
    fold = fold_type_nodes(graph, type_nodes)
    ann = annotate(
        graph,
        relation_class=c.get("relation_class", {}) or {},
        hub_degree=int(c.get("hub_degree", 40)),
    )
    after = graph.stats()

    if carried["restored"] or carried["orphaned"]:
        print(f"Images      : carried over {carried['restored']} from the previous "
              f"{out_path.name}" + (f", {carried['orphaned']} orphaned"
                                    if carried["orphaned"] else ""))
    print(f"Duplicates  : merged {dup['merged_away']} spellings into "
          f"{dup['groups']} entities")
    for a, b in dup["examples"][:4]:
        print(f"  {a!r} -> {b!r}")
    print(f"Type nodes  : detected {fold['type_nodes_detected']}, "
          f"removed {fold['type_nodes_removed']}, "
          f"folded {fold['isa_triples_folded']} is_a triples into "
          f"{fold['entities_typed']} entities")
    print(f"  e.g. {', '.join(fold['removed_names'][:12])} ...")
    print(f"Relations   : {ann['relations_core']} core / "
          f"{ann['relations_taxonomic']} taxonomic / {ann['relations_weak']} weak "
          f"-> {ann['core_triples']} core triples usable as chain interiors")
    print(f"Hubs (>={graph.meta['hub_degree']}) : {len(ann['hubs'])} — "
          f"{', '.join(ann['hubs'][:8])}")
    print()
    print(f"{'metric':22} {'before':>8} {'after':>8}")
    for k in before:
        print(f"{k:22} {before[k]:>8} {after[k]:>8}")

    if args.dry_run:
        print("\n--dry-run: nothing written.")
        return

    graph.meta["cleaned_from"] = cfg.graph_path.name
    graph.save(out_path)
    print(f"\nWrote {out_path.relative_to(cfg.base_dir)}  "
          f"({cfg.graph_path.name} left untouched)")


if __name__ == "__main__":
    main()
