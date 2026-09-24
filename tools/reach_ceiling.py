#!/usr/bin/env python
"""How much of the gold evidence the walk could reach even if it never chose wrong.

Recall is reported against the gold triples of each question, but nothing says how many
of those a bounded walk is able to see at all. This separates three limits that a single
recall number blurs together:

  reachable   -- gold triples incident to a page within `depth` hops of the question's
                 own entities, ignoring every other bound. The graph's own limit.
  rendered    -- the same, but counting only triples that survive page rendering.
                 A page shows at most `max_links_per_relation` links per relation and
                 `max_total_links` in all, so an edge can be reachable and still never
                 appear in front of the model.
  paged       -- rendered, and within the first `max_pages` pages a breadth-first walk
                 would open. The search budget's limit.

Roots are the entities named in the question, resolved through the graph's own alias
index. That is deliberately generous: it is the root set a perfect mention resolver would
produce, so what remains unreachable is not the resolver's fault.

    python tools/reach_ceiling.py --split test
"""
from __future__ import annotations

import argparse
import collections
import functools
import json
import pathlib
import re
import sys

BASE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

from wikirag.page import WikiGraph      # noqa: E402


def question_roots(g: WikiGraph, question: str) -> list[str]:
    """Entities named in the question, longest alias first so 'Tosa washi' beats 'washi'."""
    q = question.lower()
    hits: list[tuple[int, str]] = []
    for alias, canon in g.alias_index.items():
        if len(alias) < 3:
            continue
        if re.search(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", q):
            hits.append((len(alias), canon))
    hits.sort(reverse=True)
    out: list[str] = []
    for _, canon in hits:
        if canon not in out:
            out.append(canon)
    return out


# A rendered page depends only on the entity, and depth-3 neighbourhoods overlap
# heavily across 2,007 questions. Without this the same few thousand pages are
# rebuilt millions of times and the analysis never finishes.
@functools.lru_cache(maxsize=None)
def rendered_edges(g: WikiGraph, name: str) -> frozenset:
    """The triples a rendered page actually puts in front of the model."""
    page = g.page(name)
    if page is None:
        return frozenset()
    out = set()
    for l in page.links:
        out.add((l.target, l.relation, name) if l.incoming else (name, l.relation, l.target))
    return frozenset(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa", default="bench/washi_qa.json")
    ap.add_argument("--split", default="test")
    ap.add_argument("--graph", default="output/graph.v2.json")
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--max-pages", type=int, default=24)
    ap.add_argument("--skip-type", action="append", default=[],
                    help="exclude a question type from the totals")
    args = ap.parse_args()

    g = WikiGraph(BASE / args.graph)
    items = json.loads((BASE / args.qa).read_text())
    if not isinstance(items, list):
        items = items.get("items") or items.get("questions")
    keep = set(json.loads((BASE / "bench" / "split.json").read_text())[args.split])
    items = [it for it in items if it["qid"] in keep]

    tot = collections.Counter()
    per_type: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    no_root = 0

    for it in items:
        gold = {(h, r, u) for h, r, u in it["gold_triples"]}
        if not gold:
            continue
        roots = question_roots(g, it["question"])
        if not roots:
            no_root += 1

        # breadth-first over pages, recording the order they would be opened in
        seen, order, frontier = set(roots), list(roots), list(roots)
        for _ in range(args.depth):
            nxt = []
            for name in frontier:
                for _r, tgt in g.out.get(name, []) + g.inn.get(name, []):
                    if tgt not in seen:
                        seen.add(tgt)
                        nxt.append(tgt)
                        order.append(tgt)
            frontier = nxt
            if not frontier:
                break

        reach = {(h, r, u) for (h, r, u) in gold if h in seen or u in seen}
        rend: set[tuple[str, str, str]] = set()
        for name in order:
            rend |= rendered_edges(g, name)
        paged: set[tuple[str, str, str]] = set()
        for name in order[: args.max_pages]:
            paged |= rendered_edges(g, name)

        t = it["type"]
        for key, hit in (("reachable", reach),
                         ("rendered", gold & rend),
                         ("paged", gold & paged)):
            tot[key] += len(hit)
            per_type[t][key] += len(hit)
        tot["gold"] += len(gold)
        per_type[t]["gold"] += len(gold)
        per_type[t]["n"] += 1

    print(f"{len(items)} questions, {tot['gold']} gold triples, "
          f"{no_root} with no entity named in the question\n")
    print(f"{'type':16}{'n':>6}{'gold':>7}{'reachable':>11}{'rendered':>10}{'paged':>9}")
    for t in sorted(per_type):
        c = per_type[t]
        print(f"  {t:14}{c['n']:6}{c['gold']:7}"
              f"{c['reachable'] / c['gold']:11.3f}{c['rendered'] / c['gold']:10.3f}"
              f"{c['paged'] / c['gold']:9.3f}")
    print(f"  {'ALL':14}{len(items):6}{tot['gold']:7}"
          f"{tot['reachable'] / tot['gold']:11.3f}{tot['rendered'] / tot['gold']:10.3f}"
          f"{tot['paged'] / tot['gold']:9.3f}")
    if args.skip_type:
        # Roots here are the entities *named* in the question. Visual-terminal items are
        # not entered that way -- their query is a picture and the walker roots them
        # through image similarity -- so a text-mention root set understates them and
        # says nothing about what the walk can actually reach for them.
        sub = collections.Counter()
        n_sub = 0
        for t, c in per_type.items():
            if t in args.skip_type:
                continue
            n_sub += c["n"]
            for k in ("gold", "reachable", "rendered", "paged"):
                sub[k] += c[k]
        print(f"  {'ALL less ' + ','.join(args.skip_type):14}"[:16]
              + f"{n_sub:6}{sub['gold']:7}"
              f"{sub['reachable'] / sub['gold']:11.3f}{sub['rendered'] / sub['gold']:10.3f}"
              f"{sub['paged'] / sub['gold']:9.3f}")


if __name__ == "__main__":
    main()
