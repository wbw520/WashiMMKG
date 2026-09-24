"""Find items whose wording gives away what the question is supposed to make you find.

`validate.py` already rejects an item whose question contains its *answer*. It does not
check the waypoints. A chain question is meant to be walked, so naming a node in the
middle of the path shortens the walk without changing the gold: the item still looks
like a five-hop question and is scored as one. Measured over the built benchmark, 31% of
chain items name at least one intermediate node, and among the four- and five-hop items
it is more than half -- the verbalizer, asked to describe a long path in one sentence,
reaches for the waypoints as landmarks.

The multimodal `grounded` mode has the same shape for a different reason: the question
names the entity its photograph is supposed to reveal, so mention resolution anchors the
walk and the picture is decorative. Removing image roots costs that mode .039, against
.545 for `attribute` -- it is a single-hop question with a picture attached, not a
multimodal one.

Run:  python bench/clean.py            # report only
      python bench/clean.py --write    # drop them and rewrite the benchmark
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

BENCH = Path(__file__).resolve().parent


def names(text: str, entity: str) -> bool:
    """Whole-word occurrence, so `su` does not match inside `surface`."""
    return re.search(rf"\b{re.escape(entity.lower())}\b", text.lower()) is not None


def path_of(item: dict, spec: dict) -> list[list]:
    return spec.get("path") or item.get("gold_triples") or []


def verdict(item: dict, spec: dict) -> tuple[str, list[str]] | None:
    """Why this item should go, or None to keep it."""
    q, t = item["question"], item["type"]

    if t == "multimodal" and spec.get("mode") == "grounded":
        return "grounded: the question names the entity the photograph should reveal", []

    if t in ("chain", "mm_chain"):
        p = path_of(item, spec)
        if len(p) < 2:
            return None
        start = p[0][0]
        # the start may be named -- it is where the reader is told to begin. Everything
        # downstream of it is what the walk is for.
        waypoints = [e[2] for e in p[:-1]]
        named = [w for w in waypoints if w != start and names(q, w)]
        if named:
            return "names an intermediate node of the path", named

    if t == "multimodal" and spec.get("mode") in ("identify", "attribute"):
        subj = spec.get("subject")
        if isinstance(subj, str) and names(q, subj):
            return "names the subject the photograph should reveal", [subj]

    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    qa = json.loads((BENCH / "washi_qa.json").read_text())
    items = qa["items"] if isinstance(qa, dict) else qa
    spec = {x["qid"]: (x.get("spec") or {})
            for x in json.loads((BENCH / "washi_qa_verbalized.json").read_text())}

    drop, why = {}, collections.Counter()
    for it in items:
        v = verdict(it, spec.get(it["qid"], {}))
        if v:
            drop[it["qid"]] = v
            why[v[0]] += 1

    kept = [i for i in items if i["qid"] not in drop]
    print(f"{len(items)} items -> dropping {len(drop)} -> {len(kept)} kept\n")
    for reason, n in why.most_common():
        print(f"  {n:>4}  {reason}")

    print("\nby type:")
    before = collections.Counter(i["type"] for i in items)
    after = collections.Counter(i["type"] for i in kept)
    for t in sorted(before, key=lambda x: -before[x]):
        print(f"  {t:14} {before[t]:>4} -> {after.get(t,0):>4}  ({after.get(t,0)-before[t]:+d})")

    if args.write:
        (BENCH / "dropped_qids.json").write_text(json.dumps(
            {q: {"reason": r, "named": n} for q, (r, n) in drop.items()},
            ensure_ascii=False, indent=2))
        out = {**qa, "items": kept} if isinstance(qa, dict) else kept
        (BENCH / "washi_qa.json").write_text(json.dumps(out, ensure_ascii=False, indent=2))
        print(f"\nwrote {len(kept)} items; the dropped qids and their reasons are in "
              f"bench/dropped_qids.json")


if __name__ == "__main__":
    main()
