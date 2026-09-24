"""Instantiate the QA templates against the graph — stage 1 of benchmark construction.

Produces `bench/washi_qa_items.json`: items with a computed gold answer and exact
`gold_triples`, but no natural language yet. `verbalize.py` adds the question and
answer text; `validate.py` then filters. Splitting it this way means the expensive
LLM stage never runs on an item that structural checks already reject, and the gold
never depends on what a model chose to say.

Sampling is rejection-based per type until the quota fills, deduplicated on a
signature so the same underlying fact cannot yield two items. If a type cannot fill
its quota — the graph simply may not contain 140 distinct negative-set questions —
the shortfall is reported rather than padded with near-duplicates.

Run (from code/, conda `washi`):
    python bench/build_qa.py
    python bench/build_qa.py --total 200      # small set for a pipeline smoke test
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench.templates import TEMPLATES, GraphIndex, QAItem  # noqa: E402
from washi_kg.config import load_config  # noqa: E402

# How many draws to allow per accepted item before declaring a type exhausted. Deep
# chains need many more attempts than shallow ones -- a 5-hop walk succeeds in well under
# 1% of draws -- so the budget is generous rather than tuned per template.
ATTEMPT_FACTOR = 400

# A set-valued answer (intersection, negative) stops being a question and becomes an
# inventory past this size: it reads badly as open-ended text and an LLM judge scores
# a fifteen-item list unreliably.
MAX_ANSWER_ATOMS = 8


def signature(item: QAItem) -> tuple:
    """Identity of an item, so paraphrases of one question collapse to one.

    The mode belongs in the identity, not just in the spec. Two multimodal items can
    rest on exactly the same fact and still be different questions: `attribute` hides
    the subject so it must be recognised from the photograph, while `grounded` names it
    and asks the same property outright. Leaving the mode out gave them one content
    hash, so they overwrote each other on write and 58 items vanished between the build
    report and the file.
    """
    return (
        item.type,
        item.spec.get("mode"),
        tuple(sorted(item.answer_atoms)),
        tuple(sorted(tuple(t) for t in item.gold_triples)),
    )


def qid_of(item: QAItem) -> str:
    """Content-addressed id, so an item keeps its name across regenerations.

    Positional ids (`washi-00001` by shuffle order) change for every item whenever the
    set is regrown, which silently invalidates the whole verbalisation cache and the
    dev/test split along with it. Hashing the computed gold means an item that survives
    a rebuild is recognisably the same item, and only genuinely new ones are re-verbalised.
    """
    raw = json.dumps(signature(item), ensure_ascii=False, sort_keys=True)
    return "washi-" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10]


def sample_type(
    g: GraphIndex, name: str, quota: int, rng: random.Random
) -> tuple[list[QAItem], int]:
    fn = TEMPLATES[name]
    seen: set[tuple] = set()
    out: list[QAItem] = []
    attempts = 0
    budget = quota * ATTEMPT_FACTOR
    while len(out) < quota and attempts < budget:
        attempts += 1
        try:
            item = fn(g, rng)
        except Exception:
            continue
        if item is None or not item.answer_atoms or not item.gold_triples:
            continue
        if len(item.answer_atoms) > MAX_ANSWER_ATOMS:
            continue
        sig = signature(item)
        if sig in seen:
            continue
        seen.add(sig)
        out.append(item)
    return out, attempts


def enforce_mm_ratio(items: list[QAItem], ratio: float) -> tuple[list[QAItem], str]:
    """Drop text-only items until the multimodal share reaches `ratio`.

    Trimming text-only items is preferred to inventing multimodal ones: every item
    kept is still a real, computed question, and the visual subset is not padded
    with the same few illustrated entities over and over.
    """
    mm = [i for i in items if i.multimodal]
    txt = [i for i in items if not i.multimodal]
    if not items:
        return items, "empty"
    have = len(mm) / len(items)
    if have >= ratio:
        return items, f"{have:.1%} multimodal, floor {ratio:.0%} already met"
    # |mm| / (|mm| + keep) = ratio  ->  keep = |mm|(1-ratio)/ratio
    keep = int(len(mm) * (1 - ratio) / ratio)
    if keep >= len(txt):
        return items, f"{have:.1%} multimodal — cannot reach {ratio:.0%}"
    dropped = len(txt) - keep
    return mm + txt[:keep], (
        f"{have:.1%} -> {ratio:.0%} multimodal by dropping {dropped} text-only items"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--graph", default=None, help="default: the cleaned graph.v2.json")
    ap.add_argument("--total", type=int, default=0, help="scale all quotas to this total")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    bench_cfg = cfg.get("benchmark", default={}) or {}
    graph_path = Path(args.graph) if args.graph else cfg.output_dir / cfg.get(
        "paths", "clean_graph_file", default="graph.v2.json"
    )
    out_path = Path(args.out) if args.out else cfg.base_dir / "bench" / "washi_qa_items.json"

    quotas: dict = dict(bench_cfg.get("quotas") or {})
    if args.total:
        scale = args.total / max(sum(quotas.values()), 1)
        quotas = {k: max(1, round(v * scale)) for k, v in quotas.items()}

    seed = int(bench_cfg.get("seed", 0))
    rng = random.Random(seed)
    g = GraphIndex(graph_path)
    print(f"Graph: {len(g.E)} entities, {len(g.T)} triples, "
          f"{len(g.mm)} imaged, {len(g.hubs)} hubs\n")

    items: list[QAItem] = []
    print(f"{'type':16} {'quota':>6} {'got':>6} {'draws':>7}  note")
    print("-" * 62)
    for name in TEMPLATES:
        quota = int(quotas.get(name, 0))
        if quota <= 0:
            continue
        # One stream per type, keyed by its name. A single shared stream means that
        # changing how one type samples shifts every draw after it, so tightening the
        # path templates would have re-rolled intersection, comparative and the rest --
        # discarding 1,373 items that were never in question, along with their
        # verbalisations and every result already measured against them.
        # str.__hash__ is salted per process, so the stream is keyed by a digest instead
        # -- otherwise the benchmark would differ between two runs of the same command.
        stream = int(hashlib.sha1(f"{seed}:{name}".encode()).hexdigest()[:8], 16)
        got, attempts = sample_type(g, name, quota, random.Random(stream))
        note = "" if len(got) >= quota else "EXHAUSTED — graph has no more distinct items"
        print(f"{name:16} {quota:6d} {len(got):6d} {attempts:7d}  {note}")
        items.extend(got)

    rng.shuffle(items)
    items, mm_note = enforce_mm_ratio(items, float(bench_cfg.get("mm_ratio", 0.0)))
    for it in items:
        it.qid = qid_of(it)

    by_type = collections.Counter(i.type for i in items)
    n_mm = sum(1 for i in items if i.multimodal)
    hops = [i.hops for i in items]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps([i.to_json() for i in items],
                                   ensure_ascii=False, indent=2))

    print("-" * 62)
    print(f"mm floor: {mm_note}")
    print(f"\nTotal {len(items)} items | multimodal {n_mm} ({n_mm / max(len(items),1):.1%}) "
          f"| avg hops {sum(hops) / max(len(hops),1):.2f}")
    print("by type:", ", ".join(f"{k}={v}" for k, v in by_type.most_common()))
    print(f"\nWrote {out_path.relative_to(cfg.base_dir)}")


if __name__ == "__main__":
    main()
