"""Freeze a dev/test split over the benchmark.

Tuning and reporting must not touch the same items. During development of the wiki-walk
retriever, four prompt variants measured .275/.275/.225/.375 on one 40-item sample with
no change to the method itself -- at that size the standard error on accuracy is about
.07, so those numbers are barely distinguishable, and picking the best of them is
selection on noise. Reporting that maximum as the method's accuracy overstates it by
roughly the amount of the spread.

The split is stratified by question type so the small dev set still exercises every
template, and keyed on the content-addressed qid so it survives regenerating the
benchmark: an item that persists across a rebuild stays on the side it started on, and
only genuinely new items are assigned.

A reserve holds whatever is left over. Dev sets wear out -- once enough decisions have
been made against 100 items, those items have been fitted too -- and the reserve is
where a fresh one comes from without ever touching test.

Run:  python bench/split.py            # writes bench/split.json
      python bench/split.py --report   # show the current split, change nothing
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from washi_kg.config import load_config  # noqa: E402


def stratified(items: list[dict], n: int, rng: random.Random) -> list[str]:
    """Take n qids, spread across types in proportion to their share."""
    by_type: dict[str, list[dict]] = collections.defaultdict(list)
    for it in items:
        by_type[it["type"]].append(it)
    for v in by_type.values():
        rng.shuffle(v)

    total = len(items)
    picked: list[str] = []
    for t, group in sorted(by_type.items()):
        take = max(1, round(n * len(group) / total))
        picked.extend(i["qid"] for i in group[:take])
    rng.shuffle(picked)
    return picked[:n]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dev", type=int, default=87)
    ap.add_argument("--test", type=int, default=0,
                    help="0 means everything that is not dev, which is the current "
                         "arrangement -- the reserve was dissolved into test")
    ap.add_argument("--seed", type=int, default=20260818)
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    bench = cfg.base_dir / "bench"
    items = json.loads((bench / "washi_qa.json").read_text())
    split_path = bench / "split.json"

    prior = json.loads(split_path.read_text()) if split_path.exists() else {}
    if args.report and prior:
        for name in ("dev", "test"):
            ids = set(prior.get(name, []))
            sub = [i for i in items if i["qid"] in ids]
            by_t = collections.Counter(i["type"] for i in sub)
            mm = sum(1 for i in sub if i.get("multimodal"))
            print(f"{name:8} n={len(sub):5d}  mm={mm / max(len(sub),1):.1%}  "
                  + ", ".join(f"{k}={v}" for k, v in sorted(by_t.items())))
        return

    rng = random.Random(args.seed)
    known = {q for name in ("dev", "test") for q in prior.get(name, [])}
    fresh = [i for i in items if i["qid"] not in known]

    # existing assignments are never revisited: an item that has already informed a
    # decision cannot be promoted into test later
    alive = {i["qid"] for i in items}
    dev = [q for q in prior.get("dev", []) if q in alive]
    test = [q for q in prior.get("test", []) if q in alive]

    need_dev = max(0, args.dev - len(dev))
    if need_dev:
        take = stratified(fresh, need_dev, rng)
        dev += take
        fresh = [i for i in fresh if i["qid"] not in set(take)]

    # There is no reserve any more: what is not dev is test. A separate pool made sense
    # while it was the source of a fresh dev set, but it became the ablation split and
    # then went unused, holding back 746 questions from the numbers the paper reports.
    if args.test:
        take = stratified(fresh, max(0, args.test - len(test)), rng)
        test += take
    else:
        test += [i["qid"] for i in fresh]

    split_path.write_text(json.dumps(
        {"seed": args.seed, "dev": dev, "test": test},
        ensure_ascii=False, indent=2))

    for name, ids in (("dev", dev), ("test", test)):
        sub = [i for i in items if i["qid"] in set(ids)]
        by_t = collections.Counter(i["type"] for i in sub)
        mm = sum(1 for i in sub if i.get("multimodal"))
        print(f"{name:8} n={len(sub):5d}  mm={mm / max(len(sub),1):.1%}  "
              + ", ".join(f"{k}={v}" for k, v in sorted(by_t.items())))
    print(f"\nWrote {split_path.relative_to(cfg.base_dir)}")


if __name__ == "__main__":
    main()
