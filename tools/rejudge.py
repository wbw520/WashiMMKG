"""Re-grade finished runs with one fixed judge, without asking any model to answer again.

Every result file stores what the backbone actually said, so grading can be redone from
disk: one judge call per question instead of the nineteen a walk costs. A full re-run of
Table 1 is tens of GPU-hours; re-grading it is minutes.

The reason to redo it at all is that each row was graded by the backbone that produced it.
A model recognises its own phrasing, and a weak one misjudges as badly as it answers --
Qwen3.5-9B read .8897 on the condition where the answer is handed to it, and part of that
gap is its own grading rather than its own answering. Rows graded by different readers are
not comparable, however carefully the prompt is written.

The judge is Gemma-4-31B, which also wrote the questions and appears as a backbone. That
is not neutral and the paper should say so: its own row is the one to read with most
suspicion. It is chosen because it is the strongest model available locally, and a
consistent ruler with a known bias beats five different rulers.

    python tools/rejudge.py --url http://127.0.0.1:8001/v1 --model gemma-4-31b-it
    python tools/rejudge.py --url ... --model ... --only 'wikiwalk.test.*'
"""

from __future__ import annotations

import argparse
import copy
import fnmatch
import json
import os
import pathlib
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from washi_kg.config import load_config  # noqa: E402
from washi_kg.llm import LLMClient  # noqa: E402
from wikirag.evaluate import judge  # noqa: E402

BASE = pathlib.Path(__file__).resolve().parents[1]
# same switch as fill_paper.py and check_paper_numbers.py, so a re-measured set living in
# its own directory can be judged, filled and checked without moving files around
R = BASE / os.environ.get("WASHI_RESULTS", "results")


def gold_of(qa: list[dict]) -> dict[str, dict]:
    return {i["qid"]: i for i in qa}


def rescore(path: pathlib.Path, gold: dict, llm, workers: int) -> tuple[float, float, int]:
    doc = json.loads(path.read_text())
    rows = doc["rows"]
    before = sum(r["correct"] for r in rows) / max(len(rows), 1)

    def one(r: dict) -> dict:
        it = gold.get(r["qid"])
        # a row whose question has left the benchmark is dropped rather than re-graded
        if it is None:
            return None
        if r.get("error"):
            r["correct"] = False
            return r
        r["correct"] = judge(llm, it["question"], it["answer"],
                             it["answer_atoms"], r.get("answer") or "")
        return r

    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = [r for r in pool.map(one, rows) if r is not None]

    doc["rows"] = rows
    s = doc["summary"]
    s["n"] = len(rows)
    s["acc"] = sum(r["correct"] for r in rows) / max(len(rows), 1)
    s["precision"] = statistics.mean(r["precision"] for r in rows) if rows else 0.0
    s["recall"] = statistics.mean(r["recall"] for r in rows) if rows else 0.0
    s["recall_entailed"] = (statistics.mean(r["recall_entailed"] for r in rows)
                            if rows else 0.0)
    s["judged_by"] = llm.model
    s["by_type"] = {}
    for t in sorted({r["type"] for r in rows}):
        sub = [r for r in rows if r["type"] == t]
        s["by_type"][t] = {
            "n": len(sub),
            "acc": sum(r["correct"] for r in sub) / len(sub),
            "recall": statistics.mean(r["recall"] for r in sub),
        }
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=2))
    return before, s["acc"], len(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--only", default="*.test.*.json",
                    help="glob over results/ filenames")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    os.environ.setdefault("OPENAI_API_KEY", "EMPTY")
    cfg = copy.deepcopy(load_config())
    cfg.raw["llm"].update(provider="openai_compatible",
                          base_url=args.url, model=args.model)
    llm = LLMClient(cfg)

    qa = json.loads((BASE / "bench" / "washi_qa.json").read_text())
    gold = gold_of(qa)

    targets = sorted(p for p in R.glob("*.json") if fnmatch.fnmatch(p.name, args.only))
    if not targets:
        sys.exit(f"nothing in results/ matches {args.only!r}")
    print(f"re-grading {len(targets)} file(s) with {args.model}\n")
    if args.dry_run:
        for p in targets:
            print(f"  {p.name}")
        return

    print(f"{'file':38} {'was':>7} {'now':>7} {'delta':>8}  n")
    for p in targets:
        try:
            was, now, n = rescore(p, gold, llm, args.workers)
        except Exception as exc:
            print(f"  {p.name:36} FAILED: {type(exc).__name__}: {exc}")
            continue
        print(f"{p.name:38} {was:7.4f} {now:7.4f} {now - was:+8.4f}  {n}")


if __name__ == "__main__":
    main()
