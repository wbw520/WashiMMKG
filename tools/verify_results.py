"""Recompute every result file's summary from its own rows and report disagreements.

A result file is written once, at the end of a run, so its summary should always match its
rows. It stopped being obvious that this holds: `abl-no-verify.test.json` was observed with
two different summaries minutes apart while only one run of that condition appears in any
log, and precision -- which depends on retrieval alone and not on grading -- differed
between them, so the two readings cannot both describe the same run.

This does not explain that, but it catches its consequences and anything like it: a file
half-written by one process while another reads it, a summary left over from a previous
write, a row set that does not match the split being reported.

    python tools/verify_results.py
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys

BASE = pathlib.Path(__file__).resolve().parents[1]
R = BASE / "results"
TOL = 5e-5


def check(p: pathlib.Path, split: set[str]) -> list[str]:
    bad = []
    try:
        d = json.loads(p.read_text())
    except Exception as exc:
        return [f"unreadable: {type(exc).__name__}"]
    rows, s = d.get("rows"), d.get("summary")
    if not rows or not s:
        return ["no rows or no summary"]

    if s.get("n") != len(rows):
        bad.append(f"summary says n={s.get('n')}, file holds {len(rows)} rows")
    ids = [r["qid"] for r in rows]
    if len(set(ids)) != len(ids):
        bad.append(f"{len(ids) - len(set(ids))} duplicated qid(s)")
    if split and set(ids) != split:
        extra, missing = len(set(ids) - split), len(split - set(ids))
        bad.append(f"rows are not the test split (+{extra} / -{missing})")

    for key, val in (("acc", sum(r["correct"] for r in rows) / len(rows)),
                     ("precision", statistics.mean(r["precision"] for r in rows)),
                     ("recall", statistics.mean(r["recall"] for r in rows))):
        if key in s and abs(s[key] - val) > TOL:
            bad.append(f"{key}: summary {s[key]:.4f}, rows give {val:.4f}")
    return bad


def main() -> None:
    # Two budgets sit in results/ while K=8 is being measured, so the check has to be
    # aimable at one of them: a K=8 sweep in progress is full of files that are correctly
    # incomplete, and they would drown the report on the finished set.
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="",
                    help="check only files matching this glob under results/")
    args = ap.parse_args()
    try:
        split = set(json.loads((BASE / "bench" / "split.json").read_text())["test"])
    except Exception:
        split = set()
    if args.glob:
        files = sorted(R.glob(args.glob))
    else:
        files = sorted(R.glob("*.test.*.json")) + sorted(R.glob("abl-*.test.json"))
    n_bad = 0
    for p in files:
        problems = check(p, split)
        if problems:
            n_bad += 1
            print(f"{p.name}")
            for b in problems:
                print(f"    {b}")
    print(f"\n{len(files)} file(s) checked, {n_bad} with a problem")
    sys.exit(1 if n_bad else 0)


if __name__ == "__main__":
    main()
