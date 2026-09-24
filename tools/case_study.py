"""Pick case-study items from the test results, and print them with everything needed
to write the section: the question, the computed gold evidence, what each method
actually retrieved, and what each answered.

Selection is mechanical, so the examples are not chosen to flatter: a case is eligible
only when WikiWalk answered correctly and the strongest retrieval baseline did not, and
the printed evidence is whatever the run stored, not a reconstruction.

    python tools/case_study.py --tag gemma31 --type mm_chain
"""
from __future__ import annotations

import argparse
import json
import pathlib

BASE = pathlib.Path(__file__).resolve().parents[1]
R = BASE / "results"


def rows(method: str, tag: str) -> dict:
    p = R / f"{method}.test.{tag}.json"
    return {r["qid"]: r for r in json.loads(p.read_text())["rows"]} if p.exists() else {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="gemma31")
    ap.add_argument("--type", default=None, help="restrict to one question type")
    ap.add_argument("--n", type=int, default=3)
    a = ap.parse_args()

    qa = {q["qid"]: q for q in json.loads((BASE / "bench" / "washi_qa.json").read_text())}
    ww = rows("wikiwalk", a.tag)
    if not ww:
        raise SystemExit(f"no wikiwalk results for tag {a.tag}")
    others = {m: rows(m, a.tag) for m in ("graph", "fmt", "direct")}

    picks = []
    for qid, r in ww.items():
        if not r["correct"] or (a.type and r["type"] != a.type):
            continue
        beaten = [m for m, d in others.items() if d.get(qid) and not d[qid]["correct"]]
        if len(beaten) < 2:
            continue
        picks.append((r["recall"], qid, r, beaten))
    picks.sort(reverse=True, key=lambda x: x[0])

    print(f"{len(picks)} eligible cases"
          + (f" of type {a.type}" if a.type else "") + f" for {a.tag}\n")
    for _, qid, r, beaten in picks[: a.n]:
        it = qa[qid]
        print("=" * 78)
        print(f"[{it['type']}, {it.get('hops')} hops]  {it['question']}")
        print(f"\nGold answer: {it['answer']}")
        print("Gold evidence:")
        for t in it["gold_triples"]:
            print(f"   {t[0]} --{t[1]}--> {t[2]}")
        if it.get("images"):
            for im in it["images"]:
                print(f"   [image of {im.get('entity')}] {im.get('caption') or ''}")
        print(f"\nWikiWalk retrieved (recall {r['recall']:.2f}, "
              f"{r['n_calls']} calls, {r['seconds']:.0f}s):")
        for t in r["triples"]:
            print(f"   {t[0]} --{t[1]}--> {t[2]}")
        print(f"WikiWalk answered: {r['answer']}")
        for m in beaten:
            o = others[m][qid]
            print(f"\n{m} (recall {o['recall']:.2f}) answered: {(o['answer'] or '')[:300]}")
        print()


if __name__ == "__main__":
    main()
