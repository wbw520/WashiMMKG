"""Turn results/*.json into the LaTeX rows the paper needs.

Reads only what was actually run. A backbone with no result file is reported as missing
rather than filled in, because a table that cannot tell a measurement from a placeholder
is worse than an incomplete one.

    python tools/make_tables.py            # all three tables
    python tools/make_tables.py --which main
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics

R = pathlib.Path(__file__).resolve().parents[1] / "results"

BACKBONES = [
    ("qwen9",   "Qwen3.5-9B~\\cite{qwen3vl2025}"),
    ("gemma12", "Gemma-4-12B~\\cite{google2026gemma4}"),
    ("qwen27",  "Qwen3.5-27B~\\cite{qwen3vl2025}"),
    ("gemma31", "Gemma-4-31B~\\cite{google2026gemma4}"),
]
METHODS = ["direct", "know", "know+", "fmt", "graph", "wikiwalk"]


def load(method: str, tag: str) -> dict | None:
    p = R / f"{method}.test.{tag}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())["summary"]


def fmt(x: float | None) -> str:
    return "---" if x is None else f".{round(x * 10000):04d}"[:5]


def main_table() -> None:
    print("% --- tab:main_results rows " + "-" * 40)
    for tag, label in BACKBONES:
        s = {m: load(m, tag) for m in METHODS}
        if not any(s.values()):
            print(f"% {label}: not run")
            continue
        w = s["wikiwalk"]
        cells = [fmt(s[m]["acc"] if s[m] else None) for m in METHODS[:5]]
        print(f"{label}\n& " + "\n& ".join(cells)
              + f"\n& {fmt(w['acc'] if w else None)}"
              + f" & {fmt(w['precision'] if w else None)}"
              + f" & {fmt(w['recall'] if w else None)} \\\\\n")
    # the two retrieval baselines report precision/recall in the caption, not the body
    for m in ("fmt", "graph"):
        vals = [load(m, t) for t, _ in BACKBONES]
        vals = [v for v in vals if v]
        if vals:
            print(f"% {m}: Pre.={statistics.mean(v['precision'] for v in vals):.3f} "
                  f"Re.={statistics.mean(v['recall'] for v in vals):.3f} "
                  f"(mean over {len(vals)} backbones)")


def efficiency_table() -> None:
    print("\n% --- tab:efficient rows " + "-" * 40)
    for tag, label in BACKBONES:
        w = load("wikiwalk", tag)
        d = load("direct", tag)
        if not w:
            print(f"% {label}: not run")
            continue
        print(f"{label.split('~')[0]} & ${w['seconds_mean']:.2f} \\pm {w['seconds_sd']:.2f}$ "
              f"& ${w['calls_mean']:.2f} \\pm {w['calls_sd']:.2f}$ "
              f"& ${d['seconds_mean']:.2f} \\pm {d['seconds_sd']:.2f}$ \\\\"
              if d else
              f"{label.split('~')[0]} & ${w['seconds_mean']:.2f} \\pm {w['seconds_sd']:.2f}$ "
              f"& ${w['calls_mean']:.2f} \\pm {w['calls_sd']:.2f}$ & --- \\\\")


ABLATIONS = [
    ("full",             "Full WikiWalk"),
    ("text-only-kg",     "Text-only KG (no $\\mathcal{A}_{\\text{img}}$)"),
    ("no-image-roots",   "\\quad no image roots"),
    ("no-mention-roots", "\\quad no mention-anchored roots"),
    ("no-verify",        "\\quad no chain verification"),
    ("verify-question",  "\\quad verify against the whole question"),
    ("merge-cite-follow","\\quad cite and follow merged"),
    ("drain-best-chain", "\\quad best chain drained, not cycled"),
    ("subq1",            "\\quad 1 sub-query"),
    ("subq3",            "\\quad 3 sub-queries"),
    ("budget3",          "\\quad $K=3$"),
    ("budget8",          "\\quad $K=8$"),
    ("answer-images1",   "\\quad 1 retrieved image shown"),
    ("answer-images6",   "\\quad 6 retrieved images shown"),
]


def ablation_table() -> None:
    print("\n% --- ablation rows (reserve split) " + "-" * 30)
    base = None
    for tag, label in ABLATIONS:
        p = R / f"abl-{tag}.reserve.json"
        if not p.exists():
            print(f"% {label}: not run")
            continue
        s = json.loads(p.read_text())["summary"]
        if tag == "full":
            base = s["acc"]
        delta = "" if base is None or tag == "full" else f" & {s['acc'] - base:+.4f}"
        print(f"{label} & {fmt(s['acc'])} & {fmt(s['precision'])} & {fmt(s['recall'])}"
              f"{delta} \\\\")
        if tag in ("full", "text-only-kg"):
            bt = s["by_type"]
            print("%   by type: " + ", ".join(f"{k}={v['acc']:.3f}" for k, v in bt.items()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", default="all",
                    choices=["all", "main", "efficiency", "ablation"])
    a = ap.parse_args()
    if a.which in ("all", "main"):
        main_table()
    if a.which in ("all", "efficiency"):
        efficiency_table()
    if a.which in ("all", "ablation"):
        ablation_table()
