"""Write the measured numbers into sn-article.tex, in place.

`make_tables.py` prints LaTeX rows for a human to paste. That step is where the one wrong
number in this paper came from: a row filled from a progress notification instead of from
the result file, .6006 where the file said .6153. Copying by hand is the unreliable part,
so this does the copying.

Each table is located by its label, each row by the text in its first column, and only the
numeric cells are replaced -- the surrounding LaTeX, the bolding and the captions are left
exactly as they are. A row whose result file does not exist yet is left untouched rather
than blanked, so this can be run repeatedly while results are still arriving.

    python tools/fill_paper.py            # fill what is measured, report what is not
    python tools/fill_paper.py --check    # change nothing, just report disagreements
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import re
import statistics
import sys

BASE = pathlib.Path(__file__).resolve().parents[1]
R = BASE / os.environ.get("WASHI_RESULTS", "results")
TEX = BASE.parent.parent / "paper" / "sn-article.tex"

BACKBONES = [
    ("Qwen3.5-9B", "qwen9"),
    ("Gemma-4-12B", "gemma12"),
    ("Qwen3.5-27B", "qwen27"),
    ("Qwen3.8-27B", "qwen38"),
    ("Gemma-4-31B", "gemma31"),
]
METHODS = ["direct", "know", "know+", "fmt", "graph", "wikiwalk"]

ABLATIONS = [
    ("Full WikiWalk", "abl-full"),
    ("no chain verification", "abl-no-verify"),
    ("verify against the whole question", "abl-verify-question"),
    ("best chain drained", "abl-drain-best-chain"),
    ("1 picture shown to the navigator", "abl-nav1"),
    ("3 pictures shown to the navigator", "abl-nav3"),
    ("captions kept, answer withheld", "abl-keep-captions"),
    ("all captions shown", "abl-all-captions"),
]
# "3 pictures and their captions" is gone: the main configuration now blanks every caption,
# so that variant is byte-for-byte the 3-picture one. It was left in the table long enough
# to be filled from a stale file -- .6562 against a Full of .6662, carrying a $+.0025$ that
# belonged to an older baseline -- which is why a row whose file is missing must be removed
# rather than left for fill_paper to skip.
# The budget, beam/depth and edge-drop sweeps are drawn by tools/make_fig_ablations.py
# straight from the same result files, so they have no rows here.


def summary(name: str) -> dict | None:
    """The summary of a cell, but only if that cell answers the questions being reported.

    Files from the previous benchmark are still on disk: they exist, they parse, their
    numbers look reasonable, and their rows are questions that have since been removed.
    Filling from one puts a number in the table that cannot be compared with the row above
    it, and nothing about the table shows it. A cell counts only when its rows are exactly
    the current test split.
    """
    p = R / f"{name}.json"
    if not p.exists():
        return None
    try:
        doc = json.loads(p.read_text())
    except Exception:
        return None
    if {r["qid"] for r in doc.get("rows", [])} != _test_split():
        return None
    return doc["summary"]


def _test_split() -> set[str]:
    global _SPLIT
    if _SPLIT is None:
        _SPLIT = set(json.loads((BASE / "bench" / "split.json").read_text())["test"])
    return _SPLIT


_SPLIT: set[str] | None = None


def paired_z(full: str, other: str) -> float:
    """z-score of the paired accuracy difference between two runs over the same items."""
    a = {r["qid"]: bool(r["correct"]) for r in json.loads((R / f"{full}.json").read_text())["rows"]}
    b = {r["qid"]: bool(r["correct"]) for r in json.loads((R / f"{other}.json").read_text())["rows"]}
    n = len(a)
    bc = sum(1 for q in a if a[q] and not b.get(q, False))    # full right, other wrong
    cb = sum(1 for q in a if not a[q] and b.get(q, False))    # full wrong, other right
    se = math.sqrt((bc + cb) - (bc - cb) ** 2 / n) / n
    return ((cb - bc) / n) / se if se else 0.0


def four(x: float) -> str:
    return f"{x:.4f}".lstrip("0")


def keep_style(old: str, new: str) -> str:
    """Put `new` where `old`'s number was, keeping any \\textbf{} around it."""
    if new.startswith("$") and new.endswith("$"):
        # The signed deltas and the mean$\\pm$sd cells carry their own math delimiters.
        # Substituting one into the digits of `$-.0684$` nests them -- `$-$-.0678$$` --
        # which is not a number the reader ever sees, it is a broken table. Replace the
        # whole math group instead. The replacement goes through a lambda because `new`
        # contains backslashes that re.sub would otherwise read as group references.
        if re.search(r"\$[^$]*\$", old):
            return re.sub(r"\$[^$]*\$", lambda _m: new, old, count=1)
        return new
    return re.sub(r"\.\d{4}", new, old, count=1) if re.search(r"\.\d{4}", old) else new


def set_row(lines: list[str], start: int, end: int, label: str,
            cells: list[str | None], where: str = "") -> str | None:
    """Replace the numeric cells of the row whose first column starts with `label`."""
    where = f" in {where}" if where else ""
    for i in range(start, end):
        head = lines[i].lstrip()
        # The ablation rows are indented with \quad, which lstrip() does not remove. Without
        # this every \quad row misses its label, set_row reports "no such row", and the table
        # silently keeps whatever numbers it already had -- which is how the whole ablation
        # table stayed on its K=5 values while the files underneath moved to K=8.
        for pad in ("\\qquad", "\\quad"):
            if head.startswith(pad):
                head = head[len(pad):].lstrip()
                break
        if not head.startswith(label):
            continue
        buf, j = lines[i], i
        while not buf.rstrip().endswith("\\\\") and j + 1 < end:
            j += 1
            buf += "\n" + lines[j]
        parts = buf.split("&")
        if len(parts) - 1 < len(cells):
            return f"{label}{where}: row has {len(parts)-1} cells, need {len(cells)}"
        for k, v in enumerate(cells, start=1):
            if v is None:
                continue
            tail = ""
            if k == len(parts) - 1:
                m = re.search(r"\s*\\\\\s*$", parts[k])
                tail = m.group(0) if m else ""
                body = parts[k][: len(parts[k]) - len(tail)] if tail else parts[k]
            else:
                body = parts[k]
            parts[k] = keep_style(body, v) + tail
        merged = "&".join(parts)
        for k, piece in enumerate(merged.split("\n")):
            lines[i + k] = piece
        return None
    return f"{label}{where}: no such row"


def table_bounds(lines: list[str], label: str) -> tuple[int, int]:
    st = next(i for i, l in enumerate(lines) if f"label{{{label}}}" in l)
    # tabular and tabular* alike: several tables use the starred form
    en = next(i for i in range(st, len(lines)) if "end{tabular" in lines[i])
    return st, en


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    lines = TEX.read_text().splitlines()
    problems: list[str] = []
    filled = missing = 0

    # ---------------------------------------------------------------- main results
    st, en = table_bounds(lines, "tab:main_results")
    for label, tag in BACKBONES:
        # Direct / Know. / Know.+ carry accuracy alone; the four retrieval methods each
        # carry Acc, Pre., Re. A backbone the agent was not run with keeps its --- cells
        # via None.
        cells: list[str | None] = []
        for m in ("direct", "know", "know+"):
            v = summary(f"{m}.test.{tag}")
            cells.append(four(v["acc"]) if v else None)
        for m in ("fmt", "graph"):
            v = summary(f"{m}.test.{tag}")
            cells.append(four(v["acc"]) if v else None)
        for name in (f"react.test.{tag}", f"wikiwalk.test.{tag}"):
            v = summary(name)
            cells += ([four(v["acc"]), four(v["precision"]), four(v["recall"])]
                      if v else [None, None, None])
        if not any(c for c in cells):
            missing += 1
            continue
        filled += sum(1 for c in cells if c)
        err = set_row(lines, st, en, label, cells, "tab:main_results")
        if err:
            problems.append(err)

    # ------------------------------------------------------------------- ablations
    st, en = table_bounds(lines, "tab:wikiwalk_ablation")
    base = summary("abl-full.test")
    for label, tag in ABLATIONS:
        s = summary(f"{tag}.test")
        if not s:
            missing += 1
            continue
        delta = None
        if tag != "abl-full" and base:
            # The ablation answers the same items as the full run, so the difference is
            # paired: its standard error comes from the discordant items alone. A dagger
            # marks a difference the benchmark cannot distinguish from zero at 1.96 SE.
            z = paired_z("abl-full.test", f"{tag}.test")
            mark = "" if abs(z) >= 1.96 else "^{\\dagger}"
            delta = f"${s['acc'] - base['acc']:+.4f}{mark}$".replace("0.", ".")
        cells = [four(s["acc"]), four(s["precision"]), four(s["recall"]), delta]
        filled += 3
        err = set_row(lines, st, en, label, cells, "tab:wikiwalk_ablation")
        if err:
            problems.append(err)

    # ------------------------------------------------------------------ efficiency
    st, en = table_bounds(lines, "tab:efficient")
    for label, tag in BACKBONES:
        w = summary(f"wikiwalk.test.{tag}")
        if not w:
            continue
        rows = json.loads((R / f"wikiwalk.test.{tag}.json").read_text())["rows"]
        if {r["qid"] for r in rows} != _test_split():
            continue
        per = [r["seconds"] / r["n_calls"] for r in rows if r.get("n_calls")]
        cells = [f"${w['seconds_mean']:.2f} \\pm {w['seconds_sd']:.2f}$",
                 f"${w['calls_mean']:.2f} \\pm {w['calls_sd']:.2f}$",
                 f"${statistics.mean(per):.2f} \\pm {statistics.pstdev(per):.2f}$"]
        err = set_row(lines, st, en, label, cells, "tab:efficient")
        if err:
            problems.append(err)

    if problems:
        print("problems:")
        for p in problems:
            print(f"  {p}")
    print(f"{filled} cells measured, {len(BACKBONES) + len(ABLATIONS) - missing} rows "
          f"filled, {missing} rows still without a result file")

    if args.check:
        print("(--check: nothing written)")
        return
    TEX.write_text("\n".join(lines) + "\n")
    print(f"wrote {TEX}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main() or 0)
