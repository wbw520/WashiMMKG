"""Compare every number the paper states against the result file it came from.

Written after a row was filled from a progress notification instead of from disk and was
wrong by .015. Numbers in the paper should only ever come from results/*.json, and this
re-checks that they still do.

    python tools/check_paper_numbers.py
"""
from __future__ import annotations

import json
import os
import pathlib
import re
import sys

BASE = pathlib.Path(__file__).resolve().parents[1]
TEX = BASE.parent.parent / "paper" / "sn-article.tex"

ABLATIONS = {
    "Full WikiWalk": "abl-full",
    "no chain verification": "abl-no-verify",
    "verify against the whole question": "abl-verify-question",
    "best chain drained": "abl-drain-best-chain",
    "1 picture shown to the navigator": "abl-nav1",
    "3 pictures shown to the navigator": "abl-nav3",
    "captions kept, answer withheld": "abl-keep-captions",
    "all captions shown": "abl-all-captions",
}
BACKBONES = {"Gemma-4-31B": "gemma31", "Qwen3.5-27B": "qwen27",
             "Qwen3.5-9B": "qwen9", "Gemma-4-12B": "gemma12",
             "Qwen3.8-27B": "qwen38"}
METHODS = ["direct", "know", "know+", "fmt", "graph", "wikiwalk"]


def summary(path: pathlib.Path) -> dict | None:
    return json.loads(path.read_text())["summary"] if path.exists() else None


def four(x: float) -> str:
    return f"{x:.4f}".lstrip("0")


def main() -> int:
    tex = TEX.read_text().splitlines()
    bad = 0

    print("ablation table")
    for label, tag in ABLATIONS.items():
        s = summary(BASE / os.environ.get("WASHI_RESULTS", "results") / f"{tag}.test.json")
        line = next((l for l in tex if label in l and "&" in l), None)
        if s is None or line is None:
            print(f"  --   {label:36} {'no result yet' if s is None else 'no row'}")
            continue
        got = re.findall(r"\.\d{4}", line)[:3]
        want = [four(s["acc"]), four(s["precision"]), four(s["recall"])]
        if got != want:
            bad += 1
            print(f"  BAD  {label:36} paper={got} file={want}")
        else:
            print(f"  ok   {label:36} {want[0]}")

    print("\nmain table")
    # the model name also appears in prose and in the generator-bias table, so locate the
    # main table first and search only inside it
    start = next(i for i, l in enumerate(tex) if "label{tab:main_results}" in l)
    end = next(i for i in range(start, len(tex)) if "end{tabular" in tex[i])
    body = tex[start:end]
    for name, tag in BACKBONES.items():
        row = next((i for i, l in enumerate(body) if l.startswith(name)), None)
        have = {m: summary(BASE / os.environ.get("WASHI_RESULTS", "results") / f"{m}.test.{tag}.json") for m in METHODS}
        have["react"] = summary(BASE / os.environ.get("WASHI_RESULTS", "results") / f"react.test.{tag}.json")
        n = sum(v is not None for v in have.values())
        if row is None:
            print(f"  --   {name:14} no row in the table ({n}/6 measured)")
            continue
        block = " ".join(body[row:row + 9])
        missing = [m for m, v in have.items() if v is not None and four(v["acc"]) not in block]
        if missing:
            bad += 1
            print(f"  BAD  {name:14} measured but not in the paper: {', '.join(missing)}")
        else:
            print(f"  ok   {name:14} {n}/6 measured, all present")

    check_supplement()

    print("\n" + ("all paper numbers match their result files"
                  if not bad else f"{bad} discrepancies -- fix before submitting"))
    return 1 if bad else 0


def check_supplement() -> None:
    """The supplement is generated, so the check is that the file on disk matches what the
    generator produces now. A stale supplement is the same failure as a stale table: the
    numbers look fine and belong to an older measurement."""
    import subprocess, sys, tempfile, shutil, os as _os
    sup = BASE.parent.parent / "paper" / "supplementary.tex"
    print()
    print("supplement")
    if not sup.exists():
        print("  --   supplementary.tex has not been generated")
        return
    before = sup.read_text()
    backup = tempfile.NamedTemporaryFile("w", suffix=".tex", delete=False)
    backup.write(before); backup.close()
    env = dict(_os.environ)
    r = subprocess.run([sys.executable, str(BASE / "tools" / "make_supplement.py")],
                       capture_output=True, text=True, env=env)
    after = sup.read_text()
    if r.returncode != 0:
        shutil.copy(backup.name, sup)
        print(f"  !!   the generator failed: {r.stderr.strip().splitlines()[-1:]}")
    elif after == before:
        print("  ok   supplementary.tex matches the current results")
    else:
        print("  !!   supplementary.tex was stale and has been regenerated")
    _os.unlink(backup.name)


if __name__ == "__main__":
    sys.exit(main())
