"""Exit 0 when a result file actually covers the current test split.

`[ -s file ]` is not the question. Every cell of the previous benchmark left a file behind,
and those files are still on disk with their old 1,959 rows; a script that skips on
existence alone treats a stale cell as finished and fills the paper with numbers measured
against questions that no longer exist. qwen27's fmt, graph and wikiwalk were skipped that
way. Ask instead whether the rows are exactly the split being reported.

    python tools/cell_done.py results/fmt.test.qwen27.json && echo done
"""
import json, pathlib, sys

if len(sys.argv) != 2:
    sys.exit(2)
p = pathlib.Path(sys.argv[1])
if not p.exists() or p.stat().st_size == 0:
    sys.exit(1)
base = pathlib.Path(__file__).resolve().parents[1]
try:
    want = set(json.loads((base / "bench" / "split.json").read_text())["test"])
    have = {r["qid"] for r in json.loads(p.read_text())["rows"]}
except Exception:
    sys.exit(1)
sys.exit(0 if have == want else 1)
