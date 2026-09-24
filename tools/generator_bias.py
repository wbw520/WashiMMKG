"""Did Gemma get an advantage from having written the questions?

Both backbones answer the same reserve items twice: once in Gemma's wording, once in
Qwen3.8's.  Gemma's margin over Qwen3.5 is read off each wording separately.  If the two
margins agree, authorship bought nothing; if the margin is larger on Gemma's own wording,
the difference between them is the size of the home-field advantage, and that number --
not a claim of "no bias" -- is what belongs in the paper.
"""
import json, pathlib, sys, statistics

R = pathlib.Path("results")

def rows(name):
    p = R / f"bias-{name}.json"
    if not p.exists():
        sys.exit(f"missing {p} -- run tools/overnight2.sh first")
    return {r["qid"]: r for r in json.loads(p.read_text())["rows"]}

arms = {(b, w): rows(f"{b}-on-{w}")
        for b in ("gemma", "qwen35") for w in ("gemma", "qwen38")}

common = set.intersection(*(set(a) for a in arms.values()))
mm = {q for q in common if arms[("gemma", "gemma")][q]["type"] == "mm_chain"}
print(f"scored by all four arms: {len(common)} items "
      f"({len(mm)} of them mm_chain, whose two wordings come from different benchmark "
      f"revisions -- reported separately)")

def acc(b, w, ids):
    r = arms[(b, w)]
    return statistics.mean(r[q]["correct"] for q in ids)

for label, ids in (("all common items", common), ("excluding mm_chain", common - mm)):
    if not ids:
        continue
    print(f"\n--- {label} (n={len(ids)}) ---")
    m = {}
    for w in ("gemma", "qwen38"):
        g, q = acc("gemma", w, ids), acc("qwen35", w, ids)
        m[w] = g - q
        print(f"  {w:6s} wording:  Gemma {g:.4f}   Qwen3.5 {q:.4f}   margin {g - q:+.4f}")
    d = m["gemma"] - m["qwen38"]
    print(f"  home-field advantage = {m['gemma']:+.4f} - {m['qwen38']:+.4f} = {d:+.4f}")
    print(f"  -> {'negligible' if abs(d) < 0.02 else 'REPORT THIS'}: Gemma's margin is "
          f"{abs(d):.4f} {'larger' if d > 0 else 'smaller'} on questions it wrote")
