"""Stage 3 — filter and label the verbalized benchmark.

Two checks, treated differently on purpose.

**Leakage is a hard filter.** If the question states its own answer the item measures
nothing, and that is a construction defect, not a property of the domain. Naive substring
matching over-fires here: an item about "the Kurotani Washi Cooperative" whose answer
includes "Kurotani washi" is not leaking — the subject cannot be named without the
substring appearing. So the question's known subjects (from the item's spec) are masked
out before matching, and forced-choice items, which must name their options, are exempt.

**Closed-book solvability is a label, not a filter.** An item a model answers without any
evidence is easy, not broken, and deleting those would silently redefine the benchmark
around whatever the probe model happened to know — a moving target that also destroys the
Direct-setting baseline the paper reports. Recording `closed_book_solved` keeps all items
and makes the hard subset available for separate analysis. Deleting is irreversible;
labelling is not.

Run (from code/, conda `washi`):
    python bench/validate.py --base-url http://127.0.0.1:8000/v1 --model qwen3.5-27b
    python bench/validate.py --no-closed-book     # leakage filter only, no model needed
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from washi_kg.config import load_config  # noqa: E402
from washi_kg.llm import LLMClient, extract_json  # noqa: E402

# Atoms this short are matched as whole words only; "su" (the papermaking screen) would
# otherwise "leak" out of the word "surface".
SHORT_ATOM = 6

# Vocabulary that betrays the question's machine origin. A curator asks "which paper",
# never "which entity". Most of these are fixed by re-verbalizing, but a residue survives
# where the population really is heterogeneous ("things located in Gifu") and no natural
# noun covers it; those are dropped rather than shipped reading like a database dump.
PROMISED_IMAGE = re.compile(
    r"accompan(?:y|ies|ying)|attached (?:image|photo)|provided (?:image|photograph)"
    r"|(?:this|the) photograph (?:below|here)|shown below|pictured below",
    re.I)
JARGON = re.compile(
    r"\b(entit(y|ies)|triple|node|graph|relation|dataset|knowledge base)\b", re.I
)

CLOSED_BOOK_SYSTEM = """\
Answer the question about Washi (traditional Japanese paper) from your own knowledge.
You have no reference material. If you do not know, say so plainly rather than guessing —
a confident wrong answer is worse than an admission here.
Reply with at most 40 words.
"""

JUDGE_SYSTEM = """\
You compare a candidate answer with a reference answer for a Washi question.

Say "yes" only if the candidate conveys the reference's substance — the same entities,
values, or distinctions. Extra detail is fine. Missing the key content, naming something
else, or declining to answer is "no".

Return ONLY JSON: {"correct": true|false}
"""


def subjects_of(item: dict) -> list[str]:
    """Entity names the question is *about*, which it must be allowed to name."""
    s = item.get("spec") or {}
    # a visual-identification item must NOT name its subject: recognising it is the task
    if s.get("hide_subject"):
        return []
    if s.get("hide_caption"):
        # the chain's start and its relations may be named; the endpoint may not
        return [s[k] for k in ("start",) if isinstance(s.get(k), str)]
    out: list[str] = []
    for key in ("start", "entity", "left", "right", "correct", "distractor"):
        if isinstance(s.get(key), str):
            out.append(s[key])
    for key in ("candidates",):
        out.extend(x for x in (s.get(key) or []) if isinstance(x, str))
    for c in (s.get("constraints") or []):
        if isinstance(c.get("object"), str):
            out.append(c["object"])
    for side in ("population", "excluded"):
        v = s.get(side)
        if isinstance(v, dict) and isinstance(v.get("object"), str):
            out.append(v["object"])
    return out


def leaked_atoms(item: dict) -> list[str]:
    """Answer atoms the question gives away, ignoring ones it cannot avoid naming."""
    if (item.get("spec") or {}).get("answer_in_question_ok"):
        return []
    q = (item.get("question") or "").lower()
    if not q:
        return []
    # mask the subjects first: an atom only "leaks" if it survives outside their names
    for sub in sorted(subjects_of(item), key=len, reverse=True):
        q = q.replace(sub.lower(), " ")
    hits = []
    for atom in item["answer_atoms"]:
        a = atom.lower().strip()
        if not a:
            continue
        if len(a) <= SHORT_ATOM:
            if re.search(rf"(?<!\w){re.escape(a)}(?!\w)", q):
                hits.append(atom)
        elif a in q:
            hits.append(atom)
    return hits


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--items", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--no-closed-book", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    bench_dir = cfg.base_dir / "bench"
    src = Path(args.items) if args.items else bench_dir / "washi_qa_verbalized.json"
    out_path = Path(args.out) if args.out else bench_dir / "washi_qa.json"

    items = json.loads(src.read_text())

    # The verbalisation file is a cache and only ever grows, so it still holds items the
    # current build has retired -- forked paths, decorative-image questions. Reading it
    # straight through put 1,109 retired questions back into the benchmark. Keep only
    # what this build actually produced.
    built_path = bench_dir / "washi_qa_items.json"
    if built_path.exists() and not args.items:
        built = {i["qid"] for i in json.loads(built_path.read_text())}
        before = len(items)
        items = [i for i in items if i["qid"] in built]
        if before != len(items):
            print(f"cache holds {before} items; {before - len(items)} are not in the "
                  f"current build and are ignored")
    total = len(items)

    kept, dropped = [], collections.Counter()
    for it in items:
        if not (it.get("question") and it.get("answer")):
            dropped["not verbalized"] += 1
            continue
        leaks = leaked_atoms(it)
        if leaks:
            dropped["answer leaked into question"] += 1
            continue
        if JARGON.search(it["question"]):
            dropped["graph jargon in question"] += 1
            continue
        spec = it.get("spec") or {}
        if it.get("type") == "mm_chain" and PROMISED_IMAGE.search(it["question"]):
            # the endpoint's photograph hangs on the entity, not on the question: a
            # reader is told to walk to it. Wording that promises a picture in front of
            # them ("the accompanying photograph") describes a question we never ask.
            dropped["mm_chain promises an attached image"] += 1
            continue
        if spec.get("hide_caption") and isinstance(spec.get("endpoint"), str):
            if spec["endpoint"].lower() in it["question"].lower():
                dropped["visual endpoint named in question"] += 1
                continue
        if spec.get("hide_subject") and isinstance(spec.get("subject"), str):
            # the whole point is that the image is the only handle on the subject
            if spec["subject"].lower() in it["question"].lower():
                dropped["visual subject named in question"] += 1
                continue
        kept.append(it)

    print(f"{total} verbalized items")
    for reason, n in dropped.most_common():
        print(f"  -{n:5d}  {reason}")
    print(f"  ={len(kept):5d}  pass the leakage filter")

    if not args.no_closed_book and kept:
        if args.base_url:
            cfg.raw.setdefault("llm", {})
            cfg.raw["llm"]["provider"] = "openai_compatible"
            cfg.raw["llm"]["base_url"] = args.base_url
            if args.model:
                cfg.raw["llm"]["model"] = args.model
            os.environ.setdefault("OPENAI_API_KEY", "EMPTY")
        llm = LLMClient(cfg)

        def probe(it: dict) -> bool:
            try:
                guess = llm.complete(CLOSED_BOOK_SYSTEM, it["question"], max_tokens=120)
                verdict = llm.complete(
                    JUDGE_SYSTEM,
                    f"Question: {it['question']}\n\nReference: {it['answer']}\n"
                    f"Key content: {'; '.join(it['answer_atoms'])}\n\n"
                    f"Candidate: {guess.strip()}",
                    max_tokens=60,
                )
            except Exception:
                return False
            obj = extract_json(verdict, default={})
            return bool(isinstance(obj, dict) and obj.get("correct"))

        print(f"\nProbing closed-book solvability on {len(kept)} items...")
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for n, (it, solved) in enumerate(zip(kept, pool.map(probe, kept)), start=1):
                it["closed_book_solved"] = solved
                it["closed_book_probe"] = cfg.get("llm", "model")
                if n % 100 == 0 or n == len(kept):
                    print(f"  [{n}/{len(kept)}]", flush=True)

    out_path.write_text(json.dumps(kept, ensure_ascii=False, indent=2))

    by_type = collections.Counter(i["type"] for i in kept)
    n_mm = sum(1 for i in kept if i.get("multimodal"))
    solved = sum(1 for i in kept if i.get("closed_book_solved"))
    print(f"\nFinal: {len(kept)} items | multimodal {n_mm} ({n_mm/max(len(kept),1):.1%})")
    print("by type:", ", ".join(f"{k}={v}" for k, v in by_type.most_common()))
    if not args.no_closed_book:
        print(f"closed-book solvable: {solved} ({solved/max(len(kept),1):.1%}) "
              f"-> hard subset = {len(kept)-solved}")
    try:
        shown = out_path.relative_to(cfg.base_dir)
    except ValueError:
        shown = out_path
    print(f"Wrote {shown}")


if __name__ == "__main__":
    main()
