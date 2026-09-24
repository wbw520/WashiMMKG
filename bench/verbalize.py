"""Stage 2 — turn computed benchmark items into natural language.

Reads `washi_qa_items.json` (structured gold from `build_qa.py`), asks a model to
phrase each one, writes `washi_qa_verbalized.json`. The gold answer, `gold_triples`
and every other computed field pass through untouched: this stage adds `question` and
`answer` text and nothing else. `validate.py` then checks the text is faithful.

Resumable — completed items are reloaded from the output file, so an interrupted run
picks up where it stopped instead of re-billing work already done.

Run (from code/, conda `washi`, with tools/serve_qwen.sh up):
    python bench/verbalize.py --base-url http://127.0.0.1:8000/v1 --model qwen3.5-27b
    python bench/verbalize.py --limit 20 ...      # smoke-test the prompts first
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench import prompts  # noqa: E402
from washi_kg.config import load_config  # noqa: E402
from washi_kg.llm import LLMClient, extract_json  # noqa: E402


def render(item: dict) -> str | None:
    """Build the user prompt for one item from its structured spec."""
    s, t = item["spec"], item["type"]
    answer = "; ".join(item["answer_atoms"])

    if t == "chain":
        path = "\n".join(f"  {h} --{r}--> {u}" for h, r, u in s["path"])
        return prompts.CHAIN.format(path=path, start=s["start"], answer=answer)

    if t == "intersection":
        cons = "\n".join(f"  - {c['relation']}: {c['object']}" for c in s["constraints"])
        return prompts.INTERSECTION.format(constraints=cons, answer=answer)

    if t == "comparative":
        return prompts.COMPARATIVE.format(
            cls=s["class"], left=s["left"], right=s["right"], relation=s["relation"],
            left_values=", ".join(s["left_values"]),
            right_values=", ".join(s["right_values"]), answer=answer,
        )

    if t == "temporal":
        devs = "\n".join(f"  - {r}: {u}" for r, u in s["developments"])
        makers = "".join(f"\n  - made today by: {h}" for h, _ in s["present_makers"])
        return prompts.TEMPORAL.format(
            entity=s["entity"], year=s["year"], developments=devs,
            makers=makers, answer=answer,
        )

    if t == "aggregation":
        if s.get("mode") == "extremum":
            extra = ""
            if s.get("count"):
                extra = f" (appears {s['count']} times; next: " + ", ".join(
                    f"{n} ({c})" for n, c in s.get("runners_up", [])) + ")"
            return prompts.AGGREGATION_EXTREMUM.format(
                operation=s["operation"], answer_value=s["answer_value"],
                extra=extra, answer=answer,
            )
        return prompts.AGGREGATION_ABSENCE.format(
            pop_rel=s["population"]["relation"], pop_obj=s["population"]["object"],
            pop_size=s["population"]["size"], exc_rel=s["excluded"]["relation"],
            exc_obj=s["excluded"]["object"], answer=answer,
        )

    if t == "mm_chain":
        path = "\n".join(f"  {h} --{r}--> {u}" for h, r, u in s["path"])
        return prompts.MM_CHAIN.format(
            path=path, caption=s["caption"], start=s["start"], answer=answer,
        )

    if t == "multimodal":
        if s.get("mode") == "grounded":
            return prompts.MULTIMODAL_GROUNDED.format(
                subject=s["subject"], relation=s["relation"].replace("_", " "),
                value=s["value"], answer=answer,
            )
        if s.get("mode") == "attribute":
            return prompts.MULTIMODAL_ATTRIBUTE.format(
                subject=s["subject"], relation=s["relation"].replace("_", " "),
                value=s["value"], answer=answer,
            )
        facts = "\n".join(f"  - {r.replace('_', ' ')}: {u}" for r, u in s["facts"])
        return prompts.MULTIMODAL_IDENTIFY.format(
            subject=s["subject"], facts=facts, answer=answer,
        )
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--items", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--redo-types", default="",
                    help="comma-separated types to re-word even when cached. The qid "
                         "hashes the computed gold and not the sentence, so an item "
                         "survives a prompt change with its old wording attached; this "
                         "is how a reworded template reaches the items it already has")
    args = ap.parse_args()

    cfg = load_config(args.config)
    bench_dir = cfg.base_dir / "bench"
    items_path = Path(args.items) if args.items else bench_dir / "washi_qa_items.json"
    out_path = Path(args.out) if args.out else bench_dir / "washi_qa_verbalized.json"

    items = json.loads(items_path.read_text())
    if args.limit:
        items = items[: args.limit]

    redo = {t.strip() for t in args.redo_types.split(",") if t.strip()}
    done: dict[str, dict] = {}
    if out_path.exists():
        for rec in json.loads(out_path.read_text()):
            if rec.get("question") and rec.get("answer"):
                done[rec["qid"]] = rec
    if redo:
        stale = {i["qid"] for i in items if i["type"] in redo and i["qid"] in done}
        for q in stale:
            done.pop(q, None)
        print(f"re-wording {len(stale)} cached item(s) of type {sorted(redo)}")
        # anything reworded must not have an old score carried over by --reuse
        reword_path = bench_dir / "rewritten_qids.json"
        prior = set(json.loads(reword_path.read_text())) if reword_path.exists() else set()
        reword_path.write_text(json.dumps(sorted(prior | stale), indent=2))
    todo = [i for i in items if i["qid"] not in done]
    print(f"{len(items)} items, {len(done)} already verbalized, {len(todo)} to do")
    if not todo:
        print("nothing to do")
        return

    if args.base_url:
        cfg.raw.setdefault("llm", {})
        cfg.raw["llm"]["provider"] = "openai_compatible"
        cfg.raw["llm"]["base_url"] = args.base_url
        if args.model:
            cfg.raw["llm"]["model"] = args.model
        os.environ.setdefault("OPENAI_API_KEY", "EMPTY")
    llm = LLMClient(cfg)

    def work(item: dict) -> dict:
        user = render(item)
        if user is None:
            return {**item, "question": None, "answer": None, "error": "no template"}
        try:
            reply = llm.complete(prompts.SYSTEM, user, max_tokens=args.max_tokens)
        except Exception as exc:
            return {**item, "question": None, "answer": None, "error": str(exc)[:200]}
        obj = extract_json(reply, default={})
        q = obj.get("question") if isinstance(obj, dict) else None
        a = obj.get("answer") if isinstance(obj, dict) else None
        return {
            **item,
            "question": q if isinstance(q, str) and q.strip() else None,
            "answer": a if isinstance(a, str) and a.strip() else None,
        }

    results = list(done.values())
    n_ok = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for n, rec in enumerate(pool.map(work, todo), start=1):
            results.append(rec)
            n_ok += bool(rec.get("question") and rec.get("answer"))
            if n % 50 == 0 or n == len(todo):
                out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2))
                print(f"  [{n}/{len(todo)}] ok {n_ok}", flush=True)

    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    failed = [r for r in results if not (r.get("question") and r.get("answer"))]
    print(f"\nVerbalized {len(results) - len(failed)}/{len(results)}"
          + (f", {len(failed)} failed" if failed else ""))
    try:
        shown = out_path.relative_to(cfg.base_dir)
    except ValueError:
        shown = out_path
    print(f"Wrote {shown}")


if __name__ == "__main__":
    main()
