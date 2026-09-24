"""Evaluate a retrieval method on the Washi QA benchmark.

Reports answer accuracy (LLM-as-judge against the reference answer) and retrieval
precision/recall against each item's computed `gold_triples`.

One deviation from scoring retrieval by exact triple match: an alternative path that
entails the same answer is credited. The benchmark's gold chain is *a* derivation, not
the only one, and the wiki-walk is built to retrieve chains -- scoring only exact matches
would reward the method for reproducing the sampler's arbitrary choice among equivalent
routes rather than for finding evidence that answers the question. Both the strict and
the credited numbers are reported so the effect of that choice is visible rather than
buried, and the baselines receive the same treatment.

Run:
    python -m wikirag.evaluate --method wikiwalk --limit 100 \
        --base-url http://127.0.0.1:8000/v1 --model qwen3.5-27b
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from washi_kg.config import load_config  # noqa: E402
from washi_kg.llm import LLMClient, extract_json  # noqa: E402
from wikirag import baselines as B  # noqa: E402
from wikirag.counting import CountingLLM  # noqa: E402
from wikirag.index import TextIndex  # noqa: E402
from wikirag.page import WikiGraph  # noqa: E402
from wikirag.pipeline import WikiWalkRAG  # noqa: E402
from wikirag.walker import WikiWalker  # noqa: E402

JUDGE_SYSTEM = """\
You judge whether a candidate answer to a Washi question matches the reference.

Say correct only when the candidate conveys the reference's substance - the same entity,
value or distinction. Extra detail is fine; hedging is fine if the right answer is stated.
Naming something else, omitting the key content, or declining to answer is incorrect.

Return ONLY JSON: {"correct": true|false}
"""


def judge(llm, question: str, reference: str, atoms, candidate: str) -> bool:
    """Score one answer.

    `llm` must be the judge, which is deliberately not the backbone being scored. Marking
    a model's own work moves the ruler with the model: a backbone tends to recognise its
    own phrasing as correct, and a weak one misjudges as badly as it answers, so a low
    number cannot be split into "answered wrong" and "graded itself wrong". Qwen3.5-9B
    read .8897 on the perfect-evidence condition, where the answer is handed to it --
    part of that gap was its own grading. One judge for every row makes the rows
    comparable; see `--judge-url`.
    """
    try:
        reply = llm.complete(
            JUDGE_SYSTEM,
            f"Question: {question}\n\nReference: {reference}\n"
            f"Key content: {'; '.join(atoms)}\n\nCandidate: {candidate}",
            max_tokens=60,
        )
    except Exception:
        return False
    obj = extract_json(reply, default={})
    return bool(isinstance(obj, dict) and obj.get("correct"))


def prf(pred, gold):
    pred, gold = {tuple(p) for p in pred}, {tuple(g) for g in gold}
    tp = len(pred & gold)
    p = tp / len(pred) if pred else 0.0
    r = tp / len(gold) if gold else 0.0
    return p, r


def entailed_recall(pred, gold):
    """Credit a retrieved triple that connects the same pair of entities as a gold one.

    An alternative route between the same endpoints answers the same question; only the
    relation label differs.
    """
    pred, gold = {tuple(p) for p in pred}, {tuple(g) for g in gold}
    pairs = {(h, u) for h, _, u in pred} | {(u, h) for h, _, u in pred}
    hit = sum(1 for h, _, u in gold if (h, u) in pairs)
    return hit / len(gold) if gold else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--method", default="wikiwalk",
                    choices=["wikiwalk", "direct", "know", "know+", "fmt", "graph", "react"])
    ap.add_argument("--qa", default=None,
                    help="benchmark file (default bench/washi_qa.json); use to score the "
                         "same gold under a different generator's wording")
    ap.add_argument("--split", default="dev", choices=["dev", "test", "reserve", "all"],
                    help="dev for tuning, test for the number that gets reported")
    ap.add_argument("--limit", type=int, default=0, help="0 = the whole split")
    ap.add_argument("--type", default=None,
                    help="restrict to one question type (diagnostics; the reported "
                         "numbers always run the whole split)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--budget", type=int, default=5)
    ap.add_argument("--only-qids", default=None,
                    help="JSON file holding a list of qids; score only those, keeping "
                         "every other row of --out untouched")
    ap.add_argument("--judge-url", default=None,
                    help="OpenAI-compatible endpoint that grades the answers. Defaults to "
                         "the backbone under test, which makes rows incomparable; pass a "
                         "fixed judge so every row is graded by the same reader")
    ap.add_argument("--judge-model", default=None,
                    help="model name at --judge-url")
    ap.add_argument("--reuse", action="store_true",
                    help="carry rows for questions already scored in --out over instead of "
                         "asking the model again, and evaluate only what is missing")
    ap.add_argument("--nav-images", type=int, default=0,
                    help="show the navigator this many of a page's pictures instead of "
                         "their captions; 0 keeps the text-only walk")
    ap.add_argument("--nav-keep-captions", action="store_true",
                    help="with --nav-images, leave the captions in place as well, so the "
                         "navigator both looks and reads")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="qwen3.5-27b")
    ap.add_argument("--clip", default="openai/clip-vit-large-patch14")
    # vLLM holds the serving GPUs at ~0.9 utilisation, so the ranker needs its own card
    ap.add_argument("--clip-device", default="cuda:2")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--no-image-roots", action="store_true",
                    help="ablation: no CLIP image similarity in root initialisation")
    ap.add_argument("--subqueries", type=int, default=2,
                    help="ablation: how many sub-queries walk (1 = whole question only)")
    ap.add_argument("--merge-cite-follow", action="store_true",
                    help="ablation: one selection per page instead of cite vs follow")
    ap.add_argument("--verify-question", action="store_true",
                    help="ablation: verify each chain against the whole question")
    # The search itself. These were fixed in the walker's signature and never varied, so
    # the ablation table measured every part of the method except the beam it is named
    # after. Defaults here must match WikiWalker's own.
    ap.add_argument("--react-image-search", dest="react_image_search", action="store_true",
                    default=True,
                    help="react: an image_search action backed by the same CLIP index as "
                         "WikiWalk's image roots (the default; the paper's ReAct has it)")
    ap.add_argument("--no-react-image-search", dest="react_image_search", action="store_false",
                    help="react: the text-only agent (search/expand/finish only)")
    ap.add_argument("--react-steps", type=int, default=16,
                    help="react: tool-choosing turns, set to match WikiWalk's call count")
    ap.add_argument("--model-select", action="store_true",
                    help="diagnostic: the backbone chooses the final K triples instead of "
                         "taking them round-robin from the ranked chains; the walk itself "
                         "is unchanged")
    ap.add_argument("--drop-edges", type=float, default=0.0,
                    help="ablation: delete this fraction of graph edges at random")
    ap.add_argument("--drop-seed", type=int, default=0,
                    help="seed for --drop-edges, so a rate is one fixed graph")
    ap.add_argument("--beam", type=int, default=3,
                    help="ablation: partial chains kept per expansion step")
    ap.add_argument("--max-depth", type=int, default=3,
                    help="ablation: how many links deep a chain may go")
    ap.add_argument("--roots", type=int, default=2,
                    help="ablation: search roots per sub-query")
    ap.add_argument("--max-pages", type=int, default=24,
                    help="ablation: pages one sub-query may open")
    ap.add_argument("--drain-best-chain", action="store_true",
                    help="ablation: fill the budget from the best chain, do not cycle")
    ap.add_argument("--hide-answer-captions", action="store_true",
                    help="for items whose answer IS what a picture shows, withhold that "
                         "picture's caption from retrieval while still showing the "
                         "picture. Without it the caption travels with the entity and the "
                         "item is settled by arriving rather than by looking")
    ap.add_argument("--answer-images", type=int, default=3,
                    help="how many retrieved pictures the answerer is shown. Images are "
                         "collected in the order their triples appear, so on a chain the "
                         "endpoint's photograph -- the one an mm_chain item asks about "
                         "-- is the last to be added and the first to be cut")
    ap.add_argument("--text-only-kg", action="store_true",
                    help="strip A_img from the graph entirely: the ablation that asks "
                         "what the MMKG is worth over a text-only KG")
    ap.add_argument("--no-captions", action="store_true",
                    help="ablation: blank every image caption in the graph, so pages and "
                         "evidence carry pictures without any wording about them")
    ap.add_argument("--no-mention-roots", action="store_true",
                    help="ablation: rank entity names against the question instead of "
                         "resolving the mentions it makes")
    args = ap.parse_args()

    os.environ.setdefault("OPENAI_API_KEY", "EMPTY")
    cfg = load_config()
    cfg.raw["llm"].update(provider="openai_compatible",
                          base_url=args.base_url, model=args.model)

    # The grader is built from its own config so that one reader scores every backbone.
    # Falling back to the backbone keeps old commands working, but the rows they produce
    # are each graded by a different model and should not be compared across rows.
    judge_cfg = cfg
    if args.judge_url:
        import copy
        judge_cfg = copy.deepcopy(cfg)
        judge_cfg.raw["llm"].update(provider="openai_compatible",
                                    base_url=args.judge_url,
                                    model=args.judge_model or args.model)

    g = WikiGraph(str(cfg.output_dir / "graph.v2.json"),
                  text_only=args.text_only_kg,
                  drop_edges=args.drop_edges, drop_seed=args.drop_seed)
    if args.text_only_kg:
        # a text-only KG has no pictures to seed a search with, and none to hand over as
        # gold evidence either -- otherwise the ablation would still be multimodal
        args.no_image_roots = True
    if args.no_captions:
        # the pictures stay (image roots, evidence images); only their wording goes, on
        # every page the navigator reads and in every evidence line the answerer sees
        n_blank = 0
        for e in g.E.values():
            for im in e.get("attributes", {}).get("img", []):
                if im.get("caption"):
                    im["caption"] = ""; n_blank += 1
        print(f"no-captions: blanked {n_blank} graph captions", flush=True)
    qa_path = Path(args.qa) if args.qa else cfg.base_dir / "bench" / "washi_qa.json"
    qa = json.loads(qa_path.read_text())
    split_path = cfg.base_dir / "bench" / "split.json"
    if args.split != "all" and split_path.exists():
        ids = set(json.loads(split_path.read_text())[args.split])
        qa = [q for q in qa if q["qid"] in ids]
    if args.no_captions:
        # Know+ hands over the benchmark's own image records rather than the graph's, so
        # blanking A_img alone would leave the upper bound reading captions nobody else sees
        n_item = 0
        for q in qa:
            for im in q.get("images") or []:
                if im.get("caption"):
                    im["caption"] = ""; n_item += 1
        print(f"no-captions: blanked {n_item} benchmark item captions", flush=True)
    if args.type:
        qa = [q for q in qa if q["type"] == args.type]
    if args.only_qids:
        # Re-run a named handful inside an existing cell, leaving every other row as it
        # was. Used to repair items that failed on a transient server error before the
        # client retried them; re-running the whole cell would discard hours of correct
        # work to fix a few dozen rows.
        keep = set(json.loads(Path(args.only_qids).read_text()))
        qa = [q for q in qa if q["qid"] in keep]
        print(f"restricted to {len(qa)} named item(s)")
    random.Random(args.seed).shuffle(qa)
    items = qa[: args.limit] if args.limit else qa

    def _image_index():
        from wikirag.index import ImageIndex

        ent_imgs = {
            n: [i["path"] for i in e["attributes"]["img"]]
            for n, e in g.E.items() if e["attributes"]["img"]
        }
        print(f"building CLIP image index over {len(ent_imgs)} entities ...", flush=True)
        return ImageIndex(ent_imgs, args.clip, device=args.clip_device, base_dir=cfg.base_dir)

    retriever = None
    if args.method in ("fmt", "graph"):
        cls = B.FlatTripletRAG if args.method == "fmt" else B.GraphEgoRAG
        print(f"building {args.method} CLIP index ...", flush=True)
        retriever = cls(g, args.clip, device=args.clip_device,
                        base_dir=cfg.base_dir)
    elif args.method == "react":
        # The generic agentic baseline. It answers through the same branch as fmt and
        # graph below, so the only thing that differs from them is how the evidence was
        # found -- and the only thing that differs from WikiWalk is the search itself.
        # Only the index is shared: the agent holds a per-question LLM client, like the
        # walker does, because the client counts the calls that question cost.
        react_index = TextIndex(list(g.E), cfg.get("embeddings", "model"), device="cpu")
        react_img_idx = _image_index() if args.react_image_search else None
    elif args.method == "wikiwalk":
        idx = TextIndex(list(g.E), cfg.get("embeddings", "model"), device="cpu")
        img_idx = None if args.no_image_roots else _image_index()

    def run_one(it: dict) -> dict:
        # One item must not be able to end the run. ThreadPoolExecutor.map re-raises in
        # the consumer, so an exception anywhere in 1,600 items discards every result
        # already computed. A failed item is recorded as incorrect and counted, and the
        # count is printed, so a systematic breakage still shows up instead of hiding.
        try:
            return _run_one(it)
        except Exception as exc:  # noqa: BLE001
            return {
                "qid": it["qid"], "type": it["type"], "multimodal": it.get("multimodal"),
                "correct": False, "precision": 0.0, "recall": 0.0,
                "recall_entailed": 0.0, "n_calls": 0, "seconds": 0.0,
                "answer": "", "triples": [], "error": f"{type(exc).__name__}: {exc}"[:300],
            }

    def _run_one(it: dict) -> dict:
        llm = CountingLLM(LLMClient(cfg))
        # not wrapped in CountingLLM: grading is not part of the method's cost
        judge_llm = LLMClient(judge_cfg) if args.judge_url else llm
        t0 = time.time()
        triples: list = []
        images: list = []
        # q_m exists only where the question actually comes with a picture -- the visual
        # items, whose image IS the query. Everything else carries images only because
        # some entity on its reasoning path happens to be illustrated; those are evidence
        # sitting in the graph, to be reached by retrieval like any other attribute.
        # Handing them over as part of the question is wrong twice: nobody asking "what is
        # this paper made from" is holding a photograph, and the path's images can include
        # the answer entity itself, which would staple a picture of the answer to the
        # question.
        has_query_image = it["type"] == "multimodal"
        q_images = (
            [str(cfg.base_dir / i["path"]) for i in (it.get("images") or [])[:2]
             if (cfg.base_dir / i["path"]).exists()]
            if has_query_image else []
        )

        if args.method == "direct":
            ans = B.direct_answer(llm, it["question"], q_images)
        elif args.method in ("know", "know+"):
            triples = [tuple(t) for t in it["gold_triples"]]
            ev = "\n".join(f"- {B.triple_text(*t)}" for t in triples)
            paths = None
            if args.method == "know+" and it.get("images"):
                ev += "".join(f"\n- [image of {i.get('entity')}] {i.get('caption') or ''}"
                              for i in it["images"][:3])
                # A_img can outlive its file (paths were recorded under a different
                # Unicode normalisation); a missing image must not abort the run
                paths = [str(cfg.base_dir / i["path"]) for i in it["images"][:3]
                         if (cfg.base_dir / i["path"]).exists()] or None
                images = it["images"][:3]
            ans = B.answer_from_evidence(llm, it["question"], ev, paths)
        elif args.method in ("fmt", "graph", "react"):
            r = (B.ReActRAG(g, llm, react_index, steps=args.react_steps,
                            base_dir=cfg.base_dir, image_index=react_img_idx)
                 if args.method == "react" else retriever)
            triples = r.retrieve(it["question"], args.budget,
                                 query_images=q_images or None)
            ev = "\n".join(f"- {B.triple_text(*t)}" for t in triples) or "(none)"
            # A baseline that retrieves the right entity but is structurally incapable of
            # handing over its picture would lose every visual-terminal item on the
            # plumbing rather than on the retrieval. These baselines rank with images, so
            # they surrender images too, on exactly the terms WikiWalk does: same budget,
            # same caption mask.
            images = []
            seen_p: set = set()
            for h, _, u in triples:
                for n in (h, u):
                    for im in g.images_of(n):
                        if im.get("path") and im["path"] not in seen_p:
                            seen_p.add(im["path"])
                            images.append({"entity": n, **im})
            hide_b = frozenset(
                i["path"] for i in (it.get("images") or [])
            ) if (args.hide_answer_captions
                  and it["type"] in ("mm_chain", "multimodal")) else frozenset()
            for im in images[: args.answer_images]:
                cap = "" if im["path"] in hide_b else (im.get("caption") or "")
                ev += f"\n- [image of {im['entity']}] {cap}"
            paths = list(q_images)
            for im in images[: args.answer_images]:
                fp = cfg.base_dir / im["path"]
                if fp.exists() and str(fp) not in paths:
                    paths.append(str(fp))
            paths = paths[: args.answer_images]
            ans = B.answer_from_evidence(llm, it["question"], ev, paths or None)
        else:
            # the caption of the item's own image is its gold answer, and it lives on the
            # entity in the graph; leaving it visible turns "what does this photograph
            # show" into a lookup for anything that reaches the entity
            hide = frozenset(
                i["path"] for i in (it.get("images") or [])
            ) if (args.hide_answer_captions
                  and it["type"] in ("mm_chain", "multimodal")) else frozenset()
            walker = WikiWalker(
                g, llm, idx,
                hide_captions=hide,
                nav_images=args.nav_images,
                nav_keep_captions=args.nav_keep_captions,
                base_dir=cfg.base_dir,
                max_subqueries=args.subqueries,
                merge_cite_follow=args.merge_cite_follow,
                verify_against_question=args.verify_question,
                beam=args.beam,
                max_depth=args.max_depth,
                roots_per_subquery=args.roots,
                max_pages=args.max_pages,
            )
            rag = WikiWalkRAG(g, walker, llm, evidence_budget=args.budget,
                              max_answer_images=args.answer_images,
                              hide_captions=hide,
                              verify=not args.no_verify, base_dir=cfg.base_dir,
                              drain_best_chain=args.drain_best_chain,
                              use_mentions=not args.no_mention_roots,
                              model_select=args.model_select)
            iscores = None
            if img_idx is not None and q_images:
                iscores = img_idx.score_image(q_images[0])
            r = rag.run(it["question"], image_scores=iscores, query_images=q_images)
            triples, images, ans = r.triples, r.images, r.answer

        ok = judge(judge_llm, it["question"], it["answer"], it["answer_atoms"], ans)
        p, rc = prf(triples, it["gold_triples"])
        return {
            "qid": it["qid"], "type": it["type"], "multimodal": it.get("multimodal"),
            "correct": ok, "precision": p, "recall": rc,
            "recall_entailed": entailed_recall(triples, it["gold_triples"]),
            "n_calls": llm.calls, "seconds": time.time() - t0,
            "answer": ans, "triples": [list(t) for t in triples],
        }

    # Dissolving the reserve split into test grew the evaluation set, and re-running a
    # method over the questions it had already answered would cost hours per cell to
    # reproduce numbers it already holds. Each item is scored independently, so a row
    # stays valid as long as its question and gold are unchanged, and --reuse keeps those
    # and asks the model only about what is new.
    #
    # The qid hashes the *computed gold* -- type, mode, answer atoms, triples -- and not
    # the sentence, so re-wording a question leaves its id alone and a stale row would be
    # carried over silently. `bench/rewritten_qids.json` lists ids whose text has changed
    # since they were last scored; those are always re-asked.
    dest = (Path(args.out) if args.out
            else cfg.base_dir / "results" / f"{args.method}.{args.split}.json")
    kept: list[dict] = []
    if args.reuse and dest.exists():
        try:
            prior = {r["qid"]: r for r in json.loads(dest.read_text())["rows"]}
        except Exception:
            prior = {}
        stale_path = cfg.base_dir / "bench" / "rewritten_qids.json"
        stale = set(json.loads(stale_path.read_text())) if stale_path.exists() else set()
        # The re-worded list is not applied when repairing named rows. Repair keeps every
        # row it was not asked to touch; dropping the re-worded ones here removed them
        # from `prior` while leaving them out of the re-run set too, so they vanished --
        # fmt.test.qwen27 came back with 1,852 rows of 2,007. A cell that needs re-wording
        # handled is re-run as a whole cell, not repaired row by row.
        if not args.only_qids:
            prior = {q: r for q, r in prior.items() if q not in stale}
        if args.only_qids:
            # Repairing a few rows inside a finished cell: every row NOT named is kept
            # exactly as it was, and every row that IS named is re-asked even though a
            # row for it already exists -- that existing row is the failure being
            # repaired. Without the first half the file would be truncated to the named
            # handful; without the second, nothing would be re-run at all.
            named = {i["qid"] for i in items}
            kept = [r for q, r in prior.items() if q not in named]
            print(f"repairing {len(named)} row(s); {len(kept)} left untouched", flush=True)
        else:
            wanted = {i["qid"] for i in items}
            kept = [prior[q] for q in wanted & set(prior)]
            items = [i for i in items if i["qid"] not in prior]
            print(f"reusing {len(kept)} rows already in {dest.name}; "
                  f"{len(items)} left to score", flush=True)

    print(f"method={args.method}  split={args.split}  n={len(items) + len(kept)}  "
          f"budget={args.budget}", flush=True)
    rows: list[dict] = list(kept)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for n, row in enumerate(pool.map(run_one, items), start=1):
            rows.append(row)
            if n % 20 == 0 or n == len(items):
                acc = sum(r["correct"] for r in rows) / len(rows)
                print(f"  [{n}/{len(items)}] acc={acc:.3f}", flush=True)

    failed = [r for r in rows if r.get("error")]
    if failed:
        import collections as _c
        kinds = _c.Counter(r["error"].split(":")[0] for r in failed)
        print(f"  !! {len(failed)}/{len(rows)} items errored and are scored incorrect: "
              + ", ".join(f"{k}x{v}" for k, v in kinds.most_common()), flush=True)
    acc = sum(r["correct"] for r in rows) / max(len(rows), 1)
    out = {
        "method": args.method, "split": args.split,
        "n": len(rows), "budget": args.budget,
        # The command that produced this cell, so a repair can replay it instead of
        # guessing. A repair that rebuilds the command from a fixed template silently
        # re-ran rows of fmt.test.qwen27.k8 at the default budget of five inside a cell
        # measured at eight, and the only trace was the summary's own budget field
        # changing under it.
        "argv": sys.argv[1:],
        "drop_edges": args.drop_edges, "drop_seed": args.drop_seed,
        "beam": args.beam, "max_depth": args.max_depth,
        "roots": args.roots, "max_pages": args.max_pages,
        "model": args.model,
        # who graded this, so a file can be traced back to its ruler. Rows graded by
        # different judges are not comparable, and without this the file does not say.
        "judged_by": args.judge_model or (args.judge_url and args.model) or args.model,
        "acc": acc,
        "precision": statistics.mean(r["precision"] for r in rows),
        "recall": statistics.mean(r["recall"] for r in rows),
        "recall_entailed": statistics.mean(r["recall_entailed"] for r in rows),
        "calls_mean": statistics.mean(r["n_calls"] for r in rows),
        "calls_sd": statistics.pstdev([r["n_calls"] for r in rows]),
        "seconds_mean": statistics.mean(r["seconds"] for r in rows),
        "seconds_sd": statistics.pstdev([r["seconds"] for r in rows]),
        "by_type": {},
    }
    for t in sorted({r["type"] for r in rows}):
        sub = [r for r in rows if r["type"] == t]
        out["by_type"][t] = {
            "n": len(sub),
            "acc": sum(r["correct"] for r in sub) / len(sub),
            "recall": statistics.mean(r["recall"] for r in sub),
        }

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps({"summary": out, "rows": rows},
                               ensure_ascii=False, indent=2))
    print(f"\nacc={out['acc']:.4f}  P={out['precision']:.4f}  R={out['recall']:.4f}"
          f"  R_ent={out['recall_entailed']:.4f}")
    print(f"calls={out['calls_mean']:.1f}+-{out['calls_sd']:.1f}  "
          f"{out['seconds_mean']:.1f}+-{out['seconds_sd']:.1f}s")
    print("by type:", {k: round(v["acc"], 3) for k, v in out["by_type"].items()})
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
