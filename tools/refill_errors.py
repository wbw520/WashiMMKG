"""Re-run only the items that failed with a transient server error.

A 500 from the serving stack is the server dropping a request, not the model failing to
answer, but the evaluator scores it wrong either way. That cost the single-call baselines
0.3-0.8% of their items while WikiWalk, which swallows an exception at each of its ~19
steps and answers anyway, lost none -- a difference produced by where the try/except sat
rather than by the retrieval being compared. `washi_kg.llm._with_retry` now gives every
method the same tolerance, but rows already on disk were measured without it.

Rather than re-run whole cells, this re-runs only the rows carrying an `error`, using the
same command the cell was produced with. Typically a few dozen items per file.

    python tools/refill_errors.py --dry-run
    python tools/refill_errors.py --url http://127.0.0.1:8001/v1 --model gemma-4-31b-it \
        --tag gemma31 --judge-url http://127.0.0.1:8001/v1 --judge-model gemma-4-31b-it
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys

BASE = pathlib.Path(__file__).resolve().parents[1]
R = BASE / "results"


def failed(p: pathlib.Path) -> list[str]:
    try:
        d = json.loads(p.read_text())
    except Exception:
        return []
    # A request that fails during answer generation is caught there and its message is
    # written into the answer, not into `error`. Those rows are scored wrong like any
    # other wrong answer and were invisible to this repair: 30 of them sat in the
    # wikiwalk cells of three backbones. Both shapes of failure are re-asked.
    out = []
    for r in d.get("rows", []):
        if r.get("error") or "generation failed" in str(r.get("answer") or ""):
            out.append(r["qid"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag")
    ap.add_argument("--url")
    ap.add_argument("--model")
    ap.add_argument("--judge-url")
    ap.add_argument("--judge-model")
    ap.add_argument("--clip-device", default="cuda:2")
    ap.add_argument("--dry-run", action="store_true")
    # Two budgets live side by side in results/ while the K=8 set is being measured.
    # Without this a repair aimed at K=8 would also re-ask rows in the finished K=5 files.
    ap.add_argument("--suffix", default="",
                    help="only touch files whose name carries this suffix, e.g. .k8")
    args = ap.parse_args()

    targets = []
    for p in sorted(R.glob("*.test.*.json")) + sorted(R.glob("abl-*.test.json")):
        if args.suffix and args.suffix not in p.name:
            continue
        if not args.suffix and ".k8" in p.name:
            continue
        if args.tag and f".{args.tag}." not in p.name:
            continue
        q = failed(p)
        if q:
            targets.append((p, q))

    if not targets:
        print("no rows carry an error")
        return
    total = sum(len(q) for _, q in targets)
    print(f"{len(targets)} file(s), {total} item(s) to re-run")
    for p, q in targets:
        print(f"  {p.name:34} {len(q):4}")
    if args.dry_run:
        print("(--dry-run: nothing re-run)")
        return
    if not (args.url and args.model):
        sys.exit("--url and --model are required unless --dry-run")

    for p, q in targets:
        method = p.name.split(".")[0]
        only = BASE / "bench" / f".refill_{p.stem}.json"
        only.write_text(json.dumps(sorted(q)))
        # Replay the cell's own command. Rebuilding it from a template drops whatever
        # made the cell what it is -- its budget, its ablation flag, its search
        # parameters -- and re-runs those rows under a different method inside a file
        # that still calls itself the original one.
        prior_argv = (json.loads(p.read_text()).get("summary") or {}).get("argv")
        if prior_argv:
            drop_next = False
            cmd = [sys.executable, "-m", "wikirag.evaluate"]
            for tok in prior_argv:
                if drop_next:
                    drop_next = False
                    continue
                if tok in ("--out", "--only-qids", "--limit", "--base-url", "--model",
                           "--clip-device", "--judge-url", "--judge-model", "--workers"):
                    drop_next = True
                    continue
                cmd.append(tok)
            cmd += ["--workers", "8", "--reuse",
                    "--base-url", args.url, "--model", args.model,
                    "--clip-device", args.clip_device,
                    "--only-qids", str(only), "--out", str(p)]
        elif method.startswith("abl-"):
            # Older cells recorded no argv, and an ablation's flag cannot be recovered
            # from its name. Re-run the condition whole rather than repair it wrongly.
            print(f"    skipped: {p.name} predates argv recording; re-run the condition")
            only.unlink(missing_ok=True)
            continue
        else:
            cmd = [sys.executable, "-m", "wikirag.evaluate", "--method", method,
                   "--split", "test", "--workers", "8", "--reuse",
                   "--base-url", args.url, "--model", args.model,
                   "--clip-device", args.clip_device, "--hide-answer-captions",
                   "--budget", str((json.loads(p.read_text()).get("summary") or {}).get("budget", 5)),
                   "--only-qids", str(only), "--out", str(p)]
        if args.judge_url:
            cmd += ["--judge-url", args.judge_url, "--judge-model", args.judge_model or args.model]
        print(f"\n>>> {p.name}: {len(q)} item(s)")
        subprocess.run(cmd, cwd=BASE)
        only.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
