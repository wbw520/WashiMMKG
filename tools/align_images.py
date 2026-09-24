"""Backfill A_img — align the extracted images that never reached the graph.

`extract_pdf_images` pulled 1,121 images out of the source PDFs, but only 414 were
ever aligned to an entity. The remainder are mostly the seminar decks — the on-site
field investigation the paper names as its third data source, and the most valuable
imagery in the corpus. Multimodal coverage is what bounds the visual subset of the
QA benchmark, so this pass runs the Creator's `align(e)` over what was left behind.

Two things the first pass did not have, and this one needs:

**A content filter.** Slide decks embed their own furniture — logos, header rules,
bullets — once per page. Those repeat at an identical small resolution, which is the
signal used to drop them. The filter deliberately does *not* drop large images that
repeat a resolution: one camera stamps the same dimensions on every genuine photo it
takes, so treating repetition alone as a template signal discards the field photos.

**Page context.** `align_image` scopes candidate entities by the text surrounding an
image, but the extracted files carry only their path. The convention written by
`extract_pdf_images` (``<pdf stem>/p<page>_<idx>.<ext>``) is enough to find the source
PDF in raw/ and re-read that page's text.

Requires a vision backend. Point it at the local vLLM (free, no API spend):
    tools/serve_qwen.sh 0,1 8000 Qwen/Qwen3.5-27B
    python tools/align_images.py --base-url http://127.0.0.1:8000/v1 --model qwen3.5-27b

    python tools/align_images.py --dry-run     # list what would be sent, call nothing
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from washi_kg.config import load_config  # noqa: E402
from washi_kg.creator import Creator  # noqa: E402
from washi_kg.embeddings import Embedder  # noqa: E402
from washi_kg.graph import KnowledgeGraph  # noqa: E402
from washi_kg.llm import LLMClient  # noqa: E402

# A slide-furniture graphic is small AND repeats its exact size across the deck.
# Both conditions are required — see the module docstring.
MIN_SIDE = 200
TEMPLATE_MAX_PIXELS = 1_000_000
TEMPLATE_MIN_REPEATS = 8


def content_images(images_dir: Path, attached: set[str], base_dir: Path) -> list[Path]:
    """Unaligned images that look like content rather than slide furniture.

    A_img records store paths relative to ``base_dir``, so the already-aligned check
    has to compare in that form rather than against the absolute glob result.

    Byte-identical duplicates are collapsed to their first occurrence. A deck reuses
    the same artwork or exhibition logo across many slides, and aligning each copy
    independently is worse than wasteful: the model answers differently each time, so
    one picture ends up illustrating several unrelated entities.
    """
    sizes: dict[Path, tuple[int, int]] = {}
    seen: set[str] = set()
    for p in sorted(images_dir.rglob("*")):
        if not p.is_file() or str(p.relative_to(base_dir)) in attached:
            continue
        try:
            digest = hashlib.sha1(p.read_bytes()).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            sizes[p] = Image.open(p).size
        except Exception:
            continue  # unreadable / exotic encoding

    repeats = collections.Counter(sizes.values())
    out: list[Path] = []
    for p, (w, h) in sizes.items():
        if w < MIN_SIDE or h < MIN_SIDE:
            continue
        if w * h < TEMPLATE_MAX_PIXELS and repeats[(w, h)] >= TEMPLATE_MIN_REPEATS:
            continue
        out.append(p)
    return out


def page_text_index(raw_dir: Path, stems: set[str]) -> dict[tuple[str, int], str]:
    """{(pdf stem, page) -> page text} for the decks that own the pending images."""
    from pypdf import PdfReader

    index: dict[tuple[str, int], str] = {}
    for pdf in raw_dir.rglob("*.pdf"):
        if pdf.stem not in stems:
            continue
        try:
            reader = PdfReader(str(pdf))
        except Exception:
            continue
        for i, page in enumerate(reader.pages, start=1):
            try:
                index[(pdf.stem, i)] = page.extract_text() or ""
            except Exception:
                index[(pdf.stem, i)] = ""
    return index


def parse_page(path: Path) -> int | None:
    """Recover the 1-based page number from the `p<page>_<idx>` filename convention."""
    stem = path.stem
    if not stem.startswith("p") or "_" not in stem:
        return None
    try:
        return int(stem[1:].split("_", 1)[0])
    except ValueError:
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--graph", default=None, help="default: the cleaned graph.v2.json")
    ap.add_argument("--base-url", default=None, help="OpenAI-compatible endpoint (local vLLM)")
    ap.add_argument("--model", default=None, help="served model name")
    ap.add_argument("--limit", type=int, default=0, help="only process the first N (0 = all)")
    ap.add_argument("--workers", type=int, default=16, help="concurrent vision calls")
    ap.add_argument("--no-caption-retry", action="store_true",
                    help="skip the caption-rescoped second round")
    ap.add_argument("--dry-run", action="store_true", help="report the work, call nothing")
    args = ap.parse_args()

    cfg = load_config(args.config)
    graph_path = Path(args.graph) if args.graph else cfg.output_dir / cfg.get(
        "paths", "clean_graph_file", default="graph.v2.json"
    )
    graph = KnowledgeGraph.load(graph_path)
    before = graph.stats()

    attached = {
        i["path"] for e in graph.entities.values() for i in e["attributes"]["img"]
    }
    images_dir = cfg.output_dir / cfg.get("paths", "images_dir", default="extracted_images")
    pending = content_images(images_dir, attached, cfg.base_dir)
    if args.limit:
        pending = pending[: args.limit]

    by_deck = collections.Counter(p.parent.name for p in pending)
    print(f"Pending: {len(pending)} content images across {len(by_deck)} sources")
    for d, c in by_deck.most_common(10):
        print(f"  {c:5d}  {d}")

    if args.dry_run:
        print("\n--dry-run: no model called, graph untouched.")
        return

    # Route the vision calls at a local endpoint without editing the committed default.
    if args.base_url:
        cfg.raw.setdefault("llm", {})
        cfg.raw["llm"]["provider"] = "openai_compatible"
        cfg.raw["llm"]["base_url"] = args.base_url
        if args.model:
            cfg.raw["llm"]["model"] = args.model
        os.environ.setdefault("OPENAI_API_KEY", "EMPTY")

    creator = Creator(
        LLMClient(cfg),
        Embedder(cfg),
        enable_image_alignment=True,
        image_candidate_sim_threshold=float(
            cfg.get("creator", "image_candidate_sim_threshold", default=0.20)
        ),
        max_image_candidates=int(cfg.get("creator", "max_image_candidates", default=15)),
        max_text_attrs_per_entity=int(cfg.get("creator", "max_text_attrs_per_entity", default=8)),
        max_text_attr_chars=int(cfg.get("creator", "max_text_attr_chars", default=200)),
        max_img_attrs_per_entity=int(cfg.get("creator", "max_img_attrs_per_entity", default=8)),
        align_max_tokens=int(cfg.get("creator", "align_max_tokens", default=300)),
    )

    texts = page_text_index(cfg.raw_dir, {p.parent.name for p in pending})
    items = []
    for p in pending:
        page = parse_page(p)
        items.append({
            "path": p,
            "stored": str(p.relative_to(cfg.base_dir)),
            "context_text": texts.get((p.parent.name, page), "") if page else "",
            "source": p.parent.name,
            "page": page,
        })

    state = {"n": 0, "linked": 0}
    hits: collections.Counter = collections.Counter()

    def progress(_item, ent):
        state["n"] += 1
        if ent:
            state["linked"] += 1
            hits[ent] += 1
        if state["n"] % 25 == 0 or state["n"] == len(items):
            print(f"  [{state['n']}/{len(items)}] linked {state['linked']}", flush=True)

    creator.align_images(
        items, graph, workers=args.workers, on_result=progress,
        caption_retry=not args.no_caption_retry,
    )
    linked = state["linked"]
    graph.save(graph_path)
    after = graph.stats()
    print(f"\nLinked {linked}/{len(pending)} images to {len(hits)} entities.")
    print("Most-illustrated:", ", ".join(f"{e}({c})" for e, c in hits.most_common(10)))
    print(f"\n{'metric':22} {'before':>8} {'after':>8}")
    for k in before:
        print(f"{k:22} {before[k]:>8} {after[k]:>8}")


if __name__ == "__main__":
    main()
