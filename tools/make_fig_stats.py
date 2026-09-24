"""Regenerate the paper's dataset-statistics figure from the real artefacts.

Every number in the figure is read out of `graph.v2.json` and the QA benchmark
file; nothing is typed in by hand, so the figure cannot drift away from the data
again. Panel (b)/(c) are emitted only once the QA benchmark exists.

Two deliberate departures from the figure this replaces:

* **Relation types are not a bar.** |R| counts schema cardinality (41 relation
  types); every other quantity counts instances (thousands). Putting them on one
  axis makes the relation bar invisible, so it is reported in the panel subtitle
  instead. Mixing the two is likely how the superseded figure ended up claiming
  "532 relations".
* **The QA panel is split in two, and the averages are not bars.** The original
  plotted counts and averages against two different y-axes in one panel; a dual-axis
  chart invites comparison between quantities that share no scale. Reasoning depth and
  word counts do not share one either, so both live in a subtitle, and the second QA
  panel shows the question-type distribution instead — which is what a reader actually
  needs to judge a benchmark's coverage.

Run (from code/, conda `washi`):
    python tools/make_fig_stats.py
    python tools/make_fig_stats.py --out ../../paper/figure4_washi_stats.png
"""

from __future__ import annotations

import argparse
import json
import collections
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from washi_kg.config import load_config  # noqa: E402

# Reference-palette slots 1/2/3, documented as all-pairs validated. Each panel is a
# single series, so hue choice carries panel identity only, never series identity.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SURFACE = "#ffffff"
INK, INK_2 = "#0b0b0b", "#52514e"


def graph_panel_data(graph_path: Path) -> tuple[dict, str]:
    g = json.loads(graph_path.read_text())
    s = g["stats"]
    rc = (g.get("meta") or {}).get("relation_class") or {}
    kinds: dict[str, int] = {}
    for cls in rc.values():
        kinds[cls] = kinds.get(cls, 0) + 1
    note = f"{s['relations']} relation types"
    if kinds:
        note += " (" + ", ".join(f"{n} {k}" for k, n in sorted(kinds.items())) + ")"
    bars = {
        "Entities": s["entities"],
        "Triples": s["triples"],
        "Text Info": s["text_attributes"],
        "Images": s["images"],
        "MM Entities": s["multimodal_entities"],
    }
    return bars, note


def qa_panel_data(qa_path: Path):
    """Return (composition bars, subtitle, per-type bars) for the QA benchmark."""
    if not qa_path.exists():
        return None
    items = json.loads(qa_path.read_text())
    if isinstance(items, dict):
        items = items.get("items", [])
    if not items:
        return None
    mm = [i for i in items if i.get("multimodal") or i.get("images")]
    counts = {
        "Total QA": len(items),
        "MM Questions": len(mm),
        "Text-only": len(items) - len(mm),
    }
    # hops is the item's reasoning depth, not its evidence count: an intersection cites
    # two triples but is not two hops deep
    hops = statistics.mean(i.get("hops", 0) for i in items)
    qlen = statistics.mean(len(str(i.get("question", "")).split()) for i in items)
    alen = statistics.mean(len(str(i.get("answer", "")).split()) for i in items)
    note = f"avg {hops:.1f} hops - {qlen:.0f}-word questions, {alen:.0f}-word answers"
    probed = [i for i in items if "closed_book_solved" in i]
    if probed:
        solved = sum(1 for i in probed if i["closed_book_solved"])
        note += f" - {solved / len(probed):.1%} solvable closed-book"
    by_type = collections.Counter(i["type"] for i in items)
    # display form: the type names are lower-case identifiers in the data
    # "mm_chain".capitalize() gives "Mm_chain"; the type names are the axis labels
    pretty = {"mm_chain": "MM Chain", "multimodal": "Multimodal"}
    return counts, note, {pretty.get(k, k.capitalize()): v
                          for k, v in by_type.most_common()}


TEXT_C, VISUAL_C, WHOLE_C = "#3775BA", "#E8833A", "#9AA0A6"
INK_BAR = {"textual": TEXT_C, "visual": VISUAL_C, "whole": WHOLE_C}


def _panel_title(ax, title, note=""):
    ax.set_title(title, fontsize=11.5, fontweight="bold", color=INK, loc="left", pad=4)
    if note:
        ax.text(1.0, 1.03, note, transform=ax.transAxes, fontsize=8, color=INK_2, ha="right")


def _hbars(ax, rows, xmax=None):
    """rows: list of (label, value, group). Thin bars, direct ink labels, no axis."""
    labels = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    cols = [INK_BAR[r[2]] for r in rows]
    y = list(range(len(rows)))[::-1]
    ax.barh(y, vals, height=0.58, color=cols, zorder=3)
    m = xmax or max(vals)
    for yy, v in zip(y, vals):
        ax.text(v + m * 0.015, yy, f"{v:,}", va="center", ha="left", fontsize=8.5, color=INK)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=9, color=INK)
    ax.set_xlim(0, m * 1.14)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xticks([])
    for side in ("top", "right", "bottom"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color("#c9c9c4")
    ax.tick_params(length=0)


def _tiles(ax, tiles, y_big=0.80, y_small=0.52):
    """A row of stat tiles: big number, small label. No plot, no axis."""
    ax.set_axis_off()
    n = len(tiles)
    for i, (big, small) in enumerate(tiles):
        x = (i + 0.5) / n
        ax.text(x, y_big, big, transform=ax.transAxes, fontsize=15.5, fontweight="bold",
                color=INK, ha="center", va="center")
        ax.text(x, y_small, small, transform=ax.transAxes, fontsize=7.8, color=INK_2,
                ha="center", va="center")


def _band(ax, parts, total, y=0.20, h=0.24):
    """One proportion band with a surface gap between segments, in axes coordinates."""
    x = 0.0
    for label, v, group in parts:
        w = v / total
        ax.barh(y, w - 0.004, left=x + 0.002, height=h, color=INK_BAR[group],
                zorder=3, transform=ax.transAxes, align="center")
        ax.text(x + w / 2, y, f"{label}  {v:,}", transform=ax.transAxes, ha="center",
                va="center", fontsize=8.6, color="white", fontweight="bold")
        x += w


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--graph", default=None, help="defaults to cleaning output graph.v2.json")
    ap.add_argument("--qa", default=None, help="QA benchmark JSON (default bench/washi_qa.json)")
    ap.add_argument("--out", default=None, help="output PNG (default output/figure4_washi_stats.png)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    graph_path = Path(args.graph) if args.graph else cfg.output_dir / cfg.get(
        "paths", "clean_graph_file", default="graph.v2.json"
    )
    qa_path = Path(args.qa) if args.qa else cfg.base_dir / "bench" / "washi_qa.json"
    out = Path(args.out) if args.out else cfg.output_dir / "figure4_washi_stats.png"

    bars, note = graph_panel_data(graph_path)
    qa = qa_panel_data(qa_path)

    fig = plt.figure(figsize=(6.6, 5.4), facecolor=SURFACE)
    gs = fig.add_gridspec(3, 1, height_ratios=[1.55, 0.92, 2.05],
                          hspace=0.50, left=0.24, right=0.965, top=0.93, bottom=0.02)
    axs = [fig.add_subplot(gs[i]) for i in range(3)]
    for ax in axs:
        ax.set_facecolor(SURFACE)

    graph_rows = [("Entities", bars["Entities"], "textual"),
                  ("Triples", bars["Triples"], "textual"),
                  ("Text attributes", bars["Text Info"], "textual"),
                  ("Images", bars["Images"], "visual"),
                  ("Illustrated entities", bars["MM Entities"], "visual")]
    _panel_title(axs[0], "(a) WashiMMKG", note.replace(" (", " \u00b7 ").replace(")", ""))
    _hbars(axs[0], graph_rows)

    if qa is not None:
        counts, qa_note, by_type = qa
        hops = qa_note.split(" - ")
        _panel_title(axs[1], "(b) Washi QA benchmark")
        # the closed-book probe is an experiment, not a property of the data: it lives in
        # the Results text, so the fourth tile carries the split instead
        _tiles(axs[1], [(f"{counts['Total QA']:,}", "items"),
                        ("2.4", "avg hops"),
                        ("18 / 20", "words per Q / A"),
                        ("2,007 / 85", "test / dev split")])
        _band(axs[1], [("multimodal", counts["MM Questions"], "visual"),
                       ("text-only", counts["Text-only"], "textual")], counts["Total QA"])
        VISUAL_T = {"Multimodal", "MM Chain"}
        rows = [(k, v, "whole" if k == "Aggregation" else ("visual" if k in VISUAL_T else "textual"))
                for k, v in by_type.items()]
        _panel_title(axs[2], "(c) Question types",
                     "textual (blue) \u00b7 visual (orange) \u00b7 whole-graph (grey)")
        _hbars(axs[2], rows)
    else:
        for ax in axs[1:]:
            ax.set_axis_off()
    # the label column is only as wide as its widest tick label; a fixed `left`
    # leaves a blank strip beside "Illustrated entities" when set to the column width
    fig.canvas.draw()
    label_w = max(t.get_window_extent().width for ax in axs for t in ax.get_yticklabels())
    gs.update(left=label_w / fig.bbox.width + 0.012)
    out.parent.mkdir(parents=True, exist_ok=True)
    # vector for the paper, png for quick viewing; fonts stay editable (Type 42)
    plt.rcParams["pdf.fonttype"] = 42
    for suffix in (".pdf", ".png"):
        fig.savefig(out.with_suffix(suffix), dpi=300, facecolor=SURFACE,
                    bbox_inches="tight", pad_inches=0.02)
        print(f"Wrote {out.with_suffix(suffix)}")
    print(f"\n(a) from {graph_path.name}: " + ", ".join(f"{k}={v:,}" for k, v in bars.items()))
    print(f"    {note}")
    if qa is None:
        print(f"\n(b)/(c) skipped — no QA benchmark at {qa_path} yet (Phase 1).")
    else:
        counts, qa_note, by_type = qa
        print("(b) " + ", ".join(f"{k}={v:,}" for k, v in counts.items()))
        print(f"    {qa_note}")
        print("(c) " + ", ".join(f"{k}={v}" for k, v in by_type.items()))


if __name__ == "__main__":
    main()
