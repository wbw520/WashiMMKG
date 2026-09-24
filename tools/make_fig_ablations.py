"""Draw the ablation figures straight from results/*.json.

Two figures. `fig_sweeps` holds the three continuous sweeps -- evidence budget, search
width and depth, graph completeness -- which read as curves and were a wall of rows as a
table. `fig_bytype` is the per-template accuracy of the main methods and the ablations
that matter, as one heatmap, because most of what the ablations show is *where* accuracy
moves, and that never fits in an aggregate column.

Every number is read from the result file it belongs to, so the figures cannot drift from
the tables the way the prose once did.

    python tools/make_fig_ablations.py --out ../../paper
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

BASE = pathlib.Path(__file__).resolve().parents[1]
# same switch as fill_paper.py, check_paper_numbers.py and rejudge.py, so the figures are
# drawn from whichever measured set the tables were filled from
R = BASE / os.environ.get("WASHI_RESULTS", "results")

PALETTE = {
    "blue_main": "#0F4D92", "blue_secondary": "#3775BA",
    "green_3": "#8BCF8B", "red_strong": "#B64342",
    "neutral": "#CFCECE", "teal": "#42949E", "violet": "#9A4D8E",
}
ACC, PRE, REC = PALETTE["blue_main"], PALETTE["red_strong"], PALETTE["teal"]

TYPES = ["chain", "intersection", "comparative", "temporal", "aggregation", "multimodal", "mm_chain"]
TYPE_LABEL = {"chain": "Chain", "intersection": "Intersection", "comparative": "Comparative",
              "temporal": "Temporal", "aggregation": "Aggregation", "multimodal": "Multimodal",
              "mm_chain": "MM chain"}


def apply_publication_style(font_size: int = 13, axes_linewidth: float = 1.6) -> None:
    plt.rcParams.update({
        "font.family": ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"],
        "font.size": font_size, "axes.labelsize": font_size, "axes.titlesize": font_size + 1,
        "xtick.labelsize": font_size - 1, "ytick.labelsize": font_size - 1,
        "legend.fontsize": font_size - 1, "legend.frameon": False,
        "axes.linewidth": axes_linewidth, "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    })


def finalize_figure(fig, out: pathlib.Path, formats=("pdf", "png"), dpi=300, pad=0.6, tight=True):
    # tight_layout does not see legends anchored outside their axes and overrides any
    # subplots_adjust; figures that place legends in their own axes pass tight=False.
    if tight:
        fig.tight_layout(pad=pad)
    paths = []
    for f in formats:
        p = out.with_suffix(f".{f}")
        fig.savefig(p, dpi=dpi, facecolor="white", bbox_inches=None if tight else "tight", pad_inches=0.15)
        paths.append(p)
    plt.close(fig)
    return paths


def S(name: str) -> dict:
    return json.loads((R / f"{name}.json").read_text())["summary"]


def by_type(name: str) -> list[float]:
    bt = S(name)["by_type"]
    return [bt[t]["acc"] for t in TYPES]


# ------------------------------------------------------------------ sweeps
def fig_sweeps(out: pathlib.Path):
    apply_publication_style(13, 1.6)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.0))

    # (a) evidence budget K. abl-full is the K=16 run, which is now the default.
    ks = [3, 5, 8, 12, 16]
    runs = ["abl-budget3.test", "abl-budget5.test", "abl-budget8.test", "abl-budget12.test", "abl-full.test"]
    ss = [S(r) for r in runs]
    ax = axes[0]
    for key, col, lab in (("acc", ACC, "Accuracy"), ("precision", PRE, "Precision"), ("recall", REC, "Recall")):
        ax.plot(ks, [s[key] for s in ss], "-o", color=col, lw=2.2, ms=6, label=lab)
    ax.axvline(16, color=PALETTE["neutral"], ls="--", lw=1.4, zorder=0)
    ax.text(15.6, 0.31, "reported", color="0.35", fontsize=11, ha="right")
    for k, s in zip(ks, ss):
        ax.annotate(f"{s['acc']:.3f}".lstrip("0"), (k, s["acc"]), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=10, color=ACC)
    ax.set_xticks(ks)
    ax.set_xlabel("Evidence budget $K$ (triples shown to the answerer)")
    ax.set_ylim(0.3, 0.85)
    ax.set_title("(a) Budget $K$ (walk fixed, 21.1 calls)", loc="left")
    ax.legend(loc="lower right")

    # (b) search shape: accuracy against agent calls
    ax = axes[1]
    pts = {"beam 1": "abl-beam1.test", "beam 3": "abl-beam3.test", "beam 5, depth 4": "abl-full.test",
           "depth 2": "abl-depth2.test", "depth 3": "abl-depth3.test"}
    P = {k: (S(v)["calls_mean"], S(v)["acc"]) for k, v in pts.items()}
    for series, col, lab in ((["beam 1", "beam 3", "beam 5, depth 4"], PALETTE["blue_secondary"], "beam width (depth 4)"),
                             (["depth 2", "depth 3", "beam 5, depth 4"], PALETTE["violet"], "max depth (beam 5)")):
        ax.plot([P[k][0] for k in series], [P[k][1] for k in series], "-o", color=col, lw=2.2, ms=7, label=lab)
    ax.plot(*P["beam 5, depth 4"], "o", color="white", mec=ACC, mew=2.2, ms=9, zorder=5)
    offs = {"beam 1": (6, -12), "beam 3": (6, -12), "beam 5, depth 4": (-8, 9),
            "depth 2": (6, -12), "depth 3": (-10, 8)}
    for k, (x, y) in P.items():
        ha = "left" if offs[k][0] > 0 else "right"
        ax.annotate(f"{k}\n{y:.3f}".replace("0.", "."), (x, y), textcoords="offset points",
                    xytext=offs[k], ha=ha, fontsize=9.5, color="0.2")
    ax.set_xlabel("Agent calls per question")
    ax.set_ylabel("Accuracy")
    ax.set_xlim(13, 22)
    ax.set_ylim(0.64, 0.78)
    ax.set_title("(b) Search width and depth", loc="left")
    ax.legend(loc="lower right")

    # (c) graph completeness
    ax = axes[2]
    drops = [0, 10, 25, 50]
    runs = ["abl-full.test", "abl-drop10.test", "abl-drop25.test", "abl-drop50.test"]
    ss = [S(r) for r in runs]
    for key, col, lab in (("acc", ACC, "Accuracy"), ("precision", PRE, "Precision"), ("recall", REC, "Recall")):
        ax.plot(drops, [s[key] for s in ss], "-o", color=col, lw=2.2, ms=6, label=lab)
    for d, s in zip(drops, ss):
        ax.annotate(f"{s['acc']:.3f}".lstrip("0"), (d, s["acc"]), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=10, color=ACC)
    ax.set_xticks(drops)
    ax.set_xticklabels([f"{d}%" for d in drops])
    ax.set_xlabel("Edges removed from the graph")
    ax.set_ylim(0.3, 0.85)
    ax.set_title("(c) Graph completeness", loc="left")
    ax.legend(loc="lower left")

    for a in axes[::2]:
        a.set_ylabel("Score")
    return finalize_figure(fig, out / "fig_sweeps")


# ------------------------------------------------------------------ per-type: methods
def fig_methods_bytype(out: pathlib.Path):
    """Grouped bars: the five retrieval settings on each question type, with the Know+
    ceiling as a tick above each group."""
    apply_publication_style(11, 1.3)
    methods = [("Direct", "direct.test.gemma31", PALETTE["neutral"]),
               ("FMT-RAG", "fmt.test.gemma31", "#E9A6A1"),
               ("GraphRAG", "graph.test.gemma31", PALETTE["red_strong"]),
               ("ReAct", "react.test.gemma31", PALETTE["violet"]),
               ("WikiWalk", "abl-full.test", PALETTE["blue_main"])]
    ceil = by_type("know+.test.gemma31")
    n = S("abl-full.test")["by_type"]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    x = np.arange(len(TYPES)); w = 0.16
    for k, (lab, run, col) in enumerate(methods):
        vals = by_type(run)
        ax.bar(x + (k - 2) * w, vals, w, color=col, edgecolor="black", linewidth=0.6, label=lab, zorder=3)
    for j, c in enumerate(ceil):
        ax.plot([x[j] - 2.6 * w, x[j] + 2.6 * w], [c, c], color="black", lw=1.4, ls=(0, (2, 1.5)), zorder=4,
                label="Know.$^{+}$ ceiling" if j == 0 else None)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{TYPE_LABEL[t]}\n(n={n[t]['n']})" for t in TYPES], fontsize=9.5)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.grid(axis="y", color="0.9", lw=0.8, zorder=0)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.16), fontsize=9.5, handlelength=1.6, columnspacing=1.2)
    return finalize_figure(fig, out / "fig_methods_bytype", pad=0.4)


# ------------------------------------------------------------------ per-type: ablations
SHORT = {"chain": "Chain", "intersection": "Inter.", "comparative": "Comp.", "temporal": "Temp.",
         "aggregation": "Aggr.", "multimodal": "Multi.", "mm_chain": "MMch."}
GROUPS = [
    ("(a) Graph images and search roots",
     [("Full", "abl-full.test"), ("Text-only KG", "abl-text-only-kg.test"), ("no image roots", "abl-no-image-roots.test"),
      ("no mention roots", "abl-no-mention-roots.test"), ("1 root", "abl-roots1.test"), ("3 roots", "abl-roots3.test")]),
    ("(b) Sub-queries, cite/follow, page budget",
     [("Full", "abl-full.test"), ("cite+follow merged", "abl-merge-cite-follow.test"), ("1 sub-query", "abl-subq1.test"),
      ("3 sub-queries", "abl-subq3.test"), ("12 pages", "abl-pages12.test"), ("48 pages", "abl-pages48.test")]),
    ("(c) Search width and depth",
     [("Full (beam 5, depth 4)", "abl-full.test"), ("beam 1", "abl-beam1.test"), ("beam 3", "abl-beam3.test"),
      ("depth 2", "abl-depth2.test"), ("depth 3", "abl-depth3.test")]),
    ("(d) What is kept",
     [("Full", "abl-full.test"), ("no verification", "abl-no-verify.test"),
      ("verify vs. whole question", "abl-verify-question.test"), ("best chain drained", "abl-drain-best-chain.test")]),
    ("(e) Evidence budget $K$",
     [("$K=16$ (Full)", "abl-full.test"), ("$K=3$", "abl-budget3.test"), ("$K=5$", "abl-budget5.test"),
      ("$K=8$", "abl-budget8.test"), ("$K=12$", "abl-budget12.test")]),
    ("(f) Images shown to the answerer",
     [("3 images (Full)", "abl-full.test"), ("1 image", "abl-answer-images1.test"), ("6 images", "abl-answer-images6.test")]),
    # "3 pictures + captions" is gone with the captions: the main configuration now blanks
    # every caption, which makes that variant identical to the 3-picture one.
    ("(g) Pictures shown to the navigator",
     [("none (Full)", "abl-full.test"), ("1 picture", "abl-nav1.test"), ("3 pictures", "abl-nav3.test")]),
    ("(h) Graph completeness",
     [("Full", "abl-full.test"), ("10% edges dropped", "abl-drop10.test"), ("25% dropped", "abl-drop25.test"),
      ("50% dropped", "abl-drop50.test")]),
    ("(i) Image captions",
     [("none (Full)", "abl-full.test"), ("kept, answer-bearing withheld", "abl-keep-captions.test"),
      ("all shown", "abl-all-captions.test")]),
]
CFG_COLS = ["#B64342", "#E07B39", "#42949E", "#9A4D8E", "#7F7F7F"]
CFG_MARK = ["o", "s", "D", "^", "v"]
# one colour per question type for the swept panels: blues/teals for the text templates,
# grey for aggregation, warm for the two visual templates
TYPE_COL = {"chain": "#0F4D92", "intersection": "#3775BA", "comparative": "#42949E", "temporal": "#7FB3D5",
            "aggregation": "#7F7F7F", "multimodal": "#B64342", "mm_chain": "#E07B39"}
TYPE_MARK = {"chain": "o", "intersection": "s", "comparative": "D", "temporal": "^",
             "aggregation": "v", "multimodal": "o", "mm_chain": "s"}
SWEEPS = {
    "(e) Evidence budget $K$": ([3, 5, 8, 12, 16], ["abl-budget3.test", "abl-budget5.test", "abl-budget8.test",
                                                    "abl-budget12.test", "abl-full.test"], "$K$", 16),
    "(f) Images shown to the answerer": ([1, 3, 6], ["abl-answer-images1.test", "abl-full.test", "abl-answer-images6.test"],
                                         "images", 3),
    "(h) Graph completeness": ([0, 10, 25, 50], ["abl-full.test", "abl-drop10.test", "abl-drop25.test", "abl-drop50.test"],
                               "% of edges removed", 0),
}


def _delta_bars(ax, title, cfgs):
    """Change in accuracy against the full configuration, one bar per alternative, grouped
    by question type. The zero line is the full configuration, so nothing is drawn twice."""
    x = np.arange(len(TYPES))
    full = np.array(by_type(cfgs[0][1]))
    alts = cfgs[1:]; k = len(alts); w = 0.8 / k
    ax.axhline(0, color="0.15", lw=1.2, zorder=2)
    for j, (lab, run) in enumerate(alts):
        d = np.array(by_type(run)) - full
        ax.bar(x + (j - (k - 1) / 2) * w, d, w, color=CFG_COLS[j], edgecolor="black", linewidth=0.5, label=lab, zorder=3)
    ax.set_xticks(x); ax.set_xticklabels([SHORT[t] for t in TYPES], fontsize=11)
    ax.set_xlim(-0.55, len(TYPES) - 0.45)
    ax.set_ylim(-0.5, 0.2); ax.set_yticks([-0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2])
    ax.set_yticklabels(["$-$.5", "$-$.4", "$-$.3", "$-$.2", "$-$.1", "0", "+.1", "+.2"])
    ax.set_ylabel("$\\Delta$ accuracy vs. Full")
    ax.grid(axis="y", color="0.92", lw=0.8, zorder=0)
    ax.set_title(title, loc="left", fontsize=13)
    return ax.get_legend_handles_labels()


def _sweep(ax, title, xs, runs, xlabel, x_full):
    M = np.array([by_type(r) for r in runs])           # settings x types
    for k, t in enumerate(TYPES):
        ax.plot(xs, M[:, k], marker=TYPE_MARK[t], ms=6.5, lw=2.0, color=TYPE_COL[t], label=TYPE_LABEL[t], zorder=3)
    ax.axvline(x_full, color="0.75", ls="--", lw=1.2, zorder=1)
    ax.text(x_full, 0.99, " Full", color="0.4", fontsize=10, va="top")
    ax.set_xticks(xs); ax.set_xticklabels([str(x) for x in xs], fontsize=11)
    ax.set_xlabel(xlabel)
    ax.set_ylim(0, 1); ax.set_yticks([0, .2, .4, .6, .8, 1]); ax.set_ylabel("Accuracy")
    ax.grid(axis="y", color="0.92", lw=0.8, zorder=0)
    ax.set_title(title, loc="left", fontsize=13)
    return ax.get_legend_handles_labels()


def _legend_axis(ax, handles, labels, ncol):
    ax.set_axis_off()
    ax.legend(handles, labels, loc="upper center", ncol=ncol, fontsize=10.5, frameon=False,
              handletextpad=0.4, columnspacing=1.2, handlelength=2.0, borderaxespad=0)


def fig_ablation_groups(out: pathlib.Path):
    """Six panels, all lines. Top row: the three groups of discrete design choices that
    move accuracy, as profiles over question type against the full configuration. Bottom
    row: the three ordered sweeps, one line per question type. The two groups in which
    nothing moves (verification; pictures shown to the navigator) stay in the table."""
    apply_publication_style(12, 1.2)
    fig = plt.figure(figsize=(15, 8.6))
    gs = fig.add_gridspec(4, 3, height_ratios=[1, 0.13, 1, 0.08], hspace=0.45, wspace=0.28,
                          left=0.05, right=0.99, top=0.95, bottom=0.03)
    groups = dict(GROUPS)
    top = [("(a) Graph images and search roots", groups["(a) Graph images and search roots"]),
           ("(b) Sub-queries, cite/follow, page budget", groups["(b) Sub-queries, cite/follow, page budget"]),
           ("(c) Search width and depth", groups["(c) Search width and depth"])]
    for c, (title, cfgs) in enumerate(top):
        ax = fig.add_subplot(gs[0, c]); h, l = _delta_bars(ax, title, cfgs)
        _legend_axis(fig.add_subplot(gs[1, c]), h, l, ncol=3)
    sweeps = [("(d) Evidence budget $K$", SWEEPS["(e) Evidence budget $K$"]),
              ("(e) Images shown to the answerer", SWEEPS["(f) Images shown to the answerer"]),
              ("(f) Graph completeness", SWEEPS["(h) Graph completeness"])]
    for c, (title, (xs, runs, xlabel, x_full)) in enumerate(sweeps):
        ax = fig.add_subplot(gs[2, c]); h, l = _sweep(ax, title, xs, runs, xlabel, x_full)
    _legend_axis(fig.add_subplot(gs[3, :]), h, l, ncol=7)
    return finalize_figure(fig, out / "fig_ablation_groups", tight=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(BASE / "output"))
    a = ap.parse_args()
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for p in fig_sweeps(out) + fig_methods_bytype(out) + fig_ablation_groups(out):
        print("wrote", p)


if __name__ == "__main__":
    main()
