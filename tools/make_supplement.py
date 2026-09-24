"""Generate supplementary.tex from the result files.

The supplement exists because the main paper reports 7 of 30 ablations, states four times
that the configuration was "selected on the development split" without showing the
selection, and prints no prompt -- three things a reader cannot check and a reviewer will
ask about.

It is generated rather than written. Every number here comes from the same files that fill
the main tables, so the supplement cannot drift from them the way prose did twice during
this project. Run it whenever the results change:

    WASHI_RESULTS=results/final python tools/make_supplement.py
"""
from __future__ import annotations

import collections
import glob
import json
import math
import os
import pathlib
import re

BASE = pathlib.Path(__file__).resolve().parents[1]
R = BASE / os.environ.get("WASHI_RESULTS", "results")
DEV = BASE / "results/nocap/devsweep"
OUT = BASE.parent.parent / "paper" / "supplementary.tex"

TYPES = ["chain", "comparative", "intersection", "aggregation", "temporal",
         "multimodal", "mm_chain"]
TYPE_SHORT = {"chain": "Chain", "comparative": "Comp.", "intersection": "Inter.",
              "aggregation": "Aggr.", "temporal": "Temp.", "multimodal": "Multi.",
              "mm_chain": "MMch."}
BACKBONES = [("Qwen3.5-9B", "qwen9"), ("Gemma-4-12B", "gemma12"),
             ("Qwen3.5-27B", "qwen27"), ("Qwen3.8-27B", "qwen38"),
             ("Gemma-4-31B", "gemma31")]
METHODS = [("Direct", "direct"), ("Know.", "know"), ("Know.$^+$", "know+"),
           ("FMT-RAG", "fmt"), ("GraphRAG", "graph"), ("ReAct", "react"),
           ("WikiWalk", "wikiwalk")]

# readable names for the ablation tags, grouped as they are discussed in the paper
ABL_GROUPS = [
    ("Graph images and search roots", [
        ("text-only-kg", "text-only graph (no images)"),
        ("no-image-roots", "no image roots"),
        ("no-mention-roots", "no mention roots"),
        ("roots1", "1 root per sub-query"),
        ("roots3", "3 roots per sub-query")]),
    ("Search width and depth", [
        ("beam1", "beam width 1"), ("beam3", "beam width 3"),
        ("depth2", "max depth 2"), ("depth3", "max depth 3")]),
    ("Walk organisation", [
        ("subq1", "1 query walker"), ("subq3", "3 query walkers"),
        ("merge-cite-follow", "cite and follow merged"),
        ("pages12", "page budget 12"), ("pages48", "page budget 48")]),
    ("Verification and merging", [
        ("no-verify", "no chain verification"),
        ("verify-question", "verify against the whole question"),
        ("drain-best-chain", "best chain drained, not cycled")]),
    ("Evidence budget", [
        ("budget3", "$K=3$"), ("budget5", "$K=5$"),
        ("budget8", "$K=8$"), ("budget12", "$K=12$")]),
    ("Images and captions", [
        ("answer-images1", "1 image to the answerer"),
        ("answer-images6", "6 images to the answerer"),
        ("nav1", "1 picture to the navigator"),
        ("nav3", "3 pictures to the navigator"),
        ("keep-captions", "captions kept, answer withheld"),
        ("all-captions", "all captions shown")]),
    ("Graph completeness", [
        ("drop10", "10\\% of edges removed"), ("drop25", "25\\% removed"),
        ("drop50", "50\\% removed")]),
]


def rows(path: pathlib.Path) -> list[dict]:
    return json.loads(path.read_text())["rows"]


def mean(path: pathlib.Path, field: str) -> float | None:
    rs = [r for r in rows(path) if r.get(field) is not None]
    if not rs:
        return None
    return sum(bool(r[field]) if field == "correct" else r[field] for r in rs) / len(rs)


def correct_map(path: pathlib.Path) -> dict[str, bool]:
    return {r["qid"]: bool(r["correct"]) for r in rows(path) if r.get("correct") is not None}


def paired_z(a: dict[str, bool], b: dict[str, bool]) -> float:
    """z of the paired difference b - a over the items both answered."""
    q = set(a) & set(b)
    up = sum(1 for k in q if b[k] and not a[k])
    down = sum(1 for k in q if a[k] and not b[k])
    return (up - down) / math.sqrt(up + down) if up + down else 0.0


def by_type(path: pathlib.Path) -> dict[str, float]:
    acc = collections.defaultdict(lambda: [0, 0])
    for r in rows(path):
        if r.get("correct") is None:
            continue
        g = acc[r.get("type")]
        g[0] += bool(r["correct"])
        g[1] += 1
    return {k: v[0] / v[1] for k, v in acc.items() if v[1]}


def f4(x: float) -> str:
    return f"{x:.4f}".lstrip("0")


def f3(x: float) -> str:
    return f"{x:.3f}".lstrip("0")


def signed(x: float) -> str:
    return ("$+$" if x >= 0 else "$-$") + f"{abs(x):.4f}".lstrip("0")


# ------------------------------------------------------------------ S1 dev selection
def sec_dev() -> str:
    pat = re.compile(r"dev-b(\d+)d(\d+)k(\d+)\.json$")
    entries = []
    for p in sorted(DEV.glob("dev-*.json")):
        m = pat.search(p.name)
        if not m:
            continue
        beam, depth, k = (int(g) for g in m.groups())
        cm = correct_map(p)
        entries.append((beam, depth, k, sum(cm.values()) / len(cm),
                        mean(p, "recall"), mean(p, "n_calls"), cm))
    if not entries:
        return ""
    base = next(e for e in entries if (e[0], e[1], e[2]) == (3, 3, 8))
    n = len(base[6])
    lines = [
        "\\section{Selection of the search configuration}\\label{sec:s1}",
        "",
        "The beam width, search depth and evidence budget reported in the main paper were",
        "chosen on the development split of %d items, before any of them was run on the" % n,
        "test split. Table~\\ref{tab:s_dev} gives every configuration tried. Accuracies on",
        "%d items carry a standard error near $0.05$, so the column to read is not the" % n,
        "ranking but the paired comparison against the initial configuration: each variant",
        "answers the same items, and $z$ is computed from the items whose outcome differs.",
        "Every beam-5 setting is distinguishable from the initial one and no beam-3 setting",
        "is, which is what the choice rests on.",
        "",
        "\\begin{table}[h]",
        "\\centering",
        "\\caption{Search configurations on the development split (%d items, Gemma-4-31B)." % n,
        "$z$: paired difference against beam 3, depth 3, $K=8$, the configuration used before",
        "this selection. The reported configuration is marked $\\star$.}",
        "\\label{tab:s_dev}",
        "\\begin{tabular*}{\\columnwidth}{@{\\extracolsep{\\fill}}ccccccc@{}}",
        "\\toprule",
        "\\textbf{Beam} & \\textbf{Depth} & $\\mathbf{K}$ & \\textbf{Acc} & \\textbf{Re.} &"
        " \\textbf{Calls} & $\\mathbf{z}$ \\\\",
        "\\midrule",
    ]
    for beam, depth, k, acc, rec, calls, cm in sorted(entries, key=lambda e: -e[3]):
        star = "$\\star$" if (beam, depth, k) == (5, 4, 16) else ""
        z = paired_z(base[6], cm)
        zs = "---" if (beam, depth, k) == (3, 3, 8) else f"{z:.2f}"
        lines.append(f"{beam}{star} & {depth} & {k} & {f4(acc)} & {f3(rec)} & "
                     f"{calls:.1f} & {zs} \\\\")
    lines += ["\\bottomrule", "\\end{tabular*}", "\\end{table}", ""]
    return "\n".join(lines)


# ------------------------------------------------------------------ S2 full ablations
def sec_ablations() -> str:
    full = R / "abl-full.test.json"
    fm = correct_map(full)
    fa = sum(fm.values()) / len(fm)
    lines = [
        "\\section{Complete ablation results}\\label{sec:s2}",
        "",
        "Table~\\ref{tab:s_abl} lists every ablation measured, including those that appear",
        "in the main paper only as points in a figure. All use Gemma-4-31B on the %d test" % len(fm),
        "items with the fixed judge, and differ from the reported configuration in one",
        "setting each. $\\Delta$ is the paired accuracy difference against the first row and",
        "$z$ its paired statistic; $|z| < 1.96$ marks a difference this experiment cannot",
        "distinguish from zero.",
        "",
        "\\begin{table}[h]",
        "\\centering",
        "\\caption{All ablations of the reported configuration (Gemma-4-31B, test split,",
        "beam 5, depth 4, $K=16$).}",
        "\\label{tab:s_abl}",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabular*}{\\columnwidth}{@{\\extracolsep{\\fill}}lcccccc@{}}",
        "\\toprule",
        "\\textbf{Configuration} & \\textbf{Acc} & \\textbf{Pre.} & \\textbf{Re.} &"
        " \\textbf{Calls} & $\\mathbf{\\Delta}$ & $\\mathbf{z}$ \\\\",
        "\\midrule",
        f"Reported configuration & {f4(fa)} & {f3(mean(full,'precision'))} & "
        f"{f3(mean(full,'recall'))} & {mean(full,'n_calls'):.1f} & --- & --- \\\\",
    ]
    for group, tags in ABL_GROUPS:
        lines.append("\\midrule")
        lines.append(f"\\multicolumn{{7}}{{l}}{{\\textit{{{group}}}}} \\\\")
        for tag, label in tags:
            p = R / f"abl-{tag}.test.json"
            if not p.exists():
                continue
            cm = correct_map(p)
            acc = sum(cm.values()) / len(cm)
            z = paired_z(fm, cm)
            dag = "$^{\\dagger}$" if abs(z) < 1.96 else ""
            lines.append(
                f"\\quad {label} & {f4(acc)} & {f3(mean(p,'precision'))} & "
                f"{f3(mean(p,'recall'))} & {mean(p,'n_calls'):.1f} & "
                f"{signed(acc-fa)}{dag} & {z:.2f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular*}", "\\end{table}", ""]
    return "\n".join(lines)


# ------------------------------------------------------------------ S3 per-template
def sec_bytype() -> str:
    lines = [
        "\\section{Accuracy by question type for every backbone}\\label{sec:s3}",
        "",
        "Figure~5 of the main paper shows the per-template breakdown for Gemma-4-31B only.",
        "Table~\\ref{tab:s_bytype} gives it for all five backbones, which is the evidence",
        "behind the statement that WikiWalk's gains fall on the templates needing more than",
        "one route and the two visual templates.",
        "",
        "\\begin{table*}[h]",
        "\\centering",
        "\\caption{Accuracy by question type, all backbones and methods (test split, fixed judge).}",
        "\\label{tab:s_bytype}",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\footnotesize",
        "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}ll" + "c" * len(TYPES) + "@{}}",
        "\\toprule",
        "\\textbf{Backbone} & \\textbf{Method} & "
        + " & ".join(f"\\textbf{{{TYPE_SHORT[t]}}}" for t in TYPES) + " \\\\",
    ]
    for name, tag in BACKBONES:
        lines.append("\\midrule")
        for i, (label, m) in enumerate(METHODS):
            p = R / f"{m}.test.{tag}.json"
            if not p.exists():
                continue
            bt = by_type(p)
            head = f"\\multirow{{{len(METHODS)}}}{{*}}{{{name}}}" if i == 0 else ""
            lines.append(head + " & " + label + " & "
                         + " & ".join(f3(bt.get(t, 0.0)) for t in TYPES) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular*}", "\\end{table*}", ""]
    return "\n".join(lines)


# ------------------------------------------------------------------ S4 prompts
TEX_ESC = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
           "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
           "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}


def esc(text: str) -> str:
    out = "".join(TEX_ESC.get(c, c) for c in text)
    # the prompts use Markdown emphasis, which would otherwise print as literal asterisks
    return re.sub(r"[*]{2}(.+?)[*]{2}", '\\textbf{\\1}', out)


def render_prompt(body: str) -> list[str]:
    """Set a prompt so it can be read rather than merely reproduced.

    These prompts are English prose with a few structured lines in them, and verbatim
    treats both alike: every line monospaced and hard-wrapped at a fixed column, which is
    the right choice for code and the wrong one for paragraphs. Prose is set as prose and
    flows; the JSON shapes and the indented action list stay monospaced, because in those
    the alignment carries meaning.
    """
    out: list[str] = ["\\begin{framed}\\small"]
    for block in re.split(r"\n\s*\n", body.strip()):
        lines = block.split("\n")
        structured = any(l.startswith((" ", "\t")) for l in lines) or "{" in block
        if structured:
            # \footnotesize verbatim fits about 86 columns in this column width; the JSON
            # shapes run past that and would otherwise hang into the margin
            out += ["{\\footnotesize\\begin{verbatim}"]
            for l in lines:
                while len(l) > 86:
                    cut = l.rfind(" ", 0, 86)
                    cut = cut if cut > 40 else 86
                    out.append(l[:cut])
                    l = "    " + l[cut:].lstrip()
                out.append(l)
            out += ["\\end{verbatim}}"]
        else:
            out += [esc(" ".join(l.strip() for l in lines)), ""]
    out.append("\\end{framed}")
    return out


# ------------------------------------------------------------------ S4 the vault
def sec_vault() -> str:
    """Two screenshots of the graph as a reader can browse it.

    The second one matters more than it looks: the entity note is the unit of observation
    the navigator is given, so the figure shows literally what the model reads at each
    step. The vault heading is "Relations" while page.py renders the same block as
    "Facts", and the caption says so rather than letting a reader find the discrepancy.
    """
    figs = []
    here = OUT.parent
    if (here / "WASHIGraph.png").exists():
        figs.append("\n".join([
            "\\begin{figure*}[h]",
            "\\centering",
            "\\includegraphics[width=0.92\\textwidth]{WASHIGraph.png}",
            "\\caption{WashiMMKG browsed as an Obsidian vault: one note per entity, one",
            "link per triple. Node area grows with degree, so the hubs the sampler refuses",
            "as intermediate nodes are the large discs --- \\textit{washi}, \\textit{kozo},",
            "\\textit{mitsumata}, \\textit{Echizen hosho} --- and the long tail of",
            "degree-one entities, which the cleaning pass deliberately keeps, is the",
            "surrounding field of small ones. The view is the whole graph at one zoom",
            "level; labels are legible only where the layout allows.}",
            "\\label{fig:s_vault}",
            "\\end{figure*}", ""]))
    if (here / "Page.png").exists():
        figs.append("\n".join([
            "\\begin{figure}[h]",
            "\\centering",
            "\\includegraphics[width=0.86\\columnwidth]{Page.png}",
            "\\caption{The note for \\textit{mitsumata}, which is the unit of observation the",
            "navigator is given. Outgoing triples are listed with their relation and",
            "incoming ones are grouped under \\textsf{Mentioned by}, so a page states both",
            "what an entity asserts and who points at it; the walk cites from the first",
            "block and follows links from either. The vault heading reads",
            "\\textsf{Relations} where the retrieval-time renderer writes \\textsf{Facts};",
            "the content is the same. Any images attached to the entity appear on the page",
            "and are withheld or shown according to the protocol in the main text.}",
            "\\label{fig:s_page}",
            "\\end{figure}", ""]))
    if not figs:
        return ""
    return "\n".join([
        "\\section{The knowledge graph as a browsable vault}\\label{sec:s4}",
        "",
        "WashiMMKG is exported as a set of Markdown notes, one per entity, which can be",
        "read and navigated directly. The same export is the retrieval interface: the",
        "walk moves between these notes rather than over an abstract edge list, which is",
        "why a page is the unit of observation in the method.",
        ""] + figs)


def sec_prompts() -> str:
    src = (BASE / "wikirag" / "prompts.py").read_text()
    wanted = [("DECOMPOSE_SYSTEM", "Sub-query decomposition and mention extraction"),
              ("NAVIGATE_SYSTEM", "Navigation: citing and following"),
              ("VERIFY_SYSTEM", "Chain verification"),
              ("ANSWER_SYSTEM", "Answering"),
              ("REACT_SYSTEM", "ReAct baseline"),
              ("REACT_IMAGE_ACTION", "ReAct baseline: the image-search action")]
    lines = ["\\section{Prompts}\\label{sec:s5}", "",
             "Each prompt below is reproduced verbatim from the implementation. Placeholders",
             "in braces are filled at run time; the structured blocks are shown as the model",
             "receives them, and the surrounding instructions as running text.", ""]
    for name, title in wanted:
        m = re.search(name + r' = """\\\n(.*?)"""', src, re.S)
        if not m:
            continue
        lines += [f"\\subsection{{{title}}}"] + render_prompt(m.group(1).rstrip()) + [""]
    return "\n".join(lines)



def main() -> None:
    parts = [
        "%% Generated by tools/make_supplement.py -- do not edit by hand.",
        "%% Every number is read from the result files that fill the main tables.",
        "\\documentclass[pdflatex,sn-mathphys-num]{sn-jnl}",
        "\\usepackage{amsmath}",
        "\\usepackage{graphicx}",
        "\\usepackage{framed}",
        "\\usepackage{multirow}",
        "\\usepackage{booktabs}",
        # \jyear is not defined in this release of sn-jnl.cls, and \maketitle needs an
        # author; the supplement carries neither, so the title is set by hand.
        "\\begin{document}",
        "",
        "\\begin{center}",
        "{\\LARGE\\bfseries Supplementary Information}\\\\[2pt]",
        "{\\large Enhancing Washi Reasoning in MLLMs via Multimodal Knowledge Graphs",
        "and Agentic Retrieval-Augmented Generation}",
        "\\end{center}",
        "\\vspace{1em}",
        "\\renewcommand{\\thesection}{S\\arabic{section}}",
        "\\renewcommand{\\thetable}{S\\arabic{table}}",
        "\\setcounter{table}{0}",
        "",
        sec_dev(), sec_ablations(), sec_bytype(), sec_vault(), sec_prompts(),
        "\\end{document}",
    ]
    OUT.write_text("\n".join(p for p in parts if p) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
