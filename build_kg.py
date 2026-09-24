#!/usr/bin/env python3
"""WashiMMKG builder CLI.

Usage:
    python build_kg.py build          # incremental: process new/changed files in raw/
    python build_kg.py build --force  # reprocess every file (graph still merges, not reset)
    python build_kg.py rebuild        # wipe the graph + manifest, then build from scratch
    python build_kg.py watch          # keep running; auto-expand when files are added to raw/
    python build_kg.py stats          # print current graph statistics
    python build_kg.py export --format graphml   # export to output/graph.graphml

Drop .txt / .md / .pdf and image files into raw/. Re-running expands the graph.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from washi_kg import load_config
from washi_kg.graph import KnowledgeGraph
from washi_kg.ingest import iter_source_files
from washi_kg.pipeline import Pipeline
from washi_kg.state import Manifest, file_hash


def _scan_status(cfg):
    """Return (rel_path, kind, status) for every file in raw/.

    status: 'processed' | 'NEW' (never processed) | 'CHANGED' (processed but edited).
    """
    m = Manifest(cfg.manifest_path)
    exts = set(cfg.get("creator", "image_exts", default=[]))
    rows = []
    for path, kind in iter_source_files(cfg.raw_dir, exts):
        rel = str(path.relative_to(cfg.raw_dir))
        digest = file_hash(path)
        if m.is_processed(rel, digest):
            status = "processed"
        elif rel in m.entries:
            status = "CHANGED"
        else:
            status = "NEW"
        rows.append((rel, kind, status))
    return rows


def cmd_build(args) -> None:
    cfg = load_config(args.config)
    Pipeline(cfg).build(force=args.force)


def cmd_rebuild(args) -> None:
    cfg = load_config(args.config)
    for p in (cfg.graph_path, cfg.manifest_path):
        if p.exists():
            p.unlink()
            print(f"removed {p}")
    Pipeline(cfg).build(force=True)


def cmd_stats(args) -> None:
    cfg = load_config(args.config)
    kg = KnowledgeGraph.load(cfg.graph_path)
    s = kg.stats()
    if not kg.entities:
        print("Empty graph. Add files to raw/ and run: python build_kg.py build")
        return
    print(f"Graph file: {cfg.graph_path}")
    for k, v in s.items():
        print(f"  {k:>22}: {v}")
    top = sorted(kg.entities.values(), key=lambda e: e.get("degree", 0), reverse=True)[:10]
    if any(e.get("degree") for e in top):
        print("  top entities by degree:")
        for e in top:
            print(f"    {e['degree']:>4}  {e['name']}")


def cmd_status(args) -> None:
    cfg = load_config(args.config)
    rows = _scan_status(cfg)
    if not rows:
        print(f"No files in {cfg.raw_dir}. Add files and run: python build_kg.py build")
        return
    pending = [r for r in rows if r[2] != "processed"]
    print(f"{'STATUS':10} {'KIND':6} FILE")
    for rel, kind, status in rows:
        print(f"{status:10} {kind:6} {rel}")
    print(f"\n{len(rows)} file(s): {len(rows) - len(pending)} processed, {len(pending)} pending.")
    if pending:
        print("Run `python build_kg.py build` to process: " + ", ".join(r[0] for r in pending))


def _safe_name(name: str) -> str:
    import re

    s = re.sub(r'[\\/:*?"<>|#^\[\]]', " ", name).strip()
    return re.sub(r"\s+", " ", s) or "entity"


def _export_obsidian(cfg, kg) -> None:
    """Write an Obsidian vault: one note per entity, triples as [[wikilinks]],
    A_text as body, images embedded with ![[...]]. Open the folder as a vault and
    use Graph View."""
    import shutil
    from collections import defaultdict

    vault = cfg.output_dir / "obsidian_vault"
    assets = vault / "assets"
    assets.mkdir(parents=True, exist_ok=True)

    out_rel = defaultdict(list)  # name -> [(relation, tail)]
    in_rel = defaultdict(list)   # name -> [(head, relation)]
    for t in kg.triples:
        out_rel[t["h"]].append((t["r"], t["u"]))
        in_rel[t["u"]].append((t["h"], t["r"]))

    # One unique filename per entity. Case-insensitive filesystems (macOS, Windows)
    # shadow files that differ only by case, so entities like the technique "rakusui"
    # and the sample "Rakusui" would overwrite each other. Disambiguate the later one
    # with its type (e.g. "Rakusui (washi sample)") so both notes survive.
    fname_of: dict[str, str] = {}
    used_lc: dict[str, str] = {}  # lowercased filename -> owning entity
    for name, ent in kg.entities.items():
        cand = _safe_name(name)
        if cand.lower() in used_lc:
            t = ent.get("type")
            cand = f"{_safe_name(name)} ({t})" if t else f"{_safe_name(name)} (2)"
            i = 1
            while cand.lower() in used_lc:
                i += 1
                cand = f"{_safe_name(name)} ({i})"
        used_lc[cand.lower()] = name
        fname_of[name] = cand

    def link(name: str) -> str:
        fn = fname_of.get(name) or _safe_name(name)
        return f"[[{fn}]]" if fn == name else f"[[{fn}|{name}]]"

    n_notes = 0
    for name, ent in kg.entities.items():
        safe = fname_of[name]
        lines: list[str] = []

        # frontmatter (aliases let [[original name]] resolve even if the file is renamed)
        aliases = list(ent.get("aliases", []))
        if safe != name:
            aliases = [name] + aliases
        lines.append("---")
        if ent.get("type"):
            lines.append(f"type: {ent['type']}")
        lines.append(f"degree: {ent.get('degree', 0)}")
        if aliases:
            esc = ", ".join('"' + a.replace('"', "'") + '"' for a in aliases)
            lines.append(f"aliases: [{esc}]")
        lines.append("---\n")

        lines.append(f"# {name}\n")
        for txt in ent["attributes"]["text"]:
            lines.append(txt + "\n")

        if out_rel.get(name):
            lines.append("## Relations")
            for r, u in out_rel[name]:
                lines.append(f"- {r} → {link(u)}")
            lines.append("")
        if in_rel.get(name):
            lines.append("## Mentioned by")
            for h, r in in_rel[name]:
                lines.append(f"- {link(h)} ({r})")
            lines.append("")

        if ent["attributes"]["img"]:
            lines.append("## Images")
            for im in ent["attributes"]["img"]:
                src = (cfg.base_dir / im["path"])
                if src.exists():
                    dest_name = im["path"].replace("/", "_").replace("\\", "_")
                    shutil.copyfile(src, assets / dest_name)
                    lines.append(f"![[assets/{dest_name}]]")
                cap = im.get("caption") or ""
                meta = []
                if im.get("source"):
                    meta.append(im["source"] + (f" p.{im['page']}" if im.get("page") else ""))
                suffix = f" _( {'; '.join(meta)} )_" if meta else ""
                if cap or suffix:
                    lines.append(f"*{cap}*{suffix}\n")

        (vault / f"{safe}.md").write_text("\n".join(lines), encoding="utf-8")
        n_notes += 1

    print(f"wrote Obsidian vault: {vault}  ({n_notes} notes, images in assets/)")
    print("Open it in Obsidian: 'Open folder as vault' -> this folder, then open Graph View.")


def cmd_export(args) -> None:
    cfg = load_config(args.config)
    kg = KnowledgeGraph.load(cfg.graph_path)
    if not kg.triples:
        print("Nothing to export — graph is empty.")
        return
    if args.format == "obsidian":
        _export_obsidian(cfg, kg)
        return
    if args.format == "graphml":
        import networkx as nx

        g = nx.MultiDiGraph()
        for name, e in kg.entities.items():
            g.add_node(
                name,
                degree=e.get("degree", 0),
                n_images=len(e["attributes"]["img"]),
                multimodal=bool(e["attributes"]["img"]),
            )
        for t in kg.triples:
            g.add_edge(t["h"], t["u"], key=t["r"], relation=t["r"])
        out = cfg.output_dir / "graph.graphml"
        nx.write_graphml(g, out)
        print(f"wrote {out}  ({g.number_of_nodes()} nodes, {g.number_of_edges()} edges)")
    else:
        print(f"unknown format: {args.format}")


def cmd_watch(args) -> None:
    cfg = load_config(args.config)
    pipeline = Pipeline(cfg)
    print(f"Watching {cfg.raw_dir} every {args.interval}s. Ctrl-C to stop.")
    pipeline.build()  # initial pass

    def snapshot() -> set[tuple[str, str]]:
        snap = set()
        for path, _ in iter_source_files(cfg.raw_dir, set(cfg.get("creator", "image_exts", default=[]))):
            rel = str(path.relative_to(cfg.raw_dir))
            snap.add((rel, file_hash(path)))
        return snap

    last = snapshot()
    try:
        while True:
            time.sleep(args.interval)
            cur = snapshot()
            if cur != last:
                print("\nDetected changes in raw/ — expanding graph...")
                pipeline.build()
                last = snapshot()
    except KeyboardInterrupt:
        print("\nstopped.")


def main() -> None:
    p = argparse.ArgumentParser(description="Build the WashiMMKG from files in raw/.")
    p.add_argument("--config", default=None, help="path to config.yaml")
    sub = p.add_subparsers(dest="command", required=True)

    b = sub.add_parser("build", help="incrementally process new/changed files")
    b.add_argument("--force", action="store_true", help="reprocess all files")
    b.set_defaults(func=cmd_build)

    r = sub.add_parser("rebuild", help="wipe graph + manifest, then build from scratch")
    r.set_defaults(func=cmd_rebuild)

    s = sub.add_parser("stats", help="print graph statistics")
    s.set_defaults(func=cmd_stats)

    st = sub.add_parser("status", help="show which raw/ files are processed / new / changed")
    st.set_defaults(func=cmd_status)

    e = sub.add_parser("export", help="export the graph")
    e.add_argument("--format", default="graphml", choices=["graphml", "obsidian"])
    e.set_defaults(func=cmd_export)

    w = sub.add_parser("watch", help="auto-expand when files are added to raw/")
    w.add_argument("--interval", type=int, default=10, help="poll interval in seconds")
    w.set_defaults(func=cmd_watch)

    args = p.parse_args()
    try:
        args.func(args)
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()
