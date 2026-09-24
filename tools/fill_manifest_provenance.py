"""Fill source_url / source_page in the image manifest from provenance we already hold.

Two gaps were left after the byte-exact crawl:

  * Wikimedia Commons images have no URL in the manifest even though `source` names the
    exact file ("Wikimedia Commons: IseWashidrop.jpg (CC BY-SA 4.0)"). The Commons file
    page is the canonical thing to cite for attribution, and it is reconstructable.
  * Third-party images whose bytes no longer match anything on the live site still have a
    known origin: every raw/*.md records the page it was fetched from in its header. That
    is the page-level provenance a reader needs, even when the file URL has since changed.

    python tools/fill_manifest_provenance.py [--write]
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import pathlib
import re
import urllib.parse

BASE = pathlib.Path(__file__).resolve().parents[1]
RAW = BASE / "raw"
MANIFEST = BASE.parent.parent / "release" / "washi-data" / "images" / "image_manifest.csv"
# written by tools/verify_commons_images.py: the Commons file each image really is
VERIFIED = BASE / "output" / "commons_final.json"
# byte-exact addresses found in the Wayback Machine before that search was stopped
WAYBACK = BASE / "output" / "image_url_wayback.json"

# `source` for our own group names a seminar or a collaborator's document, which does not
# say who made the picture. These notes do; they follow what the authors stated about each
# group and are written per row so the manifest is self-describing.
OWN_NOTES = [
    (re.compile(r"(ゼミ|発表会|zemi)", re.I),
     "Recorded by the authors at laboratory seminars and presentations."),
    (re.compile(r"^echizen_wasi", re.I),
     "Produced by the Echizen washi collaborators and provided to the authors for research use."),
    (re.compile(r"^washi_mode$", re.I),
     "Collected by the authors."),
]

COMMONS = re.compile(r"Wikimedia Commons:\s*(.+)", re.I)
# the license is the LAST parenthesised group; file names themselves may contain
# parentheses ("Hasegawa Tohaku - Pine Trees (Shorin-zu byobu).jpg")
LICENSE_TAIL = re.compile(r"\s*\((?:CC|PD|Public|GFDL|Copyright|Fair)[^()]*\)\s*$", re.I)
# some raw files list the Commons file per image instead:  "- ise_drop.jpg -- ... (File:IseWashidrop.jpg)"
PER_FILE = re.compile(r"^-\s*([\w.-]+\.(?:jpg|jpeg|png|gif|webp|JPG|PNG))\b(.*?)(?=\n-\s|\Z)",
                      re.S | re.M)
FILE_TAG = re.compile(r"\(File:([^;)]+?)\)|File:([^;)\n]+?)[;)]")


def commons_url(source: str) -> str | None:
    m = COMMONS.search(source)
    if not m:
        return None
    name = LICENSE_TAIL.sub("", m.group(1)).strip().replace(" ", "_")
    return "https://commons.wikimedia.org/wiki/File:" + urllib.parse.quote(name, safe="_-.()'!,")


def commons_by_basename() -> dict[str, str]:
    """Commons file page per image basename, for raw files that list them inline."""
    out: dict[str, str] = {}
    for f in sorted(glob.glob(str(RAW / "*.md"))):
        text = pathlib.Path(f).read_text(encoding="utf-8", errors="ignore")
        if "File:" not in text:
            continue
        for base, body in PER_FILE.findall(text):
            m = FILE_TAG.search(body)
            if not m:
                continue
            name = (m.group(1) or m.group(2)).strip().replace(" ", "_")
            out[base] = ("https://commons.wikimedia.org/wiki/File:"
                         + urllib.parse.quote(name, safe="_-.()'!,"))
    return out


def per_image_pages() -> dict[str, str]:
    """Some raw files record the origin per image rather than once in the header, and not
    always with a scheme ("source: FBC/NTV News, news.ntv.co.jp/n/fbc/...")."""
    bare = re.compile(
        r"(?:https?://"
        r"|(?<![\w.])(?=(?:[a-z0-9-]+\.)+[a-z]{2,}/))"   # scheme-less host, any depth
        r"[^\s)\"'<>]+")
    out: dict[str, str] = {}
    for f in sorted(glob.glob(str(RAW / "*.md"))):
        text = pathlib.Path(f).read_text(encoding="utf-8", errors="ignore")
        for base, body in PER_FILE.findall(text):
            flat = " ".join(body.split())
            m = bare.search(flat)
            if m:
                u = m.group(0).rstrip(".,;")
                out[base] = u if u.startswith("http") else "https://" + u
    return out


def page_urls() -> dict[str, str]:
    """basename -> the page each raw markdown file was fetched from."""
    out: dict[str, str] = {}
    for f in sorted(glob.glob(str(RAW / "*.md"))):
        text = pathlib.Path(f).read_text(encoding="utf-8", errors="ignore")
        head = text[:600]
        m = re.search(r"https?://[^\s)\"'<>]+", head) or re.search(r"https?://[^\s)\"'<>]+", text)
        if m:
            out[os.path.basename(f)] = m.group(0).rstrip(".,;")
            continue
        # some headers write the host bare inside parentheses on the title line
        m2 = re.search(r"\(([a-z0-9.-]+\.[a-z]{2,}/[^\s)]*)\)", head)
        if m2:
            out[os.path.basename(f)] = "https://" + m2.group(1)
    return out


def lookup(source: str, pages: dict[str, str]) -> str | None:
    """`source` is either a raw markdown filename or the short folder name it produced."""
    s = source.strip()
    for cand in (s, f"{s}.md", f"{s}_web.md", f"{s}_images_web.md"):
        if cand in pages:
            return pages[cand]
    stem = s.removesuffix(".md")
    for name, url in pages.items():
        if name.startswith(stem + "_") or name.removesuffix(".md") == stem:
            return url
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    pages = page_urls()
    by_base = commons_by_basename()
    per_img = per_image_pages()
    verified = json.loads(VERIFIED.read_text()) if VERIFIED.exists() else {}
    wayback = json.loads(WAYBACK.read_text()) if WAYBACK.exists() else {}
    rows = list(csv.DictReader(MANIFEST.open()))
    fields = list(rows[0].keys())
    if "rights_note" not in fields:
        fields.append("rights_note")
        for r in rows:
            r.setdefault("rights_note", "")

    filled_url = filled_page = fixed_attr = noted = 0
    still: list[str] = []
    for r in rows:
        if r["rights"] == "own_or_collaborator" and not r.get("rights_note", "").strip():
            for pat, note in OWN_NOTES:
                if pat.search(r["source"].strip()):
                    r["rights_note"] = note
                    noted += 1
                    break
        v = verified.get(os.path.basename(r["path"]))
        if v and r["rights"] == "wikimedia_commons":
            # the recorded Commons file name was written when the graph was built, not by a
            # downloader; 18 of 42 named a page that does not exist. Cite the verified one.
            attr = f"Wikimedia Commons: {v['title'].removeprefix('File:')} ({v['license']})"
            if v.get("artist"):
                attr = (f"Wikimedia Commons: {v['title'].removeprefix('File:')} "
                        f"by {v['artist']} ({v['license']})")
            if attr != r["source"]:
                fixed_attr += 1
            r["source"], r["source_url"] = attr, v["url"]
            filled_url += 1
            continue
        if not r["source_url"].strip() and r["rights"] == "wikimedia_commons":
            u = commons_url(r["source"]) or by_base.get(os.path.basename(r["path"]))
            if u:
                r["source_url"] = u
                filled_url += 1
        if not r["source_url"].strip() and r["path"] in wayback:
            w = wayback[r["path"]]
            r["source_url"] = w["url"] if isinstance(w, dict) else w
            filled_url += 1
        if not r["source_page"].strip():
            u = per_img.get(os.path.basename(r["path"])) or lookup(r["source"], pages)
            if u:
                r["source_page"] = u
                filled_page += 1
        if not r["source_url"].strip() and not r["source_page"].strip() \
                and r["rights"] == "third_party":
            still.append(r["path"])

    print(f"commons urls written       : {filled_url}")
    print(f"commons attributions fixed : {fixed_attr}")
    print(f"source_page filled        : {filled_page}")
    print(f"own/collaborator notes    : {noted}")
    print(f"third-party with no provenance at all: {len(still)}")
    for p in still[:10]:
        print("   ", p)
    have = sum(1 for r in rows if r["source_url"].strip() or r["source_page"].strip())
    print(f"rows with some provenance : {have}/{len(rows)}")

    if args.write:
        with MANIFEST.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {MANIFEST}")
    else:
        print("(dry run -- pass --write)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
