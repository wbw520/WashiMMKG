"""Verify the Wikimedia Commons attributions on our images, and recover the right file.

The Commons file names in the manifest were written when the graph was built, not recorded
by a downloader, so some are wrong: of 42, only 8 matched a Commons file byte-for-byte and
18 named a page that does not exist. A resized copy cannot be checked by hash, so this
compares the picture itself -- a 64-bit difference hash against the Commons thumbnail --
and for a name that does not resolve, searches Commons and checks the candidates the same
way. Anything that still fails is reported as unverified rather than cited.

    python tools/verify_commons_images.py --classes /tmp/commons_classes.json --out output/commons_verified.json
"""
from __future__ import annotations

import argparse
import io
import json
import os
import pathlib
import time
import urllib.parse
import urllib.request

from PIL import Image

BASE = pathlib.Path(__file__).resolve().parents[1]
UA = {"User-Agent": "washi-provenance/1.0 (academic dataset provenance check)"}
API = "https://commons.wikimedia.org/w/api.php?"
# 8x8 dHash: two pictures of the same scene at different sizes stay within a few bits,
# different pictures are far past it.
MAX_DIST = 10


def dhash(im: Image.Image) -> int:
    g = im.convert("L").resize((9, 8), Image.LANCZOS)
    px = list(g.getdata())
    bits = 0
    for r in range(8):
        for c in range(8):
            bits = (bits << 1) | (px[r * 9 + c] < px[r * 9 + c + 1])
    return bits


def dist(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def api(params: dict) -> dict:
    params = {**params, "format": "json", "formatversion": "2"}
    req = urllib.request.Request(API + urllib.parse.urlencode(params), headers=UA)
    with urllib.request.urlopen(req, timeout=40) as r:
        return json.load(r)


def thumb_hash(title: str, width: int = 480) -> tuple[int, str] | None:
    """dHash of the Commons thumbnail for `title`, and the file page url."""
    d = api({"action": "query", "titles": title, "prop": "imageinfo",
             "iiprop": "url", "iiurlwidth": str(width)})
    pages = d.get("query", {}).get("pages", [])
    if not pages or pages[0].get("missing"):
        return None
    info = (pages[0].get("imageinfo") or [{}])[0]
    url = info.get("thumburl") or info.get("url")
    if not url:
        return None
    req = urllib.request.Request(url, headers=UA)
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            im = Image.open(io.BytesIO(r.read()))
            return dhash(im), info.get("descriptionurl", "")
    except Exception:
        return None


def search(text: str, limit: int = 8) -> list[str]:
    q = text.replace("File:", "").replace("_", " ").rsplit(".", 1)[0]
    try:
        d = api({"action": "query", "list": "search", "srsearch": q,
                 "srnamespace": "6", "srlimit": str(limit)})
        return [h["title"] for h in d.get("query", {}).get("search", [])]
    except Exception:
        return []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--classes", required=True)
    ap.add_argument("--titles", required=True, help="basename -> recorded Commons title")
    ap.add_argument("--paths", required=True, help="basename -> local path")
    ap.add_argument("--out", required=True)
    ap.add_argument("--delay", type=float, default=0.4)
    args = ap.parse_args()

    classes = json.loads(pathlib.Path(args.classes).read_text())
    titles = json.loads(pathlib.Path(args.titles).read_text())
    paths = json.loads(pathlib.Path(args.paths).read_text())

    out: dict[str, dict] = {}
    todo = [(b, "recorded") for b in classes.get("B", [])] + \
           [(b, "search") for b in classes.get("C", [])]
    for i, (base, mode) in enumerate(todo, 1):
        local = BASE / paths[base]
        if not local.exists():
            out[base] = {"status": "local file missing"}
            continue
        mine = dhash(Image.open(local))
        title = titles.get(base, "")
        got = None
        if mode == "recorded":
            r = thumb_hash(title)
            if r and dist(mine, r[0]) <= MAX_DIST:
                got = {"status": "confirmed", "title": title,
                       "url": r[1] or "https://commons.wikimedia.org/wiki/" + title.replace(" ", "_"),
                       "distance": dist(mine, r[0])}
            elif r:
                got = {"status": "different picture", "title": title,
                       "distance": dist(mine, r[0])}
        if got is None:
            best = None
            for cand in search(title or base):
                r = thumb_hash(cand)
                time.sleep(args.delay)
                if not r:
                    continue
                d = dist(mine, r[0])
                if best is None or d < best[0]:
                    best = (d, cand, r[1])
                if d <= MAX_DIST:
                    break
            if best and best[0] <= MAX_DIST:
                got = {"status": "recovered by search", "title": best[1],
                       "url": best[2] or "https://commons.wikimedia.org/wiki/" + best[1].replace(" ", "_"),
                       "distance": best[0], "recorded": title}
            else:
                got = {"status": "unverified", "recorded": title,
                       "closest": best[1] if best else None,
                       "distance": best[0] if best else None}
        out[base] = got
        print(f"  [{i}/{len(todo)}] {base[:28]:28s} {got['status']:22s} d={got.get('distance')}")
        time.sleep(args.delay)

    pathlib.Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1))
    n = sum(1 for v in out.values() if v.get("url"))
    print(f"\nverified with a url: {n}/{len(out)}   wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
