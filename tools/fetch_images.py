"""Fetch the third-party images of WashiMMKG from the pages they came from.

The data archive ships 255 of the graph's 610 images: 42 from Wikimedia Commons and 213
produced by us or our collaborators. The other 355 belong to the sites and publications
they were collected from, and we are not in a position to redistribute them. 297 of those
carry a recovered source URL in `image_manifest.csv`, and this script downloads them into
the layout the graph expects.

Running it is optional. Text-only reproduction needs no images at all; the multimodal and
multimodal-chain templates, 466 of the 2,007 test items, do.

    python tools/fetch_images.py --manifest ../washi-data/images/image_manifest.csv \
        --out output/extracted_images

Each download is checked against the SHA-256 recorded when the image was first collected,
so a file that has since been replaced on its site is reported rather than silently used.
Requests are made one at a time with a pause between them: these are small sites.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import pathlib
import time
import urllib.error
import urllib.request

UA = ("WashiMMKG-fetch/1.0 (academic dataset reconstruction; "
      "see the paper's data availability statement)")


def fetch(url: str, timeout: int = 30) -> bytes | None:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=timeout) as f:
            return f.read()
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", default="output/extracted_images",
                    help="root the graph's image paths are relative to")
    ap.add_argument("--delay", type=float, default=1.0)
    ap.add_argument("--checksums", default=None,
                    help="optional JSON of path -> sha256 to verify against")
    args = ap.parse_args()

    sums = {}
    if args.checksums and os.path.exists(args.checksums):
        import json
        sums = json.loads(open(args.checksums).read())

    rows = list(csv.DictReader(open(args.manifest)))
    todo = [r for r in rows if r.get("included") == "no" and r.get("source_url")]
    seen: set[str] = set()
    got = missing = changed = 0

    for r in todo:
        path = r["path"]
        if path in seen:
            continue
        seen.add(path)
        dest = pathlib.Path(args.out).parent / path if path.startswith("output/") \
            else pathlib.Path(args.out) / os.path.basename(path)
        if dest.exists():
            continue
        blob = fetch(r["source_url"])
        time.sleep(args.delay)
        if blob is None:
            missing += 1
            print(f"  unavailable: {r['source_url']}")
            continue
        want = sums.get(path)
        if want and hashlib.sha256(blob).hexdigest() != want:
            changed += 1
            print(f"  changed since collection, not written: {path}")
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(blob)
        got += 1

    total = len(seen)
    print(f"\n{got} of {total} fetched; {missing} no longer reachable; {changed} changed")
    print("Images without a recovered URL stay missing; the manifest lists them with their "
          "source document so they can be traced by hand.")


if __name__ == "__main__":
    main()
