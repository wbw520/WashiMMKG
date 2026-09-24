"""Put the graph's 610 images where the code expects to find them.

The images arrive in two ways and the graph does not care which: every entity refers to a
picture as `output/extracted_images/<folder>/<file>`. The data archive ships the 255 we can
redistribute -- 213 ours and our collaborators', 42 from Wikimedia Commons -- under
`images/<rights group>/<folder>/<file>`, keeping the folder because 55 file names repeat
across the seminar folders and a flat copy would collide. The other 355 belong to the sites
they were collected from; those are downloaded from the address recorded in the manifest.

One command does both:

    python tools/place_images.py --data ../washi-data

Every file, copied or downloaded, is checked against the SHA-256 recorded when the image
was first collected, so a picture that has since been replaced on its site is reported
rather than used. Pass --no-download for text-only reproduction, which needs no images;
the multimodal and multimodal-chain templates, 466 of the 2,007 test items, do.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pathlib
import shutil
import time
import urllib.request

UA = ("WashiMMKG-fetch/1.0 (academic dataset reconstruction; "
      "see the paper's data availability statement)")
GROUP = {"own_or_collaborator": "own_seminar_materials",
         "wikimedia_commons": "wikimedia_commons"}
ROOT = "output/extracted_images"


def sha256(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def fetch(url: str, timeout: int = 30) -> bytes | None:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=timeout) as f:
            return f.read()
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="the data archive directory")
    ap.add_argument("--out", default=ROOT, help="root the graph's image paths point into")
    ap.add_argument("--no-download", action="store_true",
                    help="place only the shipped images; skip the third-party ones")
    ap.add_argument("--delay", type=float, default=1.0)
    args = ap.parse_args()

    data = pathlib.Path(args.data)
    images = data / "images"
    out = pathlib.Path(args.out)
    manifest = images / "image_manifest.csv"
    if not manifest.exists():
        print(f"no manifest at {manifest} -- is --data the data archive?")
        return 1
    rows = list(csv.DictReader(manifest.open()))
    sums_path = images / "checksums.json"
    sums = json.loads(sums_path.read_text()) if sums_path.exists() else {}

    copied = already = wrong = absent = 0
    for r in (x for x in rows if x["included"] == "yes"):
        rel = pathlib.Path(r["path"]).relative_to(ROOT)
        dest = out / rel
        if dest.exists():
            already += 1
            continue
        src = images / GROUP[r["rights"]] / rel
        if not src.exists():
            absent += 1
            print(f"  not in the archive: {r['path']}")
            continue
        blob = src.read_bytes()
        want = sums.get(r["path"])
        if want and sha256(blob) != want:
            wrong += 1
            print(f"  checksum mismatch in the archive, not copied: {r['path']}")
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        copied += 1
    print(f"shipped images: {copied} placed, {already} already there, "
          f"{wrong} failed their checksum, {absent} not in the archive")

    got = changed = unreachable = no_url = 0
    todo = [r for r in rows if r["included"] == "no"]
    if args.no_download:
        print(f"third-party images: {len(todo)} skipped (--no-download)")
    else:
        print(f"third-party images: downloading {sum(1 for r in todo if r['source_url'])} "
              f"of {len(todo)}, one at a time -- these are small sites")
        for r in todo:
            dest = out / pathlib.Path(r["path"]).relative_to(ROOT)
            if dest.exists():
                already += 1
                continue
            if not r["source_url"]:
                no_url += 1
                continue
            blob = fetch(r["source_url"])
            time.sleep(args.delay)
            if blob is None:
                unreachable += 1
                print(f"  unreachable: {r['source_url']}")
                continue
            want = sums.get(r["path"])
            if want and sha256(blob) != want:
                changed += 1
                print(f"  changed since collection, not written: {r['path']}")
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(blob)
            got += 1
        print(f"  {got} fetched, {changed} changed since collection, "
              f"{unreachable} unreachable, {no_url} have no recovered address")

    here = sum(1 for r in rows if (out / pathlib.Path(r["path"]).relative_to(ROOT)).exists())
    print(f"\n{here} of {len(rows)} images are in place under {out}")
    if here < len(rows):
        print("The rest are third-party pictures we cannot redistribute; the manifest "
              "records the page each was collected from so they can be traced by hand.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
