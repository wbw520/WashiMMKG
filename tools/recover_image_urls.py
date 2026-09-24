"""Recover the source URL of each collected image by re-crawling the sites it came from.

The graph stores 610 images. 245 of them are ours or Wikimedia Commons and can be
redistributed; the other 365 came from third-party pages and cannot, so the archive ships
their records but not the files. The usual remedy -- publish a URL list and let readers
fetch the images themselves -- was unavailable because the collection step never recorded
the addresses: the stored `source` names a scraped document, not a URL.

This recovers them. For each site, candidate pages are crawled, every image they reference
is downloaded, and its SHA-256 is compared against the images we hold. The files were
saved verbatim at collection time, so the match is byte-exact rather than perceptual: on
the first site tried, 145 of 146 candidates matched exactly and none matched by accident.

The crawl is deliberately slow and identifies itself. robots.txt is read and obeyed, and
a site that asks for a crawl delay gets it.

    python tools/recover_image_urls.py --index /tmp/thirdparty_index.json \
        --out output/image_url_recovery.json
"""
from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import urllib.robotparser

UA = ("WashiResearchBot/1.0 (academic research; "
      "recovering source URLs for previously collected images)")
IMG_RE = re.compile(r'(https?://[^\s"\'<>()]+\.(?:jpg|jpeg|png|webp))', re.I)
# lazy-loading and CSS backgrounds put the address somewhere other than <img src>
ATTR_RE = re.compile(r'(?:src|data-src|data-lazy-src|data-original|href)=["\']([^"\']+\.'
                     r'(?:jpg|jpeg|png|webp))', re.I)
CSS_RE = re.compile(r'url\(["\']?([^)"\']+\.(?:jpg|jpeg|png|webp))', re.I)
PAGE_RE = re.compile(r'href=["\'](/[^"\'#?]*|https?://[^"\'#?]+)["\']', re.I)


class Site:
    """One host, with its robots rules and its own pace."""

    def __init__(self, host: str, delay: float = 1.0):
        self.host = host
        self.delay = delay
        self.last = 0.0
        self.rp = urllib.robotparser.RobotFileParser()
        try:
            self.rp.set_url(f"https://{host}/robots.txt")
            self.rp.read()
            d = self.rp.crawl_delay(UA) or self.rp.crawl_delay("*")
            if d:
                self.delay = max(self.delay, float(d))
        except Exception:
            self.rp = None

    def allowed(self, url: str) -> bool:
        if self.rp is None:
            return True
        try:
            return self.rp.can_fetch(UA, url)
        except Exception:
            return True

    def wait(self) -> None:
        gap = time.time() - self.last
        if gap < self.delay:
            time.sleep(self.delay - gap)
        self.last = time.time()


def fetch(site: Site, url: str, timeout: int = 25) -> bytes | None:
    if not site.allowed(url):
        return None
    site.wait()
    try:
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=timeout) as f:
            body = f.read()
            if f.headers.get("Content-Encoding") == "gzip":
                body = gzip.decompress(body)
            return body
    except Exception:
        return None


def image_urls(page_url: str, html: str) -> set[str]:
    out = set(IMG_RE.findall(html))
    for rel in ATTR_RE.findall(html) + CSS_RE.findall(html):
        out.add(urllib.parse.urljoin(page_url, rel))
    return {u for u in out if u.startswith("http")}


def crawl(seed: str, hashes: dict[str, str], max_pages: int, delay: float,
          verbose: bool = True) -> dict[str, dict]:
    host = urllib.parse.urlparse(seed).netloc
    site = Site(host, delay)
    seen_pages: set[str] = set()
    queue = [seed]
    tried: set[str] = set()
    found: dict[str, dict] = {}

    while queue and len(seen_pages) < max_pages:
        page = queue.pop(0)
        if page in seen_pages:
            continue
        seen_pages.add(page)
        body = fetch(site, page)
        if body is None:
            continue
        html = body.decode("utf-8", "ignore")
        for iu in image_urls(page, html):
            if iu in tried:
                continue
            tried.add(iu)
            blob = fetch(site, iu, timeout=30)
            if blob is None:
                continue
            h = hashlib.sha256(blob).hexdigest()
            if h in hashes:
                found[hashes[h]] = {"url": iu, "page": page, "match": "sha256"}
                if verbose:
                    print(f"    match {os.path.basename(hashes[h])[:38]:38s} <- {iu[:70]}",
                          flush=True)
        # follow same-host links one level at a time
        for href in PAGE_RE.findall(html):
            nxt = urllib.parse.urljoin(page, href)
            if urllib.parse.urlparse(nxt).netloc == host and nxt not in seen_pages:
                if len(queue) < max_pages * 3:
                    queue.append(nxt)
    return found


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True, help="hash index written by the caller")
    ap.add_argument("--seeds", required=True, help="JSON: {source document: [seed url, ...]}")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-pages", type=int, default=40, help="pages per seed")
    ap.add_argument("--delay", type=float, default=1.0, help="minimum seconds between requests")
    args = ap.parse_args()

    idx = json.loads(open(args.index).read())
    hashes = idx["hash_to_path"]
    meta = idx["path_meta"]
    seeds = json.loads(open(args.seeds).read())

    found: dict[str, dict] = {}
    if os.path.exists(args.out):
        found = json.loads(open(args.out).read())
        print(f"resuming with {len(found)} already recovered")

    for source, urls in seeds.items():
        outstanding = [p for p, m in meta.items()
                       if m["source"] == source and p not in found]
        if not outstanding:
            continue
        print(f"\n=== {source}: {len(outstanding)} images outstanding ===", flush=True)
        for seed in urls:
            got = crawl(seed, hashes, args.max_pages, args.delay)
            new = {k: v for k, v in got.items() if k not in found}
            found.update(new)
            print(f"  {seed} -> {len(new)} new", flush=True)
            json.dump(found, open(args.out, "w"), ensure_ascii=False, indent=1)
            if not [p for p, m in meta.items()
                    if m["source"] == source and p not in found]:
                break

    by_source = collections.Counter(meta[p]["source"] for p in found if p in meta)
    print(f"\nrecovered {len(found)} of {len(meta)} third-party images")
    for s, n in by_source.most_common(10):
        print(f"  {n:4d}  {s[:56]}")


if __name__ == "__main__":
    main()
