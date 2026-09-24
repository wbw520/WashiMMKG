"""Read raw source files into paragraphs (text) and collect image files.

Supported text inputs: .txt, .md, .pdf (via pypdf). Images are passed straight to the
Creator's multimodal-alignment step. A leading-underscore filename (e.g. _notes.txt)
is treated as a control/sidecar file and skipped by ingestion.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

# sentence boundaries for both Latin-script and Japanese text
_SENT_SPLIT = re.compile(r"(?<=[.!?。！？])\s+|(?<=[。！？])")

TEXT_EXTS = {".txt", ".md", ".markdown", ".rst"}
PDF_EXTS = {".pdf"}


@dataclass
class TextDoc:
    rel_path: str
    abs_path: Path
    paragraphs: list[str]


@dataclass
class ExtractedImage:
    path: Path        # where the image was saved
    source: str       # the document it was extracted from (rel path under raw/)
    page: int         # 1-based page number it appeared on
    page_text: str    # text of that page (used to scope entity candidates)


def _read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _read_pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    return "\n\n".join((page.extract_text() or "") for page in reader.pages)


def split_paragraphs(text: str, min_chars: int, max_chars: int) -> list[str]:
    """Split into paragraphs, merging tiny fragments and chunking oversized ones."""
    blocks = [b.strip() for b in text.replace("\r\n", "\n").split("\n\n")]
    blocks = [b for b in blocks if b]

    # merge short fragments forward
    merged: list[str] = []
    buf = ""
    for b in blocks:
        buf = (buf + "\n" + b).strip() if buf else b
        if len(buf) >= min_chars:
            merged.append(buf)
            buf = ""
    if buf:
        if merged:
            merged[-1] = (merged[-1] + "\n" + buf).strip()
        else:
            merged.append(buf)

    # chunk oversized paragraphs on sentence boundaries
    out: list[str] = []
    for p in merged:
        if len(p) <= max_chars:
            out.append(p)
            continue
        cur = ""
        for sent in _SENT_SPLIT.split(p):
            sent = (sent or "").strip()
            if not sent:
                continue
            # a single sentence longer than max_chars is hard-wrapped
            while len(sent) > max_chars:
                if cur:
                    out.append(cur.strip())
                    cur = ""
                out.append(sent[:max_chars])
                sent = sent[max_chars:]
            if len(cur) + len(sent) + 1 > max_chars and cur:
                out.append(cur.strip())
                cur = sent
            else:
                cur = (cur + " " + sent).strip()
        if cur:
            out.append(cur.strip())
    return out


def extract_pdf_images(pdf_path: Path, source_rel: str, out_dir: Path) -> list[ExtractedImage]:
    """Extract embedded raster images from a PDF, save them, and tag each with the
    page it came from plus that page's text (the locality signal for entity linking).
    """
    from pypdf import PdfReader

    reader = PdfReader(str(pdf_path))
    dest_dir = out_dir / pdf_path.stem
    out: list[ExtractedImage] = []
    for pi, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        try:
            images = list(page.images)
        except Exception:
            images = []  # some encodings pypdf can't decode; skip rather than crash
        for ii, img in enumerate(images, start=1):
            ext = (Path(img.name).suffix or ".png").lower()
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / f"p{pi:03d}_{ii:02d}{ext}"
            try:
                dest.write_bytes(img.data)
            except Exception:
                continue
            out.append(ExtractedImage(path=dest, source=source_rel, page=pi, page_text=page_text))
    return out


def is_text_file(path: Path) -> bool:
    return path.suffix.lower() in TEXT_EXTS | PDF_EXTS


def is_control_file(path: Path) -> bool:
    return path.name.startswith("_")


def load_text_doc(path: Path, raw_dir: Path, min_chars: int, max_chars: int) -> TextDoc:
    ext = path.suffix.lower()
    if ext in PDF_EXTS:
        text = _read_pdf(path)
    else:
        text = _read_text_file(path)
    return TextDoc(
        rel_path=str(path.relative_to(raw_dir)),
        abs_path=path,
        paragraphs=split_paragraphs(text, min_chars, max_chars),
    )


def iter_source_files(raw_dir: Path, image_exts: set[str]):
    """Yield ``(path, kind)`` for every relevant file under ``raw_dir``.

    kind is one of ``"text"`` or ``"image"``. Control files (``_*``) are skipped.
    """
    for path in sorted(raw_dir.rglob("*")):
        if not path.is_file() or is_control_file(path):
            continue
        ext = path.suffix.lower()
        if is_text_file(path):
            yield path, "text"
        elif ext in image_exts:
            yield path, "image"
