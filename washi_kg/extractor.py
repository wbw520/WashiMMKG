"""Extractor — high-recall triple extraction (paper §"Triple Extraction").

For a document D = {s_1, ..., s_n}, Ext(s_i) is the LLM-extracted triple set of
paragraph s_i, and the raw candidate set is T_raw = union_i Ext(s_i). This stage
deliberately over-generates; downstream stages contract and canonicalize.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import prompts
from .llm import LLMClient, extract_json


@dataclass
class Candidate:
    h: str
    r: str
    u: str
    source: str       # file the triple came from
    context: str      # paragraph it was extracted from (used as A_text + Detector context)


def _norm_relation(r: str) -> str:
    return "_".join(r.strip().lower().split()).replace("-", "_")


class Extractor:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def extract_paragraph(self, paragraph: str, source: str) -> list[Candidate]:
        reply = self.llm.complete(
            prompts.EXTRACTOR_SYSTEM,
            prompts.EXTRACTOR_USER.format(paragraph=paragraph),
        )
        items = extract_json(reply, default=[])
        out: list[Candidate] = []
        if not isinstance(items, list):
            return out
        for it in items:
            if not isinstance(it, dict):
                continue
            h, r, u = it.get("h"), it.get("r"), it.get("u")
            if not (isinstance(h, str) and isinstance(r, str) and isinstance(u, str)):
                continue
            h, u = h.strip(), u.strip()
            r = _norm_relation(r)
            if h and r and u and h.lower() != u.lower():
                out.append(Candidate(h=h, r=r, u=u, source=source, context=paragraph))
        return out
