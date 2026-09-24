"""Filter — semantic deduplication (paper §"Semantic Filtering").

A triple is kept only if no distinct kept triple exceeds the cosine-similarity
threshold delta:

    keep tau  iff  not exists tau' != tau with sim(tau, tau') > delta

Operating at the semantic (embedding) level rather than by string matching is
essential for a domain like Washi where terminology varies across regions and
historical documents.

For incremental builds, new candidates are compared against BOTH the triples
already in the graph and the candidates kept so far this run.
"""

from __future__ import annotations

import numpy as np

from .embeddings import Embedder, cosine_matrix, triple_text
from .extractor import Candidate
from .graph import KnowledgeGraph


class Filter:
    def __init__(self, embedder: Embedder, delta: float):
        self.embedder = embedder
        self.delta = delta

    def filter(self, candidates: list[Candidate], graph: KnowledgeGraph) -> list[Candidate]:
        if not candidates:
            return []

        # Baseline: triples already in the graph (their embeddings define what
        # "already known" looks like for semantic dedup).
        existing_texts = [triple_text(t["h"], t["r"], t["u"]) for t in graph.triples]
        existing_emb = self.embedder.encode(existing_texts)

        cand_texts = [triple_text(c.h, c.r, c.u) for c in candidates]
        cand_emb = self.embedder.encode(cand_texts)

        kept: list[Candidate] = []
        kept_emb_rows: list[np.ndarray] = []

        for i, cand in enumerate(candidates):
            v = cand_emb[i : i + 1]
            # against existing graph
            if existing_emb.shape[0] and cosine_matrix(v, existing_emb).max() > self.delta:
                continue
            # against already-kept candidates this run
            if kept_emb_rows:
                kept_emb = np.vstack(kept_emb_rows)
                if cosine_matrix(v, kept_emb).max() > self.delta:
                    continue
            kept.append(cand)
            kept_emb_rows.append(cand_emb[i])

        return kept
