"""Detector — entity canonicalization (paper §"Entity Canonicalization").

Implements canon: E_filtered -> E. For each filtered entity mention, the LLM compares
it against the existing canonical entities (E starts from whatever is already in the
graph) using contextual descriptions. If judged equivalent to some e in E, the mention
maps to e; otherwise a new canonical entity is created and added to E. Canon is then
applied to every triple:  canon(h, r, u) = (canon(h), r, canon(u)).

To keep the number of LLM judgements bounded, candidate canonical entities are
pre-filtered by embedding similarity before the LLM decides equivalence.
"""

from __future__ import annotations

import numpy as np

from . import prompts
from .embeddings import Embedder, cosine_matrix
from .extractor import Candidate
from .graph import KnowledgeGraph
from .llm import LLMClient, extract_json


class Detector:
    def __init__(
        self,
        llm: LLMClient,
        embedder: Embedder,
        candidate_sim_threshold: float,
        max_candidates: int,
    ):
        self.llm = llm
        self.embedder = embedder
        self.sim_threshold = candidate_sim_threshold
        self.max_candidates = max_candidates

    def canonicalize(
        self, candidates: list[Candidate], graph: KnowledgeGraph
    ) -> tuple[list[Candidate], dict[str, str]]:
        """Return (canonicalized triples, mention -> canonical map)."""
        # Canonical set E begins with the entities already in the graph.
        # Held in a mutable cell so the inner closure can grow E as it creates entities.
        dim = self.embedder.model.get_sentence_embedding_dimension()
        canon_names: list[str] = list(graph.entities.keys())
        cell = {
            "emb": self.embedder.encode(canon_names)
            if canon_names
            else np.zeros((0, dim), np.float32)
        }

        mention_to_canon: dict[str, str] = {}
        # one context sentence per mention (for the LLM judgement)
        mention_context: dict[str, str] = {}
        for c in candidates:
            mention_context.setdefault(c.h, c.context)
            mention_context.setdefault(c.u, c.context)

        def resolve(mention: str) -> str:
            if mention in mention_to_canon:
                return mention_to_canon[mention]
            # exact existing match
            if mention in graph.entities or mention in {v for v in mention_to_canon.values()}:
                # still allow exact-name reuse
                if mention in graph.entities:
                    mention_to_canon[mention] = mention
                    return mention

            # embedding pre-filter against current canonical set
            m_emb = self.embedder.encode([mention])
            cands: list[str] = []
            if cell["emb"].shape[0]:
                sims = cosine_matrix(m_emb, cell["emb"])[0]
                order = np.argsort(-sims)
                for idx in order[: self.max_candidates]:
                    if sims[idx] >= self.sim_threshold:
                        cands.append(canon_names[idx])

            chosen = None
            if cands:
                # exact string hit among candidates short-circuits the LLM
                if mention in cands:
                    chosen = mention
                else:
                    chosen = self._llm_judge(mention, mention_context.get(mention, ""), cands)

            if chosen is None:
                # new canonical entity -> add to E
                chosen = mention
                canon_names.append(mention)
                cell["emb"] = (
                    np.vstack([cell["emb"], m_emb]) if cell["emb"].shape[0] else m_emb
                )

            mention_to_canon[mention] = chosen
            return chosen

        out: list[Candidate] = []
        for c in candidates:
            ch = resolve(c.h)
            cu = resolve(c.u)
            if ch.lower() == cu.lower():
                continue  # self-loop after canonicalization
            out.append(Candidate(h=ch, r=c.r, u=cu, source=c.source, context=c.context))

        return out, mention_to_canon

    def _llm_judge(self, mention: str, context: str, candidates: list[str]) -> str | None:
        listing = "\n".join(f"- {c}" for c in candidates)
        reply = self.llm.complete(
            prompts.DETECTOR_SYSTEM,
            prompts.DETECTOR_USER.format(mention=mention, context=context, candidates=listing),
        )
        obj = extract_json(reply, default={})
        match = obj.get("match") if isinstance(obj, dict) else None
        if isinstance(match, str) and match in candidates:
            return match
        return None
