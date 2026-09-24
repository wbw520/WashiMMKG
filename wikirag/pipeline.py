"""WikiWalk-RAG end to end: decompose -> roots -> walk -> verify -> join -> answer.

`join` is where the per-sub-query walks meet. Chains are ordered by verdict first and
score second, then their edges are taken in turn until the evidence budget is full. The
budget (K triples) is the same one the baselines get, so the comparison isolates *which*
triples a method finds rather than how many it is allowed to show.

Taking edges round-robin across sub-queries rather than draining the best chain first is
deliberate: on a two-constraint question the best single chain satisfies one constraint
completely and the other not at all, and an answer needs a little of both more than it
needs all of one.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from dataclasses import dataclass, field

from . import baselines as B
from . import prompts
from .page import WikiGraph
from .walker import Chain, WikiWalker, _json

VERDICT_RANK = {"support": 0, "partial": 1, None: 2, "reject": 3}


@dataclass
class Retrieval:
    question: str
    subqueries: list[str] = field(default_factory=list)
    chains: list[Chain] = field(default_factory=list)
    triples: list[tuple[str, str, str]] = field(default_factory=list)
    images: list[dict] = field(default_factory=list)
    answer: str = ""
    n_llm_calls: int = 0
    seconds: float = 0.0


class WikiWalkRAG:
    def __init__(self, graph: WikiGraph, walker: WikiWalker, llm,
                 evidence_budget: int = 5, verify: bool = True,
                 with_images: bool = True, max_verify: int = 8,
                 base_dir=None, max_answer_images: int = 3,
                 hide_captions: frozenset = frozenset(),
                 drain_best_chain: bool = False, use_mentions: bool = True,
                 model_select: bool = False):
        self.g = graph
        self.walker = walker
        self.llm = llm
        self.budget = evidence_budget
        self.do_verify = verify
        self.with_images = with_images
        self.max_verify = max_verify
        self.base_dir = base_dir
        self.max_answer_images = max_answer_images
        self.hide_captions = hide_captions
        # ablation: fill the budget from the top-ranked chain before moving on
        self.drain_best_chain = drain_best_chain
        # ablation: fall back to ranking entity names against the question, the
        # arrangement mention-anchoring replaced
        self.use_mentions = use_mentions
        # diagnostic: let the backbone choose the final K triples instead of taking them
        # round-robin from the ranked chains. The walk is unchanged -- same pages, roots,
        # sub-queries, beam, cite/follow split and verification -- only the assembly of
        # the evidence set moves from a rule to the model. It exists because the one
        # backbone that selects facts well on its own (Qwen3.8-27B keeps 3.2 facts at
        # 0.70 precision under ReAct) is also the one WikiWalk helps least, which is what
        # this measures rather than assumes.
        self.model_select = model_select


    def _model_pick(self, question: str, evidence) -> list[tuple[str, str, str]] | None:
        """Let the backbone choose the evidence set. None means "fall back to the rule".

        Every candidate the walk produced is offered once, in ranked order, numbered.
        The cap keeps the prompt bounded; beyond it the tail is the lowest-ranked chains'
        evidence, which round-robin would rarely reach either.
        """
        cand: list[tuple[str, str, str]] = []
        seen: set = set()
        for ev in evidence:
            for e in ev:
                e = tuple(e)
                if e not in seen:
                    seen.add(e)
                    cand.append(e)
        if not cand:
            return []
        if len(cand) <= self.budget:
            return cand[: self.budget]
        listing = "\n".join(f"  {i}. {B.triple_text(*e)}"
                             for i, e in enumerate(cand[:40], 1))
        try:
            reply = self.llm.complete(
                prompts.SELECT_SYSTEM,
                prompts.SELECT_USER.format(question=question, budget=self.budget,
                                           candidates=listing),
                max_tokens=120)
            obj = _json(reply)
            idx = [int(i) for i in (obj.get("keep") or []) if str(i).strip().lstrip("-").isdigit()]
        except Exception:
            return None
        out: list[tuple[str, str, str]] = []
        for i in idx:
            if 1 <= i <= len(cand[:40]) and cand[i - 1] not in out:
                out.append(cand[i - 1])
            if len(out) >= self.budget:
                break
        return out or None

    def join(self, chains: list[Chain], question: str = "") -> list[tuple[str, str, str]]:
        """Merge chains into one evidence set, capped at the budget.

        Evidence is taken one item at a time from each chain in turn, cycling. Draining
        the best chain first looks reasonable and is wrong for every question whose answer
        needs more than one starting point: a comparison wants one fact from each of two
        entities, and a set difference wants the population from one page and the
        exclusion from another, but the strongest chain alone can fill all five slots and
        the second entity never appears. Cycling keeps the budget spread across the parts
        of the answer rather than concentrating it on whichever part scored highest.

        Chains are ordered by verdict first and score second, so the cycle still visits
        verified evidence before unverified, and rejected chains not at all.
        """
        ranked = sorted(
            [c for c in chains if c.verdict != "reject"],
            key=lambda c: (VERDICT_RANK.get(c.verdict, 2), -c.score),
        )
        if not ranked:
            return []

        evidence = [c.evidence() for c in ranked]
        if self.model_select:
            picked = self._model_pick(question, evidence)
            if picked is not None:
                return picked
        if self.drain_best_chain:
            picked, seen = [], set()
            for ev in evidence:
                for e in ev:
                    e = tuple(e)
                    if e not in seen:
                        seen.add(e)
                        picked.append(e)
                    if len(picked) >= self.budget:
                        return picked
            return picked

        cursors = [0] * len(ranked)
        picked: list[tuple[str, str, str]] = []
        seen: set[tuple[str, str, str]] = set()
        progress = True
        while len(picked) < self.budget and progress:
            progress = False
            for i, ev in enumerate(evidence):
                if len(picked) >= self.budget:
                    break
                while cursors[i] < len(ev):
                    edge = tuple(ev[cursors[i]])
                    cursors[i] += 1
                    if edge not in seen:
                        seen.add(edge)
                        picked.append(edge)
                        progress = True
                        break
        return picked

    def evidence_text(self, triples, images) -> str:
        lines = [f"- {h} --{r}--> {u}" for h, r, u in triples]
        for im in images[: self.max_answer_images]:
            # the picture is still shown to the answerer; only its wording is withheld,
            # and only when that wording is the answer being asked for
            cap = "" if im.get("path") in self.hide_captions else (im.get("caption") or "")
            lines.append(f"- [image of {im.get('entity')}] {cap}")
        return "\n".join(lines) if lines else "(no evidence retrieved)"

    def run(self, question: str, image_scores=None, query_images=None) -> Retrieval:
        t0 = time.time()
        res = Retrieval(question=question)

        counter = getattr(self.llm, "reset", None)
        if counter:
            self.llm.reset()

        res.subqueries, mentions = self.walker.decompose(question)
        if not self.use_mentions:
            mentions = []

        all_chains: list[Chain] = []
        for sub in res.subqueries:
            all_chains.extend(self.walker.walk(question, sub, mentions, image_scores))

        if self.do_verify:
            # Verification is per-chain, so an unbounded walk would pay for it linearly.
            # Only the strongest candidates are judged; the rest keep verdict None, which
            # `join` already ranks below anything verified.
            all_chains.sort(key=lambda c: -c.score)
            with ThreadPoolExecutor(max_workers=self.walker.workers) as pool:
                list(pool.map(lambda c: self.walker.verify(question, c),
                              all_chains[: self.max_verify]))

        res.chains = all_chains

        res.triples = self.join(all_chains, question)
        if self.with_images:
            seen = set()
            for h, _, u in res.triples:
                for n in (h, u):
                    for im in self.g.images_of(n):
                        key = im.get("path")
                        if key and key not in seen:
                            seen.add(key)
                            res.images.append({"entity": n, **im})

        user = prompts.ANSWER_USER.format(
            question=question,
            evidence=self.evidence_text(res.triples, res.images))
        # Show the answer model the pictures, not just their captions. A_img is the point
        # of a multimodal graph, and a forced-choice question ("which of these two is in
        # the photograph") is unanswerable from a caption -- describing the image in words
        # would either give the answer away or say nothing. Passing captions alone made
        # this method score zero on every such item while the perfect-evidence upper bound
        # scored 1.0 on them, which is a gap in the plumbing, not in the retrieval.
        # q_m first. The query image is part of the question -- "which of these two is in
        # the photograph" cannot be read without it -- so it is supplied, not retrieved.
        paths = list(query_images or [])
        if self.with_images and res.images:
            for im in res.images[: self.max_answer_images]:
                fp = (self.base_dir / im["path"]) if self.base_dir else Path(im["path"])
                if fp.exists() and str(fp) not in paths:
                    paths.append(str(fp))
        paths = paths[: self.max_answer_images]
        try:
            if paths:
                res.answer = self.llm.complete_vision(
                    prompts.ANSWER_SYSTEM, user, paths, max_tokens=200).strip()
            else:
                res.answer = self.llm.complete(
                    prompts.ANSWER_SYSTEM, user, max_tokens=200).strip()
        except Exception as exc:
            res.answer = f"(answer generation failed: {exc})"

        res.n_llm_calls = getattr(self.llm, "calls", 0)
        res.seconds = time.time() - t0
        return res
