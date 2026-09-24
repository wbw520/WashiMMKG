"""Best-first walk over the wiki — retrieval whose unit is a path, not a subgraph.

The original AD-RAG maintains a frontier and expands it breadth-first: every accepted
entity contributes all of its neighbours to the next candidate pool, an LLM prunes the
pool, and a second LLM validates survivors one at a time. Three costs follow. The pool
grows with the accepted set, so prompt size grows with search progress. Validation is
per-node, so a node is judged without knowing which route reached it. And the retrieved
object is a subgraph, while the benchmark's ground truth is a chain -- the retriever is
not optimising the thing being scored.

Here a searcher stands on a page and picks its next step, so the observation is one page
regardless of how far the search has gone. Partial paths are ranked and only the best are
extended, which is best-first search rather than breadth-first: a promising route is
followed deeply instead of every route being widened by one. Judgement happens once, on a
finished path, where "does this get a reader to the answer" is answerable -- it is not
answerable about a lone node.

One walker runs per sub-query, and they meet only at the end (`join`). Multi-constraint
questions are the reason: a shared frontier has to satisfy every constraint at once and
tends to drift to whatever satisfies the easiest, while separate walks each find their own
evidence and the answer is where they intersect.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from . import prompts
from .page import WikiGraph
from .index import TextIndex


@dataclass
class Chain:
    """A partial or finished reasoning path.

    ``edges`` holds triples in their true orientation as stored in T, while ``nodes``
    holds the route actually walked. The two differ whenever the walk follows a link
    backwards: a page lists its incoming links because they are just as navigable as
    outgoing ones ("who produces this paper?"), but the underlying triple still reads
    (producer, produces, paper). Recording the display direction instead would emit
    reversed triples that can never match T_gt, silently zeroing retrieval recall.
    """

    root: str
    subquery: str
    edges: list[tuple[str, str, str]] = field(default_factory=list)
    cited: list[tuple[str, str, str]] = field(default_factory=list)
    nodes: list[str] = field(default_factory=list)
    score: float = 0.0
    answered: bool = False
    verdict: str | None = None
    why: str = ""

    def __post_init__(self):
        if not self.nodes:
            self.nodes = [self.root]

    @property
    def head(self) -> str:
        return self.nodes[-1]

    def render(self) -> str:
        """Everything this chain offers, which is what a verdict must be passed on.

        Rendering only the walked path hid the cited facts from the verifier, so a chain
        whose entire value was what it noted along the way was judged on an unrelated
        step and rejected -- and `join` skips rejected chains, so correct evidence was
        thrown away by the component meant to protect it.
        """
        ev = self.evidence()
        if not ev:
            return self.root
        return "\n".join(f"  {h} --{r}--> {u}" for h, r, u in ev)

    def trail(self) -> str:
        return " -> ".join(self.nodes)

    def extend(self, r: str, target: str, incoming: bool, score: float) -> "Chain":
        triple = (target, r, self.head) if incoming else (self.head, r, target)
        return Chain(self.root, self.subquery, self.edges + [triple],
                     list(self.cited), self.nodes + [target], score)

    def evidence(self) -> list[tuple[str, str, str]]:
        """Cited facts first: they were judged to answer, the path only led there."""
        out, seen = [], set()
        for t in self.cited + self.edges:
            if t not in seen:
                seen.add(t)
                out.append(t)
        return out


class WikiWalker:
    def __init__(
        self,
        graph: WikiGraph,
        llm,
        text_index: TextIndex,
        *,
        beam: int = 3,
        max_depth: int = 3,
        roots_per_subquery: int = 2,
        max_choices: int = 3,
        max_cite: int = 8,
        lam: float = 0.6,
        max_pages: int = 24,
        max_subqueries: int = 2,
        mention_sim_floor: float = 0.55,
        image_sim_floor: float = 0.0,
        merge_cite_follow: bool = False,
        verify_against_question: bool = False,
        hide_captions: frozenset = frozenset(),
        nav_images: int = 0,
        nav_keep_captions: bool = False,
        base_dir=None,
        workers: int = 4,
    ):
        self.g = graph
        # How the walk perceives a picture. Zero -- the default -- means it reads the
        # caption, which is what a text-only navigator can do. A positive value hands the
        # navigator the photographs themselves and blanks every caption on the page, so
        # the two conditions are a clean substitution rather than an addition: the walk
        # either reads about the image or looks at it, never both.
        self.nav_images = nav_images
        self.nav_keep_captions = nav_keep_captions
        self.base_dir = base_dir
        # image paths whose caption is the item's gold answer: the walker may see
        # the picture but must not be handed its wording
        self.hide_captions = hide_captions
        self.llm = llm
        self.index = text_index
        self.beam = beam
        self.max_depth = max_depth
        self.roots_per_subquery = roots_per_subquery
        self.max_choices = max_choices
        # Citing and following want different budgets. "What is known about X" is answered
        # by everything on X's page, so a shared cap of 3 truncates the answer itself;
        # following, by contrast, should stay narrow or the beam degenerates into BFS.
        self.max_cite = max_cite
        self.lam = lam
        self.max_pages = max_pages
        self.max_subqueries = max_subqueries
        self.mention_sim_floor = mention_sim_floor
        self.image_sim_floor = image_sim_floor
        # Ablations. Each restores a design the method deliberately moved away from, so
        # the cost of that move can be measured rather than asserted.
        self.merge_cite_follow = merge_cite_follow
        self.verify_against_question = verify_against_question
        self.workers = workers

    # ------------------------------------------------------------------ decomposition
    def decompose(self, question: str) -> tuple[list[str], list[str]]:
        """Return (sub-queries, surface mentions to resolve into search roots)."""
        subs: list[str] = []
        mentions: list[str] = []
        try:
            reply = self.llm.complete(
                prompts.DECOMPOSE_SYSTEM,
                prompts.DECOMPOSE_USER.format(question=question),
                max_tokens=200,
            )
            obj = _json(reply)
            subs = [s for s in (obj.get("subqueries") or []) if isinstance(s, str) and s.strip()]
            mentions = [m for m in (obj.get("mentions") or [])
                        if isinstance(m, str) and m.strip()][:4]
        except Exception:
            pass
        # The full question always walks as well. A decomposer handed a chain question
        # ("A, then its region, then that region's country") tends to split it into one
        # sub-query per hop even when told not to, and each of those is satisfied by a
        # single step -- so the walk stops one hop in and the deep route is never taken.
        # Keeping the whole question as its own walker guarantees somebody goes the
        # distance; the decomposed walkers still cover the multi-constraint case.
        out = [question] + [s for s in subs if s.strip().lower() != question.strip().lower()]
        # Two walkers, not four. The whole question always walks; one decomposed sub-query
        # covers the multi-constraint case. Further sub-queries on a chain question are
        # near-duplicates of the walk already under way and cost a full walk each.
        return out[: self.max_subqueries], mentions

    # ------------------------------------------------------------------------- roots
    def roots(self, subquery: str, mentions: list[str] | None = None,
              image_scores: dict[str, float] | None = None,
              alpha: float = 0.7) -> list[str]:
        """Search roots for a sub-query.

        Resolved mentions come first. Embedding the raw question against entity names
        does not work here: a well-formed item never names its own answer, so the only
        entities in the text are the *given* ones, and they are diluted by every
        function word around them -- "which paper is produced in Nara and used for
        mounting" scored an exhibition title above `Nara`. Resolving the named things
        directly puts the walk where the question actually starts; the embedding is kept
        only as a fallback for questions that name nothing resolvable.
        """
        roots: list[str] = []

        # Mentions first. An exactly resolved name is a certainty; image similarity is a
        # guess, and most items carry an incidental picture of something on their path
        # rather than a query image. Ranking the guess ahead of the certainty let a
        # by-the-way photograph displace the entity a question actually named, which cost
        # comparative questions .32 accuracy before the ablation exposed it.
        for m in (mentions or []):
            hit = self.g.resolve(m)
            if hit is None:
                top = self.index.top(m, 1)
                if top and top[0][1] >= self.mention_sim_floor:
                    hit = top[0][0]
            if hit and hit not in roots:
                roots.append(hit)

        # Image roots then fill what remains. For an open visual question there are no
        # mentions at all, so this is the only way in; for a question that names its
        # subject it is a supplement, not a replacement.
        if image_scores:
            for n in sorted(image_scores, key=lambda n: -image_scores[n]):
                if len(roots) >= max(self.roots_per_subquery, len(mentions or [])):
                    break
                if image_scores[n] >= self.image_sim_floor and n not in roots:
                    roots.append(n)

        if len(roots) < self.roots_per_subquery:
            s = self.index.score(subquery)
            scored = {n: float(v) for n, v in zip(self.index.names, s)}
            if image_scores:
                for n in scored:
                    scored[n] = alpha * scored[n] + (1 - alpha) * image_scores.get(n, 0.0)
            for n in sorted(scored, key=lambda n: -scored[n]):
                if n not in roots:
                    roots.append(n)
                if len(roots) >= self.roots_per_subquery:
                    break
        return roots[: max(self.roots_per_subquery, len(mentions or []))]

    def _nav_image_paths(self, page) -> list[str]:
        """Existing files for the first `nav_images` pictures on the page.

        A path that does not resolve is dropped rather than passed on: the server rejects
        the whole request for one missing file, which would turn a missing thumbnail into
        a failed navigation step and a silently truncated walk."""
        out = []
        for im in page.images[: self.nav_images]:
            fp = (Path(self.base_dir) / im["path"]) if self.base_dir else Path(im["path"])
            if fp.exists():
                out.append(str(fp))
        return out

    # -------------------------------------------------------------------------- walk
    def _navigate(self, question: str, chain: Chain) -> dict:
        page = self.g.page(chain.head)
        if page is None:
            return {"answered": False, "cite": [], "follow": []}
        system = (prompts.NAVIGATE_SYSTEM
                  .replace("{max_cite}", str(self.max_cite))
                  .replace("{max_choices}", str(self.max_choices)))
        hide, shots = self.hide_captions, []
        if self.nav_images and page.images:
            # by default looking replaces reading, so that the comparison against the
            # text-only walk isolates the picture rather than measuring one more channel
            if not self.nav_keep_captions:
                hide = hide | {i.get("path") for i in page.images}
            shots = self._nav_image_paths(page)
        user = prompts.NAVIGATE_USER.format(
            question=question, subquery=chain.subquery,
            trail=chain.trail(),
            page=page.render(hide_captions=hide),
        )
        try:
            reply = (self.llm.complete_vision(system, user, shots, max_tokens=250)
                     if shots else
                     self.llm.complete(system, user, max_tokens=250))
        except Exception:
            return {"answered": False, "cite": [], "follow": []}
        obj = _json(reply)
        # the model must copy a link that exists; anything else is a hallucinated edge
        valid = {l.target: (l.relation, l.incoming) for l in page.links}
        nvalid = {}
        for _t in valid:
            nvalid.setdefault(_norm(_t), _t)

        # Accepting only a verbatim target silently discarded 46.9% of the citations
        # Qwen3.8-27B proposed on a 40-item sample: it writes the link the way the page
        # displays it, "uses_method: rakusui", and one page can be cited for several
        # targets at once, "uses_method: a, b, c". Those name real links and were counted
        # as hallucinations, which is why that backbone reached a gold entity on 94.3% of
        # items and still recovered only 0.681 of the gold triples. A name is now resolved
        # by taking the part after the relation prefix, splitting a list, and comparing
        # case- and punctuation-insensitively; anything that still matches no link on the
        # page is a hallucination and is dropped as before.
        def _candidates(raw: str):
            bits = [raw]
            if ":" in raw:
                bits.append(raw.split(":", 1)[1])
            out = []
            for b in bits:
                for piece in b.split(","):
                    piece = piece.strip()
                    if piece:
                        out.append(piece)
            return out

        def resolve(key, cap):
            names = [t for t in (obj.get(key) or []) if isinstance(t, str)]
            picked, seen_t = [], set()
            for raw in names:
                for cand in _candidates(raw):
                    t = cand if cand in valid else nvalid.get(_norm(cand))
                    if t is not None and t not in seen_t:
                        seen_t.add(t)
                        picked.append((valid[t][0], t, valid[t][1]))
                    if len(picked) >= cap:
                        break
                if len(picked) >= cap:
                    break
            return picked[:cap]

        answered = bool(obj.get("answered"))
        cite = resolve("cite", self.max_cite)
        follow = resolve("follow", self.max_choices)

        # "This page answers the question" plus an empty citation list is a contradiction:
        # the model has just asserted that the evidence is here and then named none of it.
        # It says so in its own words -- "founding year and founder are explicitly stated
        # on this page" -- while returning cite: []. No amount of instruction fixed this,
        # and the cost is total: a question whose answer sits on one page retrieves
        # nothing. So the claim is taken at face value and the page's own links are cited
        # for it, ranked by how well each matches what was being looked for.
        if answered and not cite and page.links:
            ranked = self._rank_links(chain.subquery, page)
            cite = ranked[: self.max_cite]
        if self.merge_cite_follow:
            # one undifferentiated selection, as a frontier-style expander would make:
            # everything chosen is both walked to and taken as evidence
            both = cite + [f for f in follow if f not in cite]
            cite, follow = both[: self.max_choices], both[: self.max_choices]
        return {"answered": answered, "cite": cite, "follow": follow,
                "why": str(obj.get("why", ""))[:120]}

    def _rank_links(self, subquery: str, page) -> list[tuple[str, str, bool]]:
        """Order a page's links by agreement with the sub-query."""
        import numpy as np

        links = page.links
        texts = [f"{l.relation.replace('_', ' ')} {l.target}" for l in links]
        emb = np.asarray(self.index.model.encode([subquery] + texts), dtype=np.float32)
        emb /= np.clip(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12, None)
        sims = emb[1:] @ emb[0]
        order = np.argsort(-sims)
        return [(links[i].relation, links[i].target, links[i].incoming) for i in order]

    def walk(self, question: str, subquery: str, mentions: list[str] | None = None,
             image_scores: dict[str, float] | None = None) -> list[Chain]:
        """Best-first beam walk for one sub-query. Returns finished chains."""
        beam = [Chain(r, subquery) for r in self.roots(subquery, mentions, image_scores)]
        finished: list[Chain] = []
        pages_opened = 0

        for _ in range(self.max_depth):
            if not beam or pages_opened >= self.max_pages:
                break
            with ThreadPoolExecutor(max_workers=self.workers) as pool:
                decisions = list(pool.map(lambda c: self._navigate(question, c), beam))
            pages_opened += len(beam)

            nxt: list[Chain] = []
            for chain, d in zip(beam, decisions):
                # Extend first, judge after. The navigator often reports "answered" while
                # standing on the *root* page, because the fact it was sent to find is one
                # link away and plainly visible. Treating that as a stop condition before
                # taking the link finishes a chain with no edges at all, and a chain with
                # no edges carries no evidence -- the search would report success and
                # retrieve nothing.
                # Facts cited on this page are evidence in their own right. Without this
                # the retriever can only ever emit a path, while most question types have
                # star-shaped gold: "what is known about X" is answered by several sibling
                # facts on one page, and a comparison needs one fact from each of two
                # pages. A pure path retriever scores zero on those no matter how well it
                # navigates -- it was answering a different shape of question.
                for rel, tgt, incoming in d["cite"]:
                    triple = (tgt, rel, chain.head) if incoming else (chain.head, rel, tgt)
                    if triple not in chain.cited:
                        chain.cited.append(triple)
                if d["cite"] and chain not in finished:
                    finished.append(chain)

                kids: list[Chain] = []
                for rank, (rel, tgt, incoming) in enumerate(d["follow"]):
                    if tgt in chain.nodes:
                        continue  # no cycles
                    cand = chain.extend(rel, tgt, incoming, 0.0)
                    # rank is the navigator's own preference order; blended with embedding
                    # agreement so a confident but off-topic pick is discounted
                    llm_pref = 1.0 - rank / max(len(d["follow"]), 1)
                    cand.score = self.lam * llm_pref + (1 - self.lam) * self._sim(subquery, cand)
                    kids.append(cand)

                if d["answered"]:
                    # A snapshot, not a stop. The navigator declares victory as soon as it
                    # can see *a* fact the question mentions -- standing on Jugaku Bunko it
                    # reports answered because `located_in: Taka-cho` is right there, even
                    # though the question asks for the prefecture two hops on. Recording
                    # the chain and continuing to walk it costs one more page and keeps the
                    # deep route reachable; treating it as terminal caps every multi-hop
                    # question at one hop, which is precisely the failure the walk exists
                    # to avoid. The join deduplicates the shared prefix.
                    for k in kids:
                        k.answered = True
                        k.why = d.get("why", "")
                    if kids:
                        finished.extend(kids)
                        nxt.extend(kids)
                    elif chain.edges:
                        chain.answered = True
                        chain.why = d.get("why", "")
                        finished.append(chain)
                    continue
                if not kids:
                    if (chain.edges or chain.cited) and chain not in finished:
                        finished.append(chain)   # dead end, but what it holds may stand
                    continue
                nxt.extend(kids)
            nxt.sort(key=lambda c: -c.score)
            beam = nxt[: self.beam]

        finished.extend(c for c in beam if (c.edges or c.cited) and c not in finished)
        return finished

    def _sim(self, subquery: str, chain: Chain) -> float:
        """Cosine agreement between the sub-query and the chain read as a sentence."""
        text = " ".join(f"{h} {r.replace('_',' ')} {u}" for h, r, u in chain.edges)
        if not text:
            return 0.0
        import numpy as np
        qa = self.index.model.encode([subquery, text])
        qa = np.asarray(qa, dtype=np.float32)
        qa /= np.clip(np.linalg.norm(qa, axis=1, keepdims=True), 1e-12, None)
        return float(qa[0] @ qa[1])

    # ------------------------------------------------------------------ verification
    def verify(self, question: str, chain: Chain, with_attrs: bool = True) -> Chain:
        attrs = ""
        if with_attrs:
            bits = []
            for n in chain.nodes:
                p = self.g.page(n)
                if p and p.text:
                    bits.append(f"  {n}: {p.text[0][:160]}")
            if bits:
                attrs = "\nWhat the pages say:\n" + "\n".join(bits) + "\n"
        try:
            reply = self.llm.complete(
                prompts.VERIFY_SYSTEM,
                prompts.VERIFY_USER.format(
                    question=question,
                    subquery=question if self.verify_against_question else chain.subquery,
                    chain=chain.render(), attrs=attrs),
                max_tokens=150,
            )
            obj = _json(reply)
            v = obj.get("verdict")
            chain.verdict = v if v in ("support", "partial", "reject") else "partial"
            chain.why = str(obj.get("why", ""))[:120]
        except Exception:
            chain.verdict = "partial"
        return chain


def _norm(s: str) -> str:
    """Compare link names without case, padding or punctuation differences."""
    import re as _re, unicodedata as _ud
    s = _ud.normalize("NFKC", str(s)).casefold().strip()
    return _re.sub(r"[\s\-_/()\[\]{}.,;:'\"]+", " ", s).strip()


def _json(text: str) -> dict:
    from washi_kg.llm import extract_json

    obj = extract_json(text, default={})
    return obj if isinstance(obj, dict) else {}
