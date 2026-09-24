"""Creator — graph assembly + multimodal alignment (paper §"Graph Assembly").

The Creator merges canonicalized triples into G (unique entities, relation set, triple
set) and associates multimodal attributes A(e) = {A_text(e), A_img(e)} with entities:

* A_text(e): the source paragraphs in which the entity was asserted.
* A_img(e):  images aligned to the entity. The paper defines align(e) mapping a
             canonical entity to a predefined multimodal entity (or null). Here align
             is realised by a vision LLM: each image is matched to the best candidate
             entity (pre-filtered by caption/name embedding similarity) and its caption
             is stored as evidence.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import prompts
from .embeddings import Embedder, cosine_matrix
from .extractor import Candidate
from .graph import KnowledgeGraph
from .llm import LLMClient, extract_json


class Creator:
    def __init__(
        self,
        llm: LLMClient,
        embedder: Embedder,
        *,
        enable_image_alignment: bool,
        image_candidate_sim_threshold: float,
        max_image_candidates: int,
        max_text_attrs_per_entity: int,
        max_text_attr_chars: int,
        compress_mode: str = "extractive",
        max_img_attrs_per_entity: int | None = None,
        align_max_tokens: int = 300,
    ):
        self.llm = llm
        self.embedder = embedder
        self.enable_image_alignment = enable_image_alignment
        self.img_sim_threshold = image_candidate_sim_threshold
        self.max_image_candidates = max_image_candidates
        self.max_text_attrs = max_text_attrs_per_entity
        self.max_text_attr_chars = max_text_attr_chars
        self.compress_mode = compress_mode
        self.max_img_attrs = max_img_attrs_per_entity
        # align(e) answers with one short JSON object; the global max_tokens (sized for
        # triple extraction) would let a verbose backend run on for thousands of tokens.
        self.align_max_tokens = align_max_tokens
        # Entity-name embeddings, reused across images. E only grows when the Detector
        # runs, so during an alignment pass the same matrix serves every image; without
        # this the whole entity set is re-encoded once per image.
        self._name_emb: tuple[tuple[str, ...], object] | None = None

    # ----------------------------------------------------------------- triples
    def merge_triples(
        self,
        triples: list[Candidate],
        mention_to_canon: dict[str, str],
        graph: KnowledgeGraph,
    ) -> int:
        """Add triples to G, record aliases, and attach textual attributes."""
        new_count = 0
        for c in triples:
            if graph.add_triple(c.h, c.r, c.u, source=c.source):
                new_count += 1
            graph.add_text_attr(c.h, c.context, self.max_text_attrs, self.max_text_attr_chars)
            graph.add_text_attr(c.u, c.context, self.max_text_attrs, self.max_text_attr_chars)

        # record surface forms that were merged into a different canonical name
        for mention, canon in mention_to_canon.items():
            if mention != canon:
                graph.add_alias(canon, mention)
        return new_count

    # -------------------------------------------------------- abstractive A_text
    def compress_entity_text(self, name: str, graph: KnowledgeGraph) -> None:
        """Rewrite an entity's A_text evidence into one concise description (LLM).

        Only runs when ``compress_mode == 'llm'``. Replaces the entity's text list
        with a single <= max_text_attr_chars summary grounded in the existing
        evidence. Falls back to the extractive result on any failure.
        """
        if self.compress_mode != "llm":
            return
        ent = graph.entities.get(name)
        if not ent or not ent["attributes"]["text"]:
            return
        evidence = "\n".join(ent["attributes"]["text"])
        try:
            summary = self.llm.complete(
                prompts.COMPRESS_SYSTEM.format(max_chars=self.max_text_attr_chars),
                prompts.COMPRESS_USER.format(
                    entity=name, evidence=evidence, max_chars=self.max_text_attr_chars
                ),
            ).strip().strip('"')
        except Exception:
            return  # keep the extractive text
        if summary:
            from .graph import compress_text  # enforce the hard cap regardless

            summary = compress_text(summary, [name] + ent.get("aliases", []), self.max_text_attr_chars)
            ent["attributes"]["text"] = [summary]

    # ------------------------------------------------------------------ images
    def candidate_entities(
        self, graph: KnowledgeGraph, context_text: str | None, filename_stem: str | None
    ) -> list[str]:
        """Rank entities for an image. Entities literally mentioned in the image's
        surrounding text (e.g. the PDF page it sits on) come first — that locality is
        the strongest signal for which entity the image illustrates — followed by
        embedding-similar entities, then the rest, capped at max_image_candidates.
        """
        names = list(graph.entities.keys())
        if not names:
            return []

        mentioned: list[str] = []
        if context_text:
            low = context_text.lower()
            for n in names:
                forms = [n] + graph.entities[n].get("aliases", [])
                if any(f.lower() in low for f in forms):
                    mentioned.append(n)

        ranked: list[str] = []
        query = (context_text or filename_stem or "").strip()
        if query:
            key = tuple(names)
            if self._name_emb is None or self._name_emb[0] != key:
                self._name_emb = (key, self.embedder.encode(names))
            sims = cosine_matrix(self.embedder.encode([query]), self._name_emb[1])[0]
            for i in np.argsort(-sims):
                if sims[i] >= self.img_sim_threshold:
                    ranked.append(names[i])

        out: list[str] = []
        for n in mentioned + ranked + names:
            if n not in out:
                out.append(n)
            if len(out) >= self.max_image_candidates:
                break
        return out

    def align_images(
        self,
        items: list[dict],
        graph: KnowledgeGraph,
        *,
        workers: int = 16,
        on_result=None,
        caption_retry: bool = True,
    ) -> list[str | None]:
        """Align many images, issuing the vision calls concurrently.

        Serial alignment leaves the GPU idle between calls; a batched server handles
        many in flight for nearly the wall-clock of one. Only the model calls are
        parallel — candidate selection reads the graph and the accepted results are
        merged on this thread, so ``graph`` is never mutated from a worker.

        Two rounds, because the two failure modes are different. Round one scopes
        candidates by the text surrounding the image, which is the strongest signal
        when it exists — but a slide's text is usually its heading, not a description
        of the photograph on it, so the right entity is often never offered and the
        model can only answer null. Round two re-scopes using the caption the model
        just wrote *about the image itself*, which describes the subject directly, and
        asks again over that candidate set. Only images rejected in round one are
        retried, and only when the new candidates actually differ.

        Each item is ``{path, stored, context_text?, source?, page?}``. Returns the
        aligned entity per item, in order, ``None`` where nothing matched.
        """
        from concurrent.futures import ThreadPoolExecutor

        if not self.enable_image_alignment or not graph.entities:
            return [None] * len(items)

        prepared = []
        for it in items:
            path = Path(it["path"])
            stem = path.stem.replace("_", " ").replace("-", " ")
            cands = self.candidate_entities(graph, it.get("context_text"), stem)
            prepared.append((it, path, cands))

        def ask(job) -> dict | None:
            _it, path, cands = job
            if not cands:
                return None
            try:
                return self._ask_vision(path, cands)
            except Exception:
                return None  # one unreadable image must not abort the batch

        with ThreadPoolExecutor(max_workers=workers) as pool:
            first = list(pool.map(ask, prepared))

            retries: list[tuple[int, tuple]] = []
            if caption_retry:
                for i, (obj, (it, path, cands)) in enumerate(zip(first, prepared)):
                    if obj and obj.get("entity"):
                        continue  # already matched
                    caption = (obj or {}).get("caption")
                    if not isinstance(caption, str) or not caption.strip():
                        continue  # nothing to re-query with
                    recands = self.candidate_entities(graph, caption, None)
                    if not recands or recands == cands:
                        continue  # the same offer would get the same answer
                    retries.append((i, (it, path, recands)))

            if retries:
                for (i, _job), obj in zip(retries, pool.map(ask, [j for _, j in retries])):
                    if obj and obj.get("entity"):
                        first[i] = obj

        out: list[str | None] = []
        for (it, _path, _c), obj in zip(prepared, first):
            ent = self._store_alignment(obj, it, graph)
            out.append(ent)
            if on_result is not None:
                on_result(it, ent)
        return out

    def _ask_vision(self, image_path: Path, candidates: list[str]) -> dict | None:
        listing = "\n".join(f"- {c}" for c in candidates)
        reply = self.llm.complete_vision(
            prompts.VISION_ALIGN_SYSTEM,
            prompts.VISION_ALIGN_USER.format(candidates=listing),
            [str(image_path)],
            max_tokens=self.align_max_tokens,
        )
        obj = extract_json(reply, default={})
        return obj if isinstance(obj, dict) else None

    def _store_alignment(self, obj: dict | None, item: dict, graph: KnowledgeGraph) -> str | None:
        if not obj:
            return None
        entity, caption = obj.get("entity"), obj.get("caption")
        if not (isinstance(entity, str) and entity in graph.entities):
            return None
        stored = graph.add_image_attr(
            entity,
            item["stored"],
            caption if isinstance(caption, str) else None,
            source=item.get("source"),
            page=item.get("page"),
            cap=self.max_img_attrs,
        )
        return entity if stored else None

    def align_image(
        self,
        image_path: Path,
        stored_path: str,
        graph: KnowledgeGraph,
        *,
        context_text: str | None = None,
        source: str | None = None,
        page: int | None = None,
    ) -> str | None:
        """Realise align(e) for one image via the vision LLM. Returns the entity or None.

        ``context_text`` is the text near the image (PDF page text for extracted images),
        used to scope candidate entities. ``source``/``page`` are recorded on A_img.
        """
        if not self.enable_image_alignment or not graph.entities:
            return None

        stem = image_path.stem.replace("_", " ").replace("-", " ")
        candidates = self.candidate_entities(graph, context_text, stem)
        if not candidates:
            return None

        obj = self._ask_vision(image_path, candidates)
        return self._store_alignment(
            obj, {"stored": stored_path, "source": source, "page": page}, graph
        )
