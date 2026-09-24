"""Comparison methods, sharing the wiki-walk's evidence budget.

All retrieval baselines hand the answer model at most K triples, the same K the walk
gets, so the comparison isolates *which* evidence a method finds rather than how much it
is permitted to show.

- Direct      no evidence at all; measures parametric knowledge.
- Know / Know+  the ground-truth evidence handed over directly, text-only and with images.
                Not a competitor: an upper bound on what the answer model can do once
                retrieval is perfect, which is what separates retrieval failure from
                reasoning failure in the results.
- FMT-RAG     the graph flattened to independent triples, ranked by CLIP similarity.
              Uses no topology, so it measures whether similarity matching alone suffices.
- GraphRAG    k-hop ego-subgraphs as retrieval units, ranked the same way. Uses topology
              but no query decomposition, no LLM-guided expansion, and no verification --
              it isolates what the agentic control in the walk is actually worth.
"""

from __future__ import annotations

import numpy as np

from . import prompts
from .page import WikiGraph


def _l2(a: np.ndarray) -> np.ndarray:
    return a / np.clip(np.linalg.norm(a, axis=-1, keepdims=True), 1e-12, None)


def _feat(out):
    """transformers 5.x returns an output object where 4.x returned a tensor."""
    for attr in ("text_embeds", "image_embeds", "pooler_output", "last_hidden_state"):
        v = getattr(out, attr, None)
        if v is not None:
            return v if v.dim() == 2 else v[:, 0]
    return out


def triple_text(h: str, r: str, u: str) -> str:
    return f"{h} {r.replace('_', ' ')} {u}"


def _unit_images(graph: WikiGraph, entities: list[str], base_dir) -> list[str]:
    """Existing image files for the entities an evidence unit covers."""
    from pathlib import Path

    out: list[str] = []
    for e in entities:
        for im in graph.images_of(e):
            fp = Path(base_dir) / im["path"] if base_dir else Path(im["path"])
            if fp.exists():
                out.append(str(fp))
    return out


class ClipRanker:
    """CLIP over evidence units, using both towers.

    Each unit is encoded as text; where the entities it covers carry images, those are
    encoded too and fused into the unit's vector. Text-only encoding would understate
    these baselines on a multimodal graph -- and understating a baseline overstates the
    method compared against it, which is the failure mode worth avoiding here.
    """

    def __init__(self, units: list[str], model_name: str, device: str = "cuda",
                 batch: int = 256, unit_images: list[list[str]] | None = None,
                 gamma: float = 0.7):
        self.gamma = gamma
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self.torch = torch
        self.device = device
        self.model = CLIPModel.from_pretrained(model_name).to(device).eval()
        self.proc = CLIPProcessor.from_pretrained(model_name)
        self.units = units
        embs = []
        for i in range(0, len(units), batch):
            chunk = units[i : i + batch]
            with torch.no_grad():
                inp = self.proc(text=chunk, return_tensors="pt", padding=True,
                                truncation=True, max_length=77).to(device)
                embs.append(_feat(self.model.get_text_features(**inp)).cpu().numpy())
        text_emb = _l2(np.concatenate(embs, axis=0).astype(np.float32)) if embs \
            else np.zeros((0, 768), np.float32)

        if not unit_images or not any(unit_images):
            self.emb = text_emb
            return

        from PIL import Image

        cache: dict[str, np.ndarray] = {}
        fused = text_emb.copy()
        for i, paths in enumerate(unit_images):
            vecs = []
            for path in paths[:3]:
                if path not in cache:
                    try:
                        with torch.no_grad():
                            im = Image.open(path).convert("RGB")
                            pin = self.proc(images=im, return_tensors="pt").to(device)
                            cache[path] = _feat(
                                self.model.get_image_features(**pin)
                            )[0].cpu().numpy()
                    except Exception:
                        cache[path] = None
                v = cache[path]
                if v is not None:
                    vecs.append(v)
            if vecs:
                img = _l2(np.mean(vecs, axis=0)[None, :].astype(np.float32))[0]
                fused[i] = gamma * text_emb[i] + (1.0 - gamma) * img
        self.emb = _l2(fused)

    def rank(self, query: str, k: int, query_images=None) -> list[int]:
        """Rank units against the query, using its image too when it has one.

        A visual question carries q_m, and CLIP can encode it, so withholding it from the
        baselines would make "the image can act as an index" a statement about which
        method was allowed to try rather than about whether it works. The same fusion is
        used on the query side as on the unit side.
        """
        from PIL import Image

        with self.torch.no_grad():
            inp = self.proc(text=[query], return_tensors="pt", padding=True,
                            truncation=True, max_length=77).to(self.device)
            q = _l2(_feat(self.model.get_text_features(**inp)).cpu().numpy()
                    .astype(np.float32))[0]

        vecs = []
        for path in (query_images or [])[:2]:
            try:
                with self.torch.no_grad():
                    im = Image.open(path).convert("RGB")
                    pin = self.proc(images=im, return_tensors="pt").to(self.device)
                    vecs.append(_feat(self.model.get_image_features(**pin))[0]
                                .cpu().numpy())
            except Exception:
                continue
        if vecs:
            qi = _l2(np.mean(vecs, axis=0)[None, :].astype(np.float32))[0]
            q = _l2((self.gamma * q + (1.0 - self.gamma) * qi)[None, :])[0]
        return list(np.argsort(-(self.emb @ q))[:k])


class FlatTripletRAG:
    """FMT-RAG: every triple is an isolated evidence unit."""

    name = "FMT"

    def __init__(self, graph: WikiGraph, ranker_model: str, device: str = "cuda",
                 base_dir=None):
        self.g = graph
        self.triples = [(t["h"], t["r"], t["u"]) for t in graph.T]
        imgs = [_unit_images(graph, [h, u], base_dir) for h, _, u in self.triples]
        self.ranker = ClipRanker([triple_text(*t) for t in self.triples],
                                 ranker_model, device, unit_images=imgs)

    def retrieve(self, question: str, budget: int,
                 query_images=None) -> list[tuple[str, str, str]]:
        return [self.triples[i]
                for i in self.ranker.rank(question, budget, query_images)]


class GraphEgoRAG:
    """Graph RAG: k-hop ego-subgraphs are the evidence units."""

    name = "Graph"

    def __init__(self, graph: WikiGraph, ranker_model: str, device: str = "cuda",
                 hops: int = 1, max_unit_triples: int = 12, base_dir=None):
        self.g = graph
        self.units: list[list[tuple[str, str, str]]] = []
        texts: list[str] = []
        imgs: list[list[str]] = []
        for name in graph.E:
            edges = [(name, r, u) for r, u in graph.out.get(name, [])]
            edges += [(h, r, name) for r, h in graph.inn.get(name, [])]
            if not edges:
                continue
            edges = edges[:max_unit_triples]
            self.units.append(edges)
            texts.append(name + ": " + "; ".join(triple_text(*e) for e in edges))
            imgs.append(_unit_images(graph, [name], base_dir))
        self.ranker = ClipRanker(texts, ranker_model, device, unit_images=imgs)

    def retrieve(self, question: str, budget: int,
                 query_images=None) -> list[tuple[str, str, str]]:
        out: list[tuple[str, str, str]] = []
        seen: set = set()
        for i in self.ranker.rank(question, 3, query_images):
            for e in self.units[i]:
                if e not in seen:
                    seen.add(e)
                    out.append(e)
                if len(out) >= budget:
                    return out
        return out


def answer_from_evidence(llm, question: str, evidence: str, image_paths=None) -> str:
    user = prompts.ANSWER_USER.format(question=question, evidence=evidence)
    if image_paths:
        return llm.complete_vision(prompts.ANSWER_SYSTEM, user, image_paths[:3],
                                   max_tokens=200).strip()
    return llm.complete(prompts.ANSWER_SYSTEM, user, max_tokens=200).strip()


# Mirrors ANSWER_SYSTEM's commit instruction. The closed-book setting is the reference
# point every retrieval number is read against, so it has to be measured under the same
# answering policy: a baseline told to decline when unsure scores near zero for reasons
# that have nothing to do with what it knows, which would inflate every gain over it.
DIRECT_SYSTEM = """\
Answer the question about Washi (traditional Japanese paper) from your own knowledge.
You have no reference material.

Always commit to an answer: lead with your best guess in the first sentence, even if you
are unsure. You may add a brief hedge afterwards, but never let the hedge replace the
answer. Answer in at most 60 words.
"""


def direct_answer(llm, question: str, image_paths=None) -> str:
    if image_paths:
        return llm.complete_vision(DIRECT_SYSTEM, question, image_paths[:2],
                                   max_tokens=200).strip()
    return llm.complete(DIRECT_SYSTEM, question, max_tokens=200).strip()


class ReActRAG:
    """A generic agentic retriever over the same graph: search, expand, keep, finish.

    This exists to answer one question the main table cannot: how much of WikiWalk's gain
    is agency and how much is its design. Every other baseline answers in a single call,
    so "agentic beats one-shot" and "this agent beats other agents" are confounded. This
    one is given the same graph, backbone, evidence budget and roughly the same number of
    calls, and none of the structure -- the page is not the unit of observation, citing is
    not separated from following, there is no beam over partial chains, no chain-level
    verification and no sub-query decomposition.

    Evidence is what the agent asked to keep, in the order it kept it, truncated to the
    budget. Nothing re-ranks it afterwards: a ranker at the end would put back part of
    what is being ablated.
    """

    name = "ReAct"

    def __init__(self, graph: WikiGraph, llm, index, steps: int = 16,
                 max_facts_shown: int = 30, base_dir=None, image_index=None):
        self.g = graph
        self.llm = llm
        self.index = index
        # optional: the CLIP image index WikiWalk uses for image roots, offered to the
        # agent as an `image_search` action so the entrance is not what separates them
        self.image_index = image_index
        self.steps = steps
        self.max_facts_shown = max_facts_shown
        self.base_dir = base_dir
        self.last_calls = 0

    def _facts(self, entity: str) -> list[tuple[str, str, str]]:
        out = [(entity, r, u) for r, u in self.g.out.get(entity, [])]
        out += [(h, r, entity) for r, h in self.g.inn.get(entity, [])]
        return out[: self.max_facts_shown]

    def retrieve(self, question: str, budget: int,
                 query_images=None) -> list[tuple[str, str, str]]:
        from wikirag.walker import _json

        history: list[str] = []
        # Every fact shown so far, numbered once and permanently. It used to be only the
        # last observation, which meant a fact had to be saved in the very next turn or
        # become unreachable: expanding the second entity of a comparison overwrote the
        # first one's facts. Tracing 40 comparative items found that in every one of them
        # ReAct expanded both entities and was shown the gold fact, and in every one of
        # them at least one gold fact was never saved -- the agent looked, it just could
        # no longer reach what it had seen. Vanilla ReAct has no "keep" action at all and
        # answers from the whole trajectory, so that ceiling was this harness's, not the
        # method's, and it cost the baseline most of the comparative template.
        catalogue: list[tuple[str, str, str]] = []
        kept: list[tuple[str, str, str]] = []
        seen: set = set()
        self.last_calls = 0

        system = prompts.REACT_SYSTEM.replace("{steps}", str(self.steps)) \
                                     .replace("{budget}", str(budget))
        can_image_search = bool(self.image_index is not None and query_images)
        if can_image_search:
            marker = '  {"thought": "...", "action": "expand"'
            system = system.replace(marker, prompts.REACT_IMAGE_ACTION + marker, 1)
        for _ in range(self.steps):
            user = prompts.REACT_USER.format(
                question=question,
                history=("\n".join(history) + "\n") if history else "")
            try:
                reply = self.llm.complete(system, user, max_tokens=250)
            except Exception:
                break
            self.last_calls += 1
            obj = _json(reply)
            action = str(obj.get("action", "")).strip().lower()

            for i in obj.get("keep") or []:
                try:
                    fact = catalogue[int(i) - 1]
                except (ValueError, TypeError, IndexError):
                    continue
                if fact not in seen:
                    seen.add(fact)
                    kept.append(fact)

            if action == "finish" or not action:
                break
            if action == "search":
                hits = [n for n, _ in self.index.top(str(obj.get("arg", "")), 8)]
                history.append(f"search({obj.get('arg')}) -> "
                               + (", ".join(hits) if hits else "nothing"))
            elif action == "image_search" and can_image_search:
                scores: dict = {}
                for path in query_images[:2]:
                    for ent, sc in self.image_index.score_image(path).items():
                        scores[ent] = max(scores.get(ent, -1.0), sc)
                hits = [e for e, _ in sorted(scores.items(), key=lambda x: -x[1])[:8]]
                history.append("image_search() -> "
                               + (", ".join(hits) if hits else "nothing"))
            elif action == "expand":
                name = self.g.resolve(str(obj.get("arg", "")))
                if name is None:
                    history.append(f"expand({obj.get('arg')}) -> no such entity")
                else:
                    new = self._facts(name)
                    start = len(catalogue)
                    catalogue.extend(new)
                    listing = "\n".join(f"  {i}. {triple_text(*f)}"
                                        for i, f in enumerate(new, start + 1)) or "  (none)"
                    history.append(f"expand({name}) ->\n{listing}")
            else:
                history.append(f"unknown action {action!r}")

        return kept[:budget]
