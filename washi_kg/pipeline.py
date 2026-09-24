"""Pipeline orchestration: Extractor -> Filter -> Detector -> Creator.

Incremental by design. On each run:

1. Scan ``raw/`` and select files whose hash is new or changed (Manifest).
2. Extract triples from new text files (Extractor)             -> T_raw
3. Semantically dedup against the existing graph + run (Filter) -> T_filtered
4. Canonicalize entities against E already in the graph (Detector) -> T
5. Merge triples + textual attributes, then align new images (Creator).
6. Persist the expanded graph and update the manifest.

Adding a file later and re-running expands the same graph rather than rebuilding it.
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path

from .config import Config
from .creator import Creator
from .detector import Detector
from .embeddings import Embedder
from .extractor import Extractor
from .filter import Filter
from .graph import KnowledgeGraph
from .ingest import iter_source_files, load_text_doc
from .llm import LLMClient
from .state import Manifest, file_hash


def _now() -> str:
    return _dt.datetime.now().isoformat(timespec="seconds")


class Pipeline:
    def __init__(self, config: Config, verbose: bool = True):
        self.cfg = config
        self.verbose = verbose

        self.llm = LLMClient(config)
        self.embedder = Embedder(config)

        self.extractor = Extractor(self.llm)
        self.filter = Filter(self.embedder, delta=float(config.get("filter", "delta", default=0.92)))
        self.detector = Detector(
            self.llm,
            self.embedder,
            candidate_sim_threshold=float(
                config.get("detector", "candidate_sim_threshold", default=0.60)
            ),
            max_candidates=int(config.get("detector", "max_candidates", default=8)),
        )
        self.creator = Creator(
            self.llm,
            self.embedder,
            enable_image_alignment=bool(
                config.get("creator", "enable_image_alignment", default=True)
            ),
            image_candidate_sim_threshold=float(
                config.get("creator", "image_candidate_sim_threshold", default=0.20)
            ),
            max_image_candidates=int(config.get("creator", "max_image_candidates", default=15)),
            max_text_attrs_per_entity=int(
                config.get("creator", "max_text_attrs_per_entity", default=8)
            ),
            max_text_attr_chars=int(
                config.get("creator", "max_text_attr_chars", default=200)
            ),
            compress_mode=str(config.get("creator", "compress_mode", default="extractive")),
            max_img_attrs_per_entity=int(
                config.get("creator", "max_img_attrs_per_entity", default=8)
            ),
        )
        self.image_exts = set(config.get("creator", "image_exts", default=[]))
        self.min_chars = int(config.get("extractor", "min_paragraph_chars", default=120))
        self.max_chars = int(config.get("extractor", "max_paragraph_chars", default=2000))
        self.extract_doc_images = bool(config.get("creator", "extract_doc_images", default=True))
        self.images_dir = config.output_dir / config.get(
            "paths", "images_dir", default="extracted_images"
        )

    def _log(self, *a):
        if self.verbose:
            print(*a, flush=True)

    # ------------------------------------------------------------------- build
    def build(self, force: bool = False) -> KnowledgeGraph:
        cfg = self.cfg
        cfg.raw_dir.mkdir(parents=True, exist_ok=True)

        graph = KnowledgeGraph.load(cfg.graph_path)
        manifest = Manifest(cfg.manifest_path)

        text_files: list[Path] = []
        image_files: list[Path] = []
        for path, kind in iter_source_files(cfg.raw_dir, self.image_exts):
            rel = str(path.relative_to(cfg.raw_dir))
            digest = file_hash(path)
            if not force and manifest.is_processed(rel, digest):
                continue
            (text_files if kind == "text" else image_files).append(path)

        if not text_files and not image_files:
            self._log("Nothing new in raw/. Graph is up to date.")
            self._log(self._fmt_stats(graph.stats()))
            return graph

        self._log(f"New/changed: {len(text_files)} text file(s), {len(image_files)} image(s).")

        # ---- text files: Extractor -> Filter -> Detector -> Creator ----------
        for path in text_files:
            rel = str(path.relative_to(cfg.raw_dir))
            self._log(f"\n[text] {rel}")
            doc = load_text_doc(path, cfg.raw_dir, self.min_chars, self.max_chars)

            raw_candidates = []
            for i, para in enumerate(doc.paragraphs):
                cands = self.extractor.extract_paragraph(para, source=rel)
                raw_candidates.extend(cands)
                self._log(f"  Extractor: para {i + 1}/{len(doc.paragraphs)} -> {len(cands)} triples")

            self._log(f"  T_raw = {len(raw_candidates)}")
            filtered = self.filter.filter(raw_candidates, graph)
            self._log(f"  Filter -> T_filtered = {len(filtered)}")

            canon, mapping = self.detector.canonicalize(filtered, graph)
            self._log(f"  Detector -> {len(canon)} canonical triples, "
                      f"{len(set(mapping.values()))} canonical entities touched")

            new_triples = self.creator.merge_triples(canon, mapping, graph)
            self._log(f"  Creator -> +{new_triples} new triples in T")

            # abstractive A_text compression for entities touched by this file
            if self.creator.compress_mode == "llm":
                touched = set(mapping.values())
                for ent_name in touched:
                    self.creator.compress_entity_text(ent_name, graph)
                self._log(f"  Creator -> LLM-compressed A_text for {len(touched)} entities")

            # pull embedded images out of PDFs and link them to entities by page locality
            if self.extract_doc_images and path.suffix.lower() == ".pdf":
                from .ingest import extract_pdf_images

                extracted = extract_pdf_images(path, rel, self.images_dir)
                linked = 0
                for ex in extracted:
                    stored = str(ex.path.relative_to(self.cfg.base_dir))
                    ent = self.creator.align_image(
                        ex.path, stored, graph,
                        context_text=ex.page_text, source=ex.source, page=ex.page,
                    )
                    if ent:
                        linked += 1
                self._log(
                    f"  Creator -> extracted {len(extracted)} image(s) from PDF, "
                    f"linked {linked} to entities"
                )

            manifest.record(rel, file_hash(path), n_triples=len(canon), processed_at=_now())
            # persist after each file so a long run is resumable
            graph.save(cfg.graph_path)
            manifest.save()

        # ---- standalone image files in raw/: Creator multimodal alignment ----
        for path in image_files:
            rel = str(path.relative_to(cfg.raw_dir))
            entity = self.creator.align_image(path, rel, graph, source=rel)
            self._log(f"[image] {rel} -> {entity if entity else '(no match)'}")
            manifest.record(rel, file_hash(path), n_triples=0, processed_at=_now())

        graph.meta["updated"] = _now()
        graph.save(cfg.graph_path)
        manifest.save()

        self._log("\nDone. " + self._fmt_stats(graph.stats()))
        return graph

    @staticmethod
    def _fmt_stats(s: dict) -> str:
        return (
            f"Graph: {s['entities']} entities, {s['relations']} relations, "
            f"{s['triples']} triples | {s['multimodal_entities']} multimodal entities, "
            f"{s['images']} images, {s['text_attributes']} text attributes."
        )
