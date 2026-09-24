"""Embedding index for search-root initialisation and path scoring.

Only entity *names* are embedded, not their attributes. A_text runs to several hundred
characters of prose; embedding it makes every well-described entity broadly similar to
every query and the ranking collapses toward whatever the corpus discusses most. The name
is the identity, and the walk is what recovers the context the name lacks.

Image similarity is computed with CLIP when a query carries an image, matching the
contrastive vision-language retrieval the baselines use, so the root stage is comparable
across methods.
"""

from __future__ import annotations

import numpy as np


def _l2(a: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(a, axis=-1, keepdims=True)
    return a / np.clip(n, 1e-12, None)


def _feat(out):
    """transformers 5.x returns an output object where 4.x returned a tensor."""
    for attr in ("image_embeds", "text_embeds", "pooler_output"):
        v = getattr(out, attr, None)
        if v is not None:
            return v
    return out


class TextIndex:
    def __init__(self, names: list[str], model_name: str, device: str | None = None):
        from sentence_transformers import SentenceTransformer

        self.names = names
        self.model = SentenceTransformer(model_name, device=device)
        self.emb = _l2(np.asarray(
            self.model.encode(names, batch_size=256, show_progress_bar=False),
            dtype=np.float32,
        ))

    def score(self, query: str) -> np.ndarray:
        q = _l2(np.asarray(self.model.encode([query]), dtype=np.float32))
        return (self.emb @ q[0]).astype(np.float32)

    def top(self, query: str, k: int) -> list[tuple[str, float]]:
        s = self.score(query)
        idx = np.argsort(-s)[:k]
        return [(self.names[i], float(s[i])) for i in idx]


class ImageIndex:
    """CLIP over A_img, used only when a query supplies an image."""

    def __init__(self, entity_images: dict[str, list[str]], model_name: str,
                 device: str = "cuda", base_dir=None):
        import torch
        from PIL import Image
        from transformers import CLIPModel, CLIPProcessor

        self.torch = torch
        self.device = device
        self.model = CLIPModel.from_pretrained(model_name).to(device).eval()
        self.proc = CLIPProcessor.from_pretrained(model_name)

        self.entities: list[str] = []
        vecs: list[np.ndarray] = []
        for ent, paths in entity_images.items():
            embs = []
            for p in paths:
                fp = (base_dir / p) if base_dir else p
                try:
                    im = Image.open(fp).convert("RGB")
                except Exception:
                    continue
                with torch.no_grad():
                    inp = self.proc(images=im, return_tensors="pt").to(device)
                    embs.append(_feat(self.model.get_image_features(**inp))[0].cpu().numpy())
            if embs:
                # one vector per entity: an entity's images are views of one thing, so the
                # mean is the entity's appearance rather than any single photograph's
                self.entities.append(ent)
                vecs.append(np.mean(embs, axis=0))
        self.emb = _l2(np.asarray(vecs, dtype=np.float32)) if vecs else np.zeros((0, 768), np.float32)

    def score_image(self, image_path) -> dict[str, float]:
        from PIL import Image

        if not len(self.emb):
            return {}
        try:
            im = Image.open(image_path).convert("RGB")
        except Exception:
            return {}
        with self.torch.no_grad():
            inp = self.proc(images=im, return_tensors="pt").to(self.device)
            q = _feat(self.model.get_image_features(**inp))[0].cpu().numpy()
        q = _l2(q[None, :])[0]
        return {e: float(s) for e, s in zip(self.entities, self.emb @ q)}
