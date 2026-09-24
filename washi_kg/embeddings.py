"""Sentence embeddings for the Filter (semantic dedup) and Detector (canonicalization).

The paper uses a Sentence-BERT-style model; we default to all-MiniLM-L6-v2 and expose
cosine similarity helpers.
"""

from __future__ import annotations

import numpy as np

from .config import Config


class Embedder:
    def __init__(self, config: Config):
        from sentence_transformers import SentenceTransformer

        model_name = config.get(
            "embeddings", "model", default="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        """Return L2-normalised embeddings of shape ``(len(texts), dim)``."""
        if not texts:
            return np.zeros((0, self.model.get_sentence_embedding_dimension()), dtype=np.float32)
        return self.model.encode(
            texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
        ).astype(np.float32)


def cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity of every row in ``a`` against every row in ``b``.

    Assumes both are L2-normalised (as returned by :meth:`Embedder.encode`).
    """
    if a.shape[0] == 0 or b.shape[0] == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    return a @ b.T


def triple_text(h: str, r: str, u: str) -> str:
    """Natural-language rendering of a triple, used for embedding-based similarity."""
    return f"{h} {r.replace('_', ' ')} {u}"
