"""WashiMMKG construction pipeline.

A faithful implementation of the multimodal knowledge-graph construction described
in the paper "Enhancing Washi Reasoning in MLLMs via Multimodal Knowledge Graphs
and Agentic RAG":

    Extractor -> Filter -> Detector -> Creator

The pipeline is incremental: dropping new files into ``raw/`` and re-running expands
the existing graph rather than rebuilding it from scratch.
"""

from .config import Config, load_config
from .graph import KnowledgeGraph
from .pipeline import Pipeline

__all__ = ["Config", "load_config", "KnowledgeGraph", "Pipeline"]
