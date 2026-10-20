from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Protocol

import jieba
import numpy as np
from rank_bm25 import BM25Okapi

from .models import Chunk, SearchResult


def tokenize(text: str) -> list[str]:
    return [token.strip().lower() for token in jieba.lcut(text) if token.strip()]


class BM25Retriever:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        corpus = [tokenize(chunk.text) for chunk in chunks]
        self._index = BM25Okapi(corpus) if corpus else None

    def search(self, query: str, top_k: int = 3) -> list[SearchResult]:
        if not self._index or not query.strip():
            return []
        scores = np.asarray(self._index.get_scores(tokenize(query)), dtype=float)
        indices = np.argsort(scores)[::-1][: min(top_k, len(self.chunks))]
        return [
            SearchResult(chunk=self.chunks[int(index)], score=float(scores[index]), rank=rank)
            for rank, index in enumerate(indices, start=1)
        ]


class Embedder(Protocol):
    def encode(self, texts: list[str], normalize_embeddings: bool = True) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)

    def encode(self, texts: list[str], normalize_embeddings: bool = True) -> np.ndarray:
        return np.asarray(
            self._model.encode(
                texts,
                normalize_embeddings=normalize_embeddings,
                show_progress_bar=False,
            ),
            dtype=np.float32,
        )


def corpus_digest(chunks: list[Chunk]) -> str:
    payload = json.dumps([chunk.to_dict() for chunk in chunks], ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class VectorRetriever:
    def __init__(
        self,
        chunks: list[Chunk],
        model_name: str,
        index_dir: Path,
        embedder: Embedder | None = None,
    ):
        self.chunks = chunks
        self.model_name = model_name
        self.index_dir = index_dir
        self._embedder = embedder
        self._embeddings: np.ndarray | None = None

    @property
    def embeddings_path(self) -> Path:
        return self.index_dir / "embeddings.npy"

    @property
    def manifest_path(self) -> Path:
        return self.index_dir / "manifest.json"

    def _get_embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = SentenceTransformerEmbedder(self.model_name)
        return self._embedder

    def _cache_is_valid(self) -> bool:
        if not self.embeddings_path.exists() or not self.manifest_path.exists():
            return False
        try:
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        return (
            manifest.get("model_name") == self.model_name
            and manifest.get("corpus_digest") == corpus_digest(self.chunks)
            and manifest.get("chunk_ids") == [chunk.chunk_id for chunk in self.chunks]
        )

    @property
    def cache_is_valid(self) -> bool:
        return self._cache_is_valid()

    def build(self, force: bool = False) -> None:
        if not self.chunks:
            self._embeddings = np.empty((0, 0), dtype=np.float32)
            return
        self.index_dir.mkdir(parents=True, exist_ok=True)
        if not force and self._cache_is_valid():
            self._embeddings = np.load(self.embeddings_path)
            return
        embeddings = self._get_embedder().encode(
            [chunk.text for chunk in self.chunks], normalize_embeddings=True
        )
        self._embeddings = np.asarray(embeddings, dtype=np.float32)
        np.save(self.embeddings_path, self._embeddings)
        manifest = {
            "model_name": self.model_name,
            "corpus_digest": corpus_digest(self.chunks),
            "chunk_ids": [chunk.chunk_id for chunk in self.chunks],
            "dimensions": int(self._embeddings.shape[1]),
        }
        self.manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def search(self, query: str, top_k: int = 3) -> list[SearchResult]:
        if not query.strip() or not self.chunks:
            return []
        if self._embeddings is None:
            self.build()
        assert self._embeddings is not None
        query_vector = self._get_embedder().encode([query], normalize_embeddings=True)[0]
        scores = self._embeddings @ np.asarray(query_vector, dtype=np.float32)
        indices = np.argsort(scores)[::-1][: min(top_k, len(self.chunks))]
        return [
            SearchResult(chunk=self.chunks[int(index)], score=float(scores[index]), rank=rank)
            for rank, index in enumerate(indices, start=1)
        ]

