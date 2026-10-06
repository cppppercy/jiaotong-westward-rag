from pathlib import Path

import numpy as np

from westward_rag.models import Chunk
from westward_rag.retrieval import BM25Retriever, VectorRetriever


class FakeEmbedder:
    def __init__(self) -> None:
        self.calls = 0

    def encode(self, texts: list[str], normalize_embeddings: bool = True) -> np.ndarray:
        self.calls += 1
        vectors = []
        for text in texts:
            vector = np.array(
                [float("西迁" in text), float("电机" in text), float("食堂" in text)],
                dtype=np.float32,
            )
            if not vector.any():
                vector[2] = 0.1
            if normalize_embeddings:
                vector = vector / np.linalg.norm(vector)
            vectors.append(vector)
        return np.asarray(vectors)


def make_chunks() -> list[Chunk]:
    return [
        Chunk("C-1", "a.md", "a.md", "交通大学西迁支持西北建设。", "片段 1"),
        Chunk("C-2", "b.md", "b.md", "电机实验室开展科学研究。", "片段 1"),
        Chunk("C-3", "c.md", "c.md", "食堂供应日常饭菜。", "片段 1"),
    ]


def test_bm25_returns_ranked_results() -> None:
    results = BM25Retriever(make_chunks()).search("西迁 西北", top_k=2)
    assert len(results) == 2
    assert results[0].chunk.chunk_id == "C-1"
    assert results[0].score >= results[1].score


def test_vector_index_persists_and_reloads(tmp_path: Path) -> None:
    first_embedder = FakeEmbedder()
    first = VectorRetriever(make_chunks(), "fake-model", tmp_path, first_embedder)
    first.build()
    assert (tmp_path / "embeddings.npy").exists()
    assert (tmp_path / "manifest.json").exists()
    assert first.search("西迁", top_k=1)[0].chunk.chunk_id == "C-1"

    second_embedder = FakeEmbedder()
    second = VectorRetriever(make_chunks(), "fake-model", tmp_path, second_embedder)
    second.build()
    assert second_embedder.calls == 0
    assert second.search("电机", top_k=1)[0].chunk.chunk_id == "C-2"

