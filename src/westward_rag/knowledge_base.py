from __future__ import annotations

from pathlib import Path

from .chunking import chunk_sections
from .config import Settings
from .loaders import discover_documents, load_directory
from .models import Chunk, SearchResult
from .retrieval import BM25Retriever, Embedder, VectorRetriever


class KnowledgeBase:
    def __init__(
        self,
        chunks: list[Chunk],
        settings: Settings,
        embedder: Embedder | None = None,
    ):
        self.chunks = chunks
        self.settings = settings
        self.bm25 = BM25Retriever(chunks)
        self.vector = VectorRetriever(
            chunks,
            model_name=settings.embedding_model,
            index_dir=settings.index_dir,
            embedder=embedder,
        )

    @classmethod
    def from_directory(
        cls,
        settings: Settings,
        directory: Path | None = None,
        embedder: Embedder | None = None,
    ) -> "KnowledgeBase":
        document_dir = directory or settings.documents_dir
        sections = load_directory(document_dir)
        chunks = chunk_sections(
            sections,
            chunk_size=settings.chunk_size,
            overlap=settings.chunk_overlap,
        )
        return cls(chunks, settings, embedder=embedder)

    @property
    def document_paths(self) -> list[Path]:
        return discover_documents(self.settings.documents_dir)

    def build_vector_index(self, force: bool = False) -> None:
        self.vector.build(force=force)

    def search(self, query: str, method: str, top_k: int | None = None) -> list[SearchResult]:
        limit = top_k or self.settings.top_k
        if method == "bm25":
            return self.bm25.search(query, limit)
        if method == "vector":
            return self.vector.search(query, limit)
        raise ValueError("检索方式必须是 bm25 或 vector。")

