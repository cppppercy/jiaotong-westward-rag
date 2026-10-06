from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    project_root: Path = PROJECT_ROOT
    documents_dir: Path = PROJECT_ROOT / "data" / "documents"
    index_dir: Path = PROJECT_ROOT / "data" / "index"
    eval_dir: Path = PROJECT_ROOT / "eval"
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    top_k: int = 3
    bm25_min_score: float = 1.5
    vector_min_similarity: float = 0.45
    chunk_size: int = 600
    chunk_overlap: int = 80
    llm_api_key: str = ""
    llm_base_url: str = "https://api.openai.com/v1"
    llm_model: str = "gpt-4o-mini"

    @classmethod
    def from_env(cls, project_root: Path | None = None) -> "Settings":
        root = project_root or PROJECT_ROOT
        load_dotenv(root / ".env")
        return cls(
            project_root=root,
            documents_dir=root / "data" / "documents",
            index_dir=root / "data" / "index",
            eval_dir=root / "eval",
            embedding_model=os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5"),
            top_k=int(os.getenv("TOP_K", "3")),
            bm25_min_score=float(os.getenv("BM25_MIN_SCORE", "1.5")),
            vector_min_similarity=float(os.getenv("VECTOR_MIN_SIMILARITY", "0.45")),
            chunk_size=int(os.getenv("CHUNK_SIZE", "600")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "80")),
            llm_api_key=os.getenv("LLM_API_KEY", "").strip(),
            llm_base_url=os.getenv("LLM_BASE_URL", "https://api.openai.com/v1").strip(),
            llm_model=os.getenv("LLM_MODEL", "gpt-4o-mini").strip(),
        )

    def validate(self) -> None:
        if self.top_k < 1:
            raise ValueError("TOP_K 必须大于等于 1。")
        if self.chunk_size < 100:
            raise ValueError("CHUNK_SIZE 不能小于 100。")
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("CHUNK_OVERLAP 必须大于等于 0 且小于 CHUNK_SIZE。")

