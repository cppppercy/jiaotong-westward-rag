from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


@dataclass(frozen=True)
class LoadedSection:
    source_path: str
    document_name: str
    text: str
    page: int | None = None


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    source_path: str
    document_name: str
    text: str
    position: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Chunk":
        return cls(**value)


@dataclass(frozen=True)
class SearchResult:
    chunk: Chunk
    score: float
    rank: int

    def to_dict(self) -> dict[str, Any]:
        return {"chunk": self.chunk.to_dict(), "score": self.score, "rank": self.rank}


AnswerStatus = Literal["answered", "refused", "configuration_error", "generation_error"]


@dataclass
class AnswerResult:
    question: str
    method: str
    status: AnswerStatus
    answer: str
    citations: list[Chunk] = field(default_factory=list)
    retrieval_results: list[SearchResult] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "method": self.method,
            "status": self.status,
            "answer": self.answer,
            "citations": [item.to_dict() for item in self.citations],
            "retrieval_results": [item.to_dict() for item in self.retrieval_results],
            "elapsed_seconds": self.elapsed_seconds,
            "message": self.message,
        }

