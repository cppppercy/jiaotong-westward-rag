from pathlib import Path

from westward_rag.config import Settings
from westward_rag.llm import GeneratedAnswer, REFUSAL_TEXT
from westward_rag.models import Chunk, SearchResult
from westward_rag.pipeline import RAGPipeline


CHUNK = Chunk("C-valid", "note.md", "note.md", "交通大学迁往西安。", "片段 1")


class StubKnowledgeBase:
    def __init__(self, score: float = 3.0):
        self.score = score

    def search(self, query: str, method: str, top_k: int):
        return [SearchResult(CHUNK, self.score, 1)]


class FailingKnowledgeBase:
    def search(self, query: str, method: str, top_k: int):
        raise RuntimeError("index unavailable")


class StubGenerator:
    def __init__(self, result: GeneratedAnswer):
        self.result = result

    def generate(self, question, evidence):
        return self.result


def settings(tmp_path: Path, **values) -> Settings:
    return Settings(
        project_root=tmp_path,
        documents_dir=tmp_path / "documents",
        index_dir=tmp_path / "index",
        eval_dir=tmp_path / "eval",
        **values,
    )


def test_low_score_refuses_before_generation(tmp_path: Path) -> None:
    pipeline = RAGPipeline(
        StubKnowledgeBase(score=0.2),
        settings(tmp_path, bm25_min_score=1.5),
        StubGenerator(GeneratedAnswer("不应调用", ["C-valid"], False)),
    )
    result = pipeline.answer("问题", "bm25")
    assert result.status == "refused"
    assert result.answer == REFUSAL_TEXT


def test_missing_api_is_configuration_error(tmp_path: Path) -> None:
    result = RAGPipeline(StubKnowledgeBase(), settings(tmp_path)).answer("问题", "bm25")
    assert result.status == "configuration_error"


def test_valid_citation_is_rendered(tmp_path: Path) -> None:
    generator = StubGenerator(GeneratedAnswer("因为国家建设需要。", ["C-valid"], False))
    result = RAGPipeline(StubKnowledgeBase(), settings(tmp_path), generator).answer("为什么", "bm25")
    assert result.status == "answered"
    assert result.citations == [CHUNK]


def test_invalid_citation_fails_closed(tmp_path: Path) -> None:
    generator = StubGenerator(GeneratedAnswer("未经支持的答案", ["C-fake"], False))
    result = RAGPipeline(StubKnowledgeBase(), settings(tmp_path), generator).answer("为什么", "bm25")
    assert result.status == "refused"
    assert result.citations == []


def test_retrieval_exception_is_recorded_as_engineering_error(tmp_path: Path) -> None:
    result = RAGPipeline(FailingKnowledgeBase(), settings(tmp_path)).answer("为什么", "bm25")
    assert result.status == "retrieval_error"
    assert result.answer == "检索过程发生错误。"
    assert "index unavailable" in (result.message or "")

