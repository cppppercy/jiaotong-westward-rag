from __future__ import annotations

import time

from .config import Settings
from .knowledge_base import KnowledgeBase
from .llm import AnswerGenerator, OpenAICompatibleGenerator, REFUSAL_TEXT
from .models import AnswerResult


class RAGPipeline:
    def __init__(
        self,
        knowledge_base: KnowledgeBase,
        settings: Settings,
        generator: AnswerGenerator | None = None,
    ):
        self.knowledge_base = knowledge_base
        self.settings = settings
        self._generator = generator

    def _get_generator(self) -> AnswerGenerator | None:
        if self._generator is not None:
            return self._generator
        if not self.settings.llm_api_key:
            return None
        self._generator = OpenAICompatibleGenerator(
            api_key=self.settings.llm_api_key,
            base_url=self.settings.llm_base_url,
            model=self.settings.llm_model,
        )
        return self._generator

    def answer(self, question: str, method: str) -> AnswerResult:
        started = time.perf_counter()
        question = question.strip()
        if not question:
            return AnswerResult(
                question=question,
                method=method,
                status="refused",
                answer=REFUSAL_TEXT,
                elapsed_seconds=time.perf_counter() - started,
                message="问题不能为空。",
            )

        try:
            results = self.knowledge_base.search(question, method, self.settings.top_k)
        except Exception as exc:
            return AnswerResult(
                question=question,
                method=method,
                status="retrieval_error",
                answer="检索过程发生错误。",
                elapsed_seconds=time.perf_counter() - started,
                message=str(exc),
            )
        threshold = (
            self.settings.bm25_min_score
            if method == "bm25"
            else self.settings.vector_min_similarity
        )
        if not results or results[0].score < threshold:
            return AnswerResult(
                question=question,
                method=method,
                status="refused",
                answer=REFUSAL_TEXT,
                retrieval_results=results,
                elapsed_seconds=time.perf_counter() - started,
                message=f"最高检索分数未达到 {method} 阈值 {threshold:.2f}。",
            )

        generator = self._get_generator()
        if generator is None:
            return AnswerResult(
                question=question,
                method=method,
                status="configuration_error",
                answer="尚未配置回答模型。请复制 .env.example 为 .env 并填写 API 配置。",
                retrieval_results=results,
                elapsed_seconds=time.perf_counter() - started,
            )

        try:
            generated = generator.generate(question, results)
        except Exception as exc:  # API clients expose many provider-specific exceptions.
            return AnswerResult(
                question=question,
                method=method,
                status="generation_error",
                answer="回答模型调用失败。",
                retrieval_results=results,
                elapsed_seconds=time.perf_counter() - started,
                message=str(exc),
            )

        if generated.refused:
            return AnswerResult(
                question=question,
                method=method,
                status="refused",
                answer=REFUSAL_TEXT,
                retrieval_results=results,
                elapsed_seconds=time.perf_counter() - started,
                message="模型判断当前证据不足。",
            )

        allowed = {result.chunk.chunk_id: result.chunk for result in results}
        citation_ids = list(dict.fromkeys(generated.citation_ids))
        citations = [allowed[item] for item in citation_ids if item in allowed]
        invalid = [item for item in citation_ids if item not in allowed]
        if not generated.answer or not citations or invalid:
            return AnswerResult(
                question=question,
                method=method,
                status="refused",
                answer=REFUSAL_TEXT,
                retrieval_results=results,
                elapsed_seconds=time.perf_counter() - started,
                message="模型没有提供可验证的有效引用，系统已安全拒答。",
            )

        return AnswerResult(
            question=question,
            method=method,
            status="answered",
            answer=generated.answer,
            citations=citations,
            retrieval_results=results,
            elapsed_seconds=time.perf_counter() - started,
        )

