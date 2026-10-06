from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .pipeline import RAGPipeline


VALID_TYPES = {"in_document", "trap", "multi_hop"}
EXPECTED_COUNTS = {"in_document": 10, "trap": 5, "multi_hop": 5}
PLACEHOLDER_VALUES = {
    "请本人填写",
    "请填写文档名",
    "请粘贴原文依据",
    "第一处必要依据",
    "第二处必要依据",
}


@dataclass(frozen=True)
class EvaluationQuestion:
    id: str
    type: str
    question: str
    reference_answer: str
    source_documents: list[str]
    evidence_quotes: list[str]
    required_chunk_ids: list[str]

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "EvaluationQuestion":
        return cls(
            id=str(value.get("id", "")).strip(),
            type=str(value.get("type", "")).strip(),
            question=str(value.get("question", "")).strip(),
            reference_answer=str(value.get("reference_answer", "")).strip(),
            source_documents=_as_list(value.get("source_documents", [])),
            evidence_quotes=_as_list(value.get("evidence_quotes", [])),
            required_chunk_ids=_as_list(value.get("required_chunk_ids", [])),
        )


def _as_list(value: Any) -> list[str]:
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        separator = "|" if "|" in value else ";"
        return [item.strip() for item in value.split(separator) if item.strip()]
    raise ValueError("列表字段必须是数组，CSV 中可用 | 分隔。")


def load_questions(path: Path) -> list[EvaluationQuestion]:
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(payload, list):
            raise ValueError("评测 JSON 顶层必须是数组。")
        return [EvaluationQuestion.from_dict(item) for item in payload]
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return [EvaluationQuestion.from_dict(item) for item in csv.DictReader(handle)]
    raise ValueError("评测集只支持 JSON 或 CSV。")


def validate_questions(questions: list[EvaluationQuestion], require_complete: bool = True) -> list[str]:
    errors: list[str] = []
    if require_complete and len(questions) != 20:
        errors.append(f"评测集必须恰好包含 20 题，当前为 {len(questions)} 题。")
    identifiers = [item.id for item in questions]
    if len(identifiers) != len(set(identifiers)):
        errors.append("题号 id 不能重复。")
    counts = {key: 0 for key in VALID_TYPES}
    for index, item in enumerate(questions, start=1):
        label = item.id or f"第 {index} 题"
        if not item.id or not item.question or not item.reference_answer:
            errors.append(f"{label} 缺少 id、question 或 reference_answer。")
        supplied_values = {
            item.question,
            item.reference_answer,
            *item.source_documents,
            *item.evidence_quotes,
        }
        if supplied_values & PLACEHOLDER_VALUES:
            errors.append(f"{label} 仍含模板占位文字，请本人完成后再运行。")
        if item.type not in VALID_TYPES:
            errors.append(f"{label} 的 type 必须是 in_document、trap 或 multi_hop。")
            continue
        counts[item.type] += 1
        if item.type == "trap":
            if item.source_documents or item.evidence_quotes or item.required_chunk_ids:
                errors.append(f"{label} 是陷阱题，来源和证据字段应为空。")
        elif not item.source_documents or not item.evidence_quotes:
            errors.append(f"{label} 必须填写来源文档和依据原文。")
        if item.type == "multi_hop" and len(item.evidence_quotes) < 2:
            errors.append(f"{label} 是跨片段题，至少需要两处依据原文。")
    if require_complete:
        for key, expected in EXPECTED_COUNTS.items():
            if counts[key] != expected:
                errors.append(f"{key} 应有 {expected} 题，当前为 {counts[key]} 题。")
    return errors


def _retrieval_hit(question: EvaluationQuestion, result: dict[str, Any]) -> bool | None:
    if question.type == "trap":
        return None
    retrieved = result["retrieval_results"]
    retrieved_ids = {item["chunk"]["chunk_id"] for item in retrieved}
    retrieved_docs = {item["chunk"]["document_name"] for item in retrieved}
    if question.required_chunk_ids:
        return set(question.required_chunk_ids).issubset(retrieved_ids)
    expected_docs = set(question.source_documents)
    if question.type == "multi_hop":
        return expected_docs.issubset(retrieved_docs)
    return bool(expected_docs & retrieved_docs)


def run_evaluation(
    questions: list[EvaluationQuestion],
    pipeline: RAGPipeline,
    methods: tuple[str, ...] = ("bm25", "vector"),
) -> list[dict[str, Any]]:
    errors = validate_questions(questions, require_complete=True)
    if errors:
        raise ValueError("\n".join(errors))
    rows: list[dict[str, Any]] = []
    for method in methods:
        for question in questions:
            answer = pipeline.answer(question.question, method).to_dict()
            rows.append(
                {
                    "question_id": question.id,
                    "question_type": question.type,
                    "question": question.question,
                    "reference_answer": question.reference_answer,
                    "expected_source_documents": question.source_documents,
                    "expected_evidence_quotes": question.evidence_quotes,
                    "required_chunk_ids": question.required_chunk_ids,
                    "retrieval_method": method,
                    **answer,
                    "retrieval_hit": _retrieval_hit(question, answer),
                    "human_correct": None,
                    "human_hallucination": None,
                    "human_notes": "",
                }
            )
    return rows


def save_results(rows: list[dict[str, Any]], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    target = output_dir / f"evaluation-{stamp}.json"
    target.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return target


def calculate_metrics(rows: list[dict[str, Any]]) -> dict[str, dict[str, float | int | None]]:
    metrics: dict[str, dict[str, float | int | None]] = {}
    for method in ("bm25", "vector"):
        subset = [row for row in rows if row.get("retrieval_method") == method]
        judged = [row for row in subset if isinstance(row.get("human_correct"), bool)]
        traps = [
            row
            for row in subset
            if row.get("question_type") == "trap"
            and isinstance(row.get("human_hallucination"), bool)
        ]
        answerable = [
            row for row in subset if row.get("question_type") != "trap" and row.get("retrieval_hit") is not None
        ]
        metrics[method] = {
            "total_rows": len(subset),
            "judged_rows": len(judged),
            "answer_accuracy": (
                sum(row["human_correct"] for row in judged) / len(judged) if judged else None
            ),
            "hallucination_rate": (
                sum(row["human_hallucination"] for row in traps) / len(traps) if traps else None
            ),
            "retrieval_hit_rate": (
                sum(bool(row["retrieval_hit"]) for row in answerable) / len(answerable)
                if answerable
                else None
            ),
            "average_latency_seconds": (
                sum(float(row.get("elapsed_seconds", 0)) for row in subset) / len(subset)
                if subset
                else None
            ),
        }
    return metrics
