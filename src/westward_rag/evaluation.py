from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from .llm import PROMPT_VERSION, SYSTEM_PROMPT
from .pipeline import RAGPipeline
from .retrieval import corpus_digest


VALID_TYPES = {"in_document", "trap", "multi_hop"}
EXPECTED_COUNTS = {"in_document": 10, "trap": 5, "multi_hop": 5}
ENGINEERING_ERROR_STATUSES = {
    "configuration_error",
    "retrieval_error",
    "generation_error",
    "engineering_error",
}
PLACEHOLDER_VALUES = {
    "请本人填写",
    "请填写文档名",
    "请粘贴原文依据",
    "第一处必要依据",
    "第二处必要依据",
    "请填写等价片段编号",
}


@dataclass(frozen=True)
class EvaluationQuestion:
    id: str
    type: str
    question: str
    reference_answer: str
    source_documents: list[str]
    evidence_quotes: list[str]
    required_chunk_ids: list[str] = field(default_factory=list)
    required_evidence_groups: list[list[str]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "EvaluationQuestion":
        legacy_ids = _as_list(value.get("required_chunk_ids", []))
        groups = _as_evidence_groups(value.get("required_evidence_groups", []))
        if not groups and legacy_ids:
            groups = [[chunk_id] for chunk_id in legacy_ids]
        return cls(
            id=str(value.get("id", "")).strip(),
            type=str(value.get("type", "")).strip(),
            question=str(value.get("question", "")).strip(),
            reference_answer=str(value.get("reference_answer", "")).strip(),
            source_documents=_as_list(value.get("source_documents", [])),
            evidence_quotes=_as_list(value.get("evidence_quotes", [])),
            required_chunk_ids=legacy_ids,
            required_evidence_groups=groups,
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


def _as_evidence_groups(value: Any) -> list[list[str]]:
    if value is None or value == "":
        return []
    if isinstance(value, list):
        groups: list[list[str]] = []
        for group in value:
            if isinstance(group, list):
                cleaned = [str(item).strip() for item in group if str(item).strip()]
            else:
                cleaned = [str(group).strip()] if str(group).strip() else []
            if cleaned:
                groups.append(cleaned)
        return groups
    if isinstance(value, str):
        return [
            [item.strip() for item in group.split("|") if item.strip()]
            for group in value.split(";")
            if group.strip()
        ]
    raise ValueError("required_evidence_groups 必须是二维数组，CSV 中使用 组1候选|候选;组2候选。")


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
            *(chunk_id for group in item.required_evidence_groups for chunk_id in group),
        }
        if supplied_values & PLACEHOLDER_VALUES:
            errors.append(f"{label} 仍含模板占位文字，请本人完成后再运行。")
        if item.type not in VALID_TYPES:
            errors.append(f"{label} 的 type 必须是 in_document、trap 或 multi_hop。")
            continue
        counts[item.type] += 1
        if item.type == "trap":
            if (
                item.source_documents
                or item.evidence_quotes
                or item.required_chunk_ids
                or item.required_evidence_groups
            ):
                errors.append(f"{label} 是陷阱题，来源和证据字段应为空。")
            continue
        if not item.source_documents or not item.evidence_quotes:
            errors.append(f"{label} 必须填写来源文档和依据原文。")
        if not item.required_evidence_groups:
            errors.append(f"{label} 必须填写 required_evidence_groups，不能仅按文档名判断命中。")
        if item.type == "multi_hop":
            if len(item.evidence_quotes) < 2:
                errors.append(f"{label} 是跨片段题，至少需要两处依据原文。")
            if len(item.required_evidence_groups) < 2:
                errors.append(f"{label} 是跨片段题，至少需要两组必要证据片段。")
    if require_complete:
        for key, expected in EXPECTED_COUNTS.items():
            if counts[key] != expected:
                errors.append(f"{key} 应有 {expected} 题，当前为 {counts[key]} 题。")
    return errors


def retrieval_hit(question: EvaluationQuestion, result: dict[str, Any]) -> bool | None:
    if question.type == "trap":
        return None
    retrieved_ids = {
        item["chunk"]["chunk_id"]
        for item in result.get("retrieval_results", [])
        if item.get("chunk", {}).get("chunk_id")
    }
    groups = question.required_evidence_groups
    if not groups:
        return False
    return all(bool(retrieved_ids.intersection(group)) for group in groups)


def _engineering_failure(answer: dict[str, Any]) -> bool:
    return answer.get("status") in ENGINEERING_ERROR_STATUSES


def _error_answer(question: str, method: str, exc: Exception) -> dict[str, Any]:
    return {
        "question": question,
        "method": method,
        "status": "engineering_error",
        "answer": "评测执行发生工程错误。",
        "citations": [],
        "retrieval_results": [],
        "elapsed_seconds": 0.0,
        "message": f"{type(exc).__name__}: {exc}",
    }


ProgressCallback = Callable[[list[dict[str, Any]], int, int], None]


def run_evaluation(
    questions: list[EvaluationQuestion],
    pipeline: RAGPipeline,
    methods: tuple[str, ...] = ("bm25", "vector"),
    existing_rows: list[dict[str, Any]] | None = None,
    progress_callback: ProgressCallback | None = None,
    retry_errors: bool = False,
) -> list[dict[str, Any]]:
    errors = validate_questions(questions, require_complete=True)
    if errors:
        raise ValueError("\n".join(errors))

    rows = list(existing_rows or [])
    total = len(methods) * len(questions)
    completed = {(row.get("retrieval_method"), row.get("question_id")): row for row in rows}

    for method in methods:
        for question in questions:
            key = (method, question.id)
            previous = completed.get(key)
            if previous and (not retry_errors or not _engineering_failure(previous)):
                continue
            if previous:
                rows.remove(previous)
            try:
                answer = pipeline.answer(question.question, method).to_dict()
            except Exception as exc:
                answer = _error_answer(question.question, method, exc)
            failed = _engineering_failure(answer)
            row = {
                "question_id": question.id,
                "question_type": question.type,
                "question": question.question,
                "reference_answer": question.reference_answer,
                "expected_source_documents": question.source_documents,
                "expected_evidence_quotes": question.evidence_quotes,
                "required_chunk_ids": question.required_chunk_ids,
                "required_evidence_groups": question.required_evidence_groups,
                "retrieval_method": method,
                **answer,
                "execution_ok": not failed,
                "engineering_error": (
                    answer.get("message") or answer.get("answer") or answer.get("status")
                    if failed
                    else None
                ),
                "retrieval_hit": retrieval_hit(question, answer),
                "human_correct": None,
                "human_hallucination": None,
                "human_notes": "",
            }
            rows.append(row)
            completed[key] = row
            if progress_callback:
                progress_callback(rows, len(completed), total)
    return rows


def _atomic_json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def save_checkpoint(rows: list[dict[str, Any]], target: Path) -> Path:
    _atomic_json_write(target, rows)
    return target


def load_result_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, list):
        raise ValueError("评测结果 JSON 顶层必须是数组。")
    return payload


def _git_commit(project_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def build_experiment_config(settings: Any, chunks: list[Any]) -> dict[str, Any]:
    return {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "code_commit": _git_commit(settings.project_root),
        "corpus_digest": corpus_digest(chunks),
        "document_count": len({chunk.source_path for chunk in chunks}),
        "chunk_count": len(chunks),
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "top_k": settings.top_k,
        "bm25_min_score": settings.bm25_min_score,
        "vector_min_similarity": settings.vector_min_similarity,
        "embedding_model": settings.embedding_model,
        "llm_base_url": settings.llm_base_url,
        "llm_model": settings.llm_model,
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
    }


def _write_scoring_csv(rows: list[dict[str, Any]], target: Path) -> None:
    fields = [
        "question_id",
        "question_type",
        "retrieval_method",
        "question",
        "reference_answer",
        "answer",
        "status",
        "retrieval_hit",
        "human_correct",
        "human_hallucination",
        "human_notes",
        "elapsed_seconds",
        "engineering_error",
    ]
    with target.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def save_result_bundle(
    rows: list[dict[str, Any]],
    output_dir: Path,
    experiment_config: dict[str, Any],
    run_id: str | None = None,
) -> dict[str, Path]:
    identifier = run_id or datetime.now().strftime("run-%Y%m%d-%H%M%S")
    run_dir = output_dir / identifier
    run_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "combined": run_dir / "combined_raw.json",
        "bm25": run_dir / "bm25_raw.json",
        "vector": run_dir / "vector_raw.json",
        "scoring": run_dir / "human_scoring.csv",
        "metrics": run_dir / "metrics_summary.json",
        "config": run_dir / "experiment_config.json",
    }
    _atomic_json_write(artifacts["combined"], rows)
    _atomic_json_write(
        artifacts["bm25"], [row for row in rows if row.get("retrieval_method") == "bm25"]
    )
    _atomic_json_write(
        artifacts["vector"], [row for row in rows if row.get("retrieval_method") == "vector"]
    )
    _write_scoring_csv(rows, artifacts["scoring"])
    _atomic_json_write(artifacts["metrics"], calculate_metrics(rows))
    _atomic_json_write(artifacts["config"], experiment_config)
    return artifacts


def save_results(rows: list[dict[str, Any]], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    target = output_dir / f"evaluation-{stamp}.json"
    _atomic_json_write(target, rows)
    return target


def _row_execution_ok(row: dict[str, Any]) -> bool:
    if "execution_ok" in row:
        return bool(row["execution_ok"])
    return row.get("status") not in ENGINEERING_ERROR_STATUSES


def calculate_metrics(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    metrics: dict[str, dict[str, Any]] = {}
    for method in ("bm25", "vector"):
        subset = [row for row in rows if row.get("retrieval_method") == method]
        judged = [row for row in subset if isinstance(row.get("human_correct"), bool)]
        trap_rows = [row for row in subset if row.get("question_type") == "trap"]
        judged_traps = [
            row for row in trap_rows if isinstance(row.get("human_hallucination"), bool)
        ]
        answerable = [row for row in subset if row.get("question_type") != "trap"]
        correct = sum(bool(row["human_correct"]) for row in judged)
        hallucinations = sum(bool(row["human_hallucination"]) for row in judged_traps)
        retrieval_hits = sum(bool(row.get("retrieval_hit")) for row in answerable)
        engineering_errors = sum(not _row_execution_ok(row) for row in subset)
        run_complete = len(subset) == 20 and engineering_errors == 0
        scoring_complete = run_complete and len(judged) == 20 and len(judged_traps) == 5
        retrieval_complete = run_complete and len(answerable) == 15
        temporary_accuracy = correct / len(judged) if judged else None
        temporary_hallucination = hallucinations / len(judged_traps) if judged_traps else None
        temporary_retrieval = retrieval_hits / len(answerable) if answerable else None
        temporary_latency = (
            sum(float(row.get("elapsed_seconds", 0)) for row in subset) / len(subset)
            if subset
            else None
        )
        metrics[method] = {
            "result_progress": f"{len(subset)}/20",
            "scoring_progress": f"{len(judged)}/20",
            "trap_scoring_progress": f"{len(judged_traps)}/5",
            "engineering_error_count": engineering_errors,
            "answer_correct_count": correct,
            "answer_judged_count": len(judged),
            "temporary_answer_accuracy": temporary_accuracy,
            "official_answer_accuracy": temporary_accuracy if scoring_complete else None,
            "hallucination_count": hallucinations,
            "hallucination_judged_count": len(judged_traps),
            "temporary_hallucination_rate": temporary_hallucination,
            "official_hallucination_rate": temporary_hallucination if scoring_complete else None,
            "retrieval_hit_count": retrieval_hits,
            "retrieval_answerable_count": len(answerable),
            "temporary_retrieval_hit_rate": temporary_retrieval,
            "official_retrieval_hit_rate": temporary_retrieval if retrieval_complete else None,
            "temporary_average_latency_seconds": temporary_latency,
            "official_average_latency_seconds": temporary_latency if run_complete else None,
            "official_metrics_ready": scoring_complete and retrieval_complete,
        }
    return metrics
