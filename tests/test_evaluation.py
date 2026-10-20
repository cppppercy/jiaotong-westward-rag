import json

from westward_rag.evaluation import (
    EvaluationQuestion,
    calculate_metrics,
    retrieval_hit,
    run_evaluation,
    save_result_bundle,
    validate_questions,
)
from westward_rag.models import AnswerResult


def make_questions() -> list[EvaluationQuestion]:
    items = []
    for index in range(10):
        items.append(
            EvaluationQuestion(
                f"Q{index + 1:02}",
                "in_document",
                "问题",
                "答案",
                ["note.md"],
                ["依据"],
                required_evidence_groups=[[f"C-{index + 1:02}", f"C-{index + 1:02}-overlap"]],
            )
        )
    for index in range(10, 15):
        items.append(
            EvaluationQuestion(
                f"Q{index + 1:02}",
                "trap",
                "陷阱题",
                "文档中没有相关信息",
                [],
                [],
            )
        )
    for index in range(15, 20):
        items.append(
            EvaluationQuestion(
                f"Q{index + 1:02}",
                "multi_hop",
                "跨片段问题",
                "答案",
                ["one.md", "two.md"],
                ["依据一", "依据二"],
                required_evidence_groups=[[f"C-{index + 1:02}-a"], [f"C-{index + 1:02}-b"]],
            )
        )
    return items


def test_complete_question_set_passes_validation() -> None:
    assert validate_questions(make_questions()) == []


def test_wrong_type_count_is_rejected() -> None:
    questions = make_questions()
    questions[0] = EvaluationQuestion(
        "Q01", "trap", "问题", "文档中没有相关信息", [], [], []
    )
    errors = validate_questions(questions)
    assert any("in_document 应有 10 题" in item for item in errors)
    assert any("trap 应有 5 题" in item for item in errors)


def test_template_placeholders_are_rejected() -> None:
    questions = make_questions()
    questions[0] = EvaluationQuestion(
        "Q01", "in_document", "请本人填写", "请本人填写", ["请填写文档名"], ["请粘贴原文依据"], []
    )
    assert any("模板占位文字" in item for item in validate_questions(questions))


def test_metrics_do_not_invent_unjudged_scores() -> None:
    rows = [
        {
            "retrieval_method": "bm25",
            "question_type": "in_document",
            "human_correct": True,
            "human_hallucination": None,
            "retrieval_hit": True,
            "elapsed_seconds": 1.0,
            "status": "answered",
        },
        {
            "retrieval_method": "bm25",
            "question_type": "trap",
            "human_correct": None,
            "human_hallucination": None,
            "retrieval_hit": None,
            "elapsed_seconds": 3.0,
            "status": "refused",
        },
    ]
    metrics = calculate_metrics(rows)["bm25"]
    assert metrics["scoring_progress"] == "1/20"
    assert metrics["temporary_answer_accuracy"] == 1.0
    assert metrics["official_answer_accuracy"] is None
    assert metrics["official_hallucination_rate"] is None
    assert metrics["temporary_average_latency_seconds"] == 2.0
    assert metrics["official_metrics_ready"] is False


def _result_with_chunks(*chunk_ids: str) -> dict:
    return {
        "retrieval_results": [
            {"chunk": {"chunk_id": chunk_id, "document_name": "history.md"}}
            for chunk_id in chunk_ids
        ]
    }


def test_same_document_unrelated_chunk_is_not_a_hit() -> None:
    question = EvaluationQuestion(
        "Q01",
        "in_document",
        "问题",
        "答案",
        ["history.md"],
        ["正确依据"],
        required_evidence_groups=[["C-answer", "C-answer-overlap"]],
    )
    assert retrieval_hit(question, _result_with_chunks("C-unrelated")) is False


def test_multi_hop_requires_every_evidence_group() -> None:
    question = EvaluationQuestion(
        "Q16",
        "multi_hop",
        "问题",
        "答案",
        ["history.md"],
        ["依据一", "依据二"],
        required_evidence_groups=[["C-a", "C-a-overlap"], ["C-b"]],
    )
    assert retrieval_hit(question, _result_with_chunks("C-a")) is False
    assert retrieval_hit(question, _result_with_chunks("C-a-overlap", "C-b")) is True


def test_official_metrics_require_complete_human_scoring() -> None:
    rows = []
    for index in range(20):
        is_trap = 10 <= index < 15
        rows.append(
            {
                "retrieval_method": "bm25",
                "question_type": "trap" if is_trap else "in_document",
                "human_correct": True,
                "human_hallucination": False if is_trap else None,
                "retrieval_hit": None if is_trap else True,
                "elapsed_seconds": 1.0,
                "status": "refused" if is_trap else "answered",
                "execution_ok": True,
            }
        )
    metrics = calculate_metrics(rows)["bm25"]
    assert metrics["official_metrics_ready"] is True
    assert metrics["official_answer_accuracy"] == 1.0
    assert metrics["official_hallucination_rate"] == 0.0
    assert metrics["official_retrieval_hit_rate"] == 1.0


class SometimesFailingPipeline:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def answer(self, question: str, method: str) -> AnswerResult:
        self.calls.append(question)
        if len(self.calls) == 2:
            raise RuntimeError("temporary failure")
        return AnswerResult(question, method, "refused", "文档中没有相关信息")


def test_evaluation_preserves_rows_and_records_single_item_error() -> None:
    pipeline = SometimesFailingPipeline()
    snapshots: list[int] = []
    rows = run_evaluation(
        make_questions(),
        pipeline,
        methods=("bm25",),
        progress_callback=lambda current, completed, total: snapshots.append(len(current)),
    )
    assert len(rows) == 20
    assert rows[1]["status"] == "engineering_error"
    assert rows[1]["execution_ok"] is False
    assert "temporary failure" in rows[1]["engineering_error"]
    assert snapshots[-1] == 20


def test_resume_skips_completed_items() -> None:
    previous = {
        "question_id": "Q01",
        "retrieval_method": "bm25",
        "status": "refused",
        "execution_ok": True,
    }
    pipeline = SometimesFailingPipeline()
    rows = run_evaluation(
        make_questions(),
        pipeline,
        methods=("bm25",),
        existing_rows=[previous],
    )
    assert len(rows) == 20
    assert len(pipeline.calls) == 19


def test_result_bundle_splits_methods_and_records_config(tmp_path) -> None:
    rows = [
        {"question_id": "Q01", "retrieval_method": "bm25"},
        {"question_id": "Q01", "retrieval_method": "vector"},
    ]
    artifacts = save_result_bundle(rows, tmp_path, {"corpus_digest": "abc"}, run_id="run-test")
    assert all(path.exists() for path in artifacts.values())
    assert len(json.loads(artifacts["bm25"].read_text(encoding="utf-8"))) == 1
    assert len(json.loads(artifacts["vector"].read_text(encoding="utf-8"))) == 1
    config = json.loads(artifacts["config"].read_text(encoding="utf-8"))
    assert config["corpus_digest"] == "abc"
