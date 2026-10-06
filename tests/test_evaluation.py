from westward_rag.evaluation import EvaluationQuestion, calculate_metrics, validate_questions


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
                [],
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
                [],
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
        },
        {
            "retrieval_method": "bm25",
            "question_type": "trap",
            "human_correct": None,
            "human_hallucination": None,
            "retrieval_hit": None,
            "elapsed_seconds": 3.0,
        },
    ]
    metrics = calculate_metrics(rows)["bm25"]
    assert metrics["answer_accuracy"] == 1.0
    assert metrics["hallucination_rate"] is None
    assert metrics["average_latency_seconds"] == 2.0
