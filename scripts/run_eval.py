from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from westward_rag.config import Settings  # noqa: E402
from westward_rag.evaluation import (  # noqa: E402
    calculate_metrics,
    load_questions,
    run_evaluation,
    save_results,
    validate_questions,
)
from westward_rag.knowledge_base import KnowledgeBase  # noqa: E402
from westward_rag.pipeline import RAGPipeline  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="运行两套检索方式的 20 题评测。")
    parser.add_argument("--questions", type=Path, default=ROOT / "eval" / "questions.json")
    parser.add_argument("--output", type=Path, default=ROOT / "eval" / "results")
    args = parser.parse_args()

    settings = Settings.from_env(ROOT)
    questions = load_questions(args.questions)
    errors = validate_questions(questions, require_complete=True)
    if errors:
        print("评测集校验失败：", file=sys.stderr)
        print("\n".join(f"- {item}" for item in errors), file=sys.stderr)
        return 2
    knowledge_base = KnowledgeBase.from_directory(settings)
    if not knowledge_base.chunks:
        print("知识库为空。请先向 data/documents 添加资料。", file=sys.stderr)
        return 2
    if not settings.llm_api_key:
        print("LLM_API_KEY 未配置。", file=sys.stderr)
        return 2
    rows = run_evaluation(questions, RAGPipeline(knowledge_base, settings))
    target = save_results(rows, args.output)
    print(f"结果已保存：{target}")
    print(json.dumps(calculate_metrics(rows), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

