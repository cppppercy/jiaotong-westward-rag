from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
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
from westward_rag.loaders import discover_documents  # noqa: E402
from westward_rag.pipeline import RAGPipeline  # noqa: E402


st.set_page_config(page_title="交大西迁知识库", page_icon="📚", layout="wide")


def document_signature(settings: Settings) -> tuple[tuple[str, int, int], ...]:
    return tuple(
        (path.as_posix(), path.stat().st_size, path.stat().st_mtime_ns)
        for path in discover_documents(settings.documents_dir)
    )


@st.cache_resource(show_spinner=False)
def get_knowledge_base(signature: tuple[tuple[str, int, int], ...]) -> KnowledgeBase:
    del signature
    settings = Settings.from_env(ROOT)
    settings.validate()
    return KnowledgeBase.from_directory(settings)


def get_runtime() -> tuple[Settings, KnowledgeBase, RAGPipeline]:
    settings = Settings.from_env(ROOT)
    settings.validate()
    knowledge_base = get_knowledge_base(document_signature(settings))
    return settings, knowledge_base, RAGPipeline(knowledge_base, settings)


def render_search_results(results) -> None:
    if not results:
        st.info("没有检索结果。")
        return
    for item in results:
        title = (
            f"Top {item.rank} · {item.chunk.document_name} · "
            f"{item.chunk.position} · 分数 {item.score:.4f}"
        )
        with st.expander(title, expanded=item.rank == 1):
            st.caption(f"片段编号：{item.chunk.chunk_id}")
            st.write(item.chunk.text)


def qa_page() -> None:
    st.title("交通大学西迁知识库问答")
    st.caption("答案必须由检索证据支持；引用由程序校验，证据不足时拒答。")
    settings, knowledge_base, pipeline = get_runtime()
    if not knowledge_base.chunks:
        st.warning("知识库为空。请先到“资料与索引”页面上传资料。")
        return

    method_label = st.radio(
        "检索方式",
        options=["bm25", "vector"],
        format_func=lambda item: "关键词检索（BM25）" if item == "bm25" else "向量检索（BGE）",
        horizontal=True,
    )
    question = st.text_area("问题", placeholder="例如：交通大学为什么要西迁？", height=100)
    if st.button("提交问题", type="primary", use_container_width=True):
        if method_label == "vector":
            with st.spinner("正在加载向量模型并检索，首次运行可能需要下载模型……"):
                result = pipeline.answer(question, method_label)
        else:
            with st.spinner("正在检索并生成回答……"):
                result = pipeline.answer(question, method_label)

        if result.status == "answered":
            st.success(result.answer)
        elif result.status == "refused":
            st.warning(result.answer)
        else:
            st.error(result.answer)
        if result.message:
            st.caption(result.message)
        st.metric("完整回答耗时", f"{result.elapsed_seconds:.3f} 秒")

        if result.citations:
            st.subheader("已校验引用")
            for citation in result.citations:
                st.markdown(f"**[{citation.chunk_id}] {citation.document_name} · {citation.position}**")
                st.code(citation.text, language=None)
        st.subheader("Top-3 检索证据")
        render_search_results(result.retrieval_results)

    with st.sidebar:
        st.caption(f"片段数：{len(knowledge_base.chunks)}")
        st.caption(f"Top-K：{settings.top_k}")
        st.caption(f"回答模型：{settings.llm_model}")
        st.caption("API：已配置" if settings.llm_api_key else "API：未配置")


def documents_page() -> None:
    st.title("资料与索引")
    settings = Settings.from_env(ROOT)
    settings.documents_dir.mkdir(parents=True, exist_ok=True)
    paths = discover_documents(settings.documents_dir)

    col1, col2, col3 = st.columns(3)
    col1.metric("资料文件", len(paths))
    col2.metric("支持格式", "MD / TXT / PDF")
    try:
        knowledge_base = get_knowledge_base(document_signature(settings))
        col3.metric("切分片段", len(knowledge_base.chunks))
    except Exception as exc:
        knowledge_base = None
        col3.error(f"读取失败：{exc}")

    if paths:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "文件": path.name,
                        "格式": path.suffix.lower(),
                        "大小（KB）": round(path.stat().st_size / 1024, 1),
                    }
                    for path in paths
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )

    uploads = st.file_uploader(
        "上传资料",
        type=["md", "txt", "pdf"],
        accept_multiple_files=True,
        help="文件会保存到本机 data/documents；请确认隐私和使用权限。",
    )
    if st.button("保存上传文件", disabled=not uploads):
        saved = []
        for upload in uploads or []:
            safe_name = Path(upload.name).name
            target = settings.documents_dir / safe_name
            target.write_bytes(upload.getvalue())
            saved.append(safe_name)
        get_knowledge_base.clear()
        st.success(f"已保存：{', '.join(saved)}")
        st.rerun()

    if st.button("重建向量索引", disabled=knowledge_base is None or not knowledge_base.chunks):
        assert knowledge_base is not None
        with st.spinner("正在生成向量并持久化，首次运行需要下载 BGE 模型……"):
            try:
                knowledge_base.build_vector_index(force=True)
            except Exception as exc:
                st.error(f"索引构建失败：{exc}")
            else:
                st.success("向量索引已重建。")

    st.info(
        "课程要求的 1500 字本人原创学习笔记需由你完成。"
        "模板位于 data/templates/个人学习笔记模板.md。"
    )


def _read_uploaded_result(upload) -> list[dict]:
    payload = json.loads(upload.getvalue().decode("utf-8-sig"))
    if not isinstance(payload, list):
        raise ValueError("结果 JSON 顶层必须是数组。")
    return payload


def evaluation_page() -> None:
    st.title("评测")
    settings, knowledge_base, pipeline = get_runtime()
    template_path = settings.eval_dir / "questions.template.json"
    if template_path.exists():
        st.download_button(
            "下载 20 题模板",
            data=template_path.read_bytes(),
            file_name="questions.template.json",
            mime="application/json",
        )

    st.warning(
        "请本人完成问题、标准答案和依据。批量运行会调用回答 API 最多 40 次，可能产生费用。"
    )
    uploaded = st.file_uploader("上传已完成的评测集", type=["json", "csv"], key="questions")
    questions_path = settings.eval_dir / "questions.json"
    if uploaded and st.button("校验并保存评测集"):
        target = settings.eval_dir / ("questions.csv" if uploaded.name.lower().endswith(".csv") else "questions.json")
        target.write_bytes(uploaded.getvalue())
        try:
            questions = load_questions(target)
            errors = validate_questions(questions, require_complete=True)
        except Exception as exc:
            st.error(f"读取失败：{exc}")
        else:
            if errors:
                st.error("\n".join(f"- {item}" for item in errors))
            else:
                if target.suffix == ".json":
                    questions_path = target
                st.success(f"评测集校验通过：{target.name}")

    candidates = [settings.eval_dir / "questions.json", settings.eval_dir / "questions.csv"]
    active = next((path for path in candidates if path.exists()), None)
    run_disabled = active is None or not knowledge_base.chunks or not settings.llm_api_key
    if st.button("运行 BM25 与向量检索评测", type="primary", disabled=run_disabled):
        assert active is not None
        try:
            questions = load_questions(active)
            errors = validate_questions(questions, require_complete=True)
            if errors:
                raise ValueError("\n".join(errors))
            progress = st.progress(0, text="准备评测……")
            # The core runner keeps the two retrieval settings identical except for retrieval method.
            rows = run_evaluation(questions, pipeline)
            progress.progress(100, text="评测运行完成")
            target = save_results(rows, settings.eval_dir / "results")
            st.success(f"原始结果已保存到 {target.relative_to(ROOT)}")
            st.json(calculate_metrics(rows))
        except Exception as exc:
            st.error(f"评测失败：{exc}")
    if run_disabled:
        reasons = []
        if active is None:
            reasons.append("尚无通过校验的 questions.json/csv")
        if not knowledge_base.chunks:
            reasons.append("知识库为空")
        if not settings.llm_api_key:
            reasons.append("API 未配置")
        st.caption("暂不能运行：" + "；".join(reasons))

    st.divider()
    result_upload = st.file_uploader("查看已人工标注的结果 JSON", type=["json"], key="results")
    if result_upload:
        try:
            rows = _read_uploaded_result(result_upload)
            st.json(calculate_metrics(rows))
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
        except Exception as exc:
            st.error(f"结果文件无效：{exc}")


page = st.sidebar.radio("导航", ["知识库问答", "资料与索引", "评测"])
if page == "知识库问答":
    qa_page()
elif page == "资料与索引":
    documents_page()
else:
    evaluation_page()

