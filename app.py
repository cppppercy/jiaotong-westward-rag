from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from westward_rag.config import Settings  # noqa: E402
from westward_rag.evaluation import (  # noqa: E402
    calculate_metrics,
    build_experiment_config,
    load_questions,
    run_evaluation,
    save_checkpoint,
    save_result_bundle,
    validate_questions,
)
from westward_rag.knowledge_base import KnowledgeBase  # noqa: E402
from westward_rag.loaders import discover_documents, load_document  # noqa: E402
from westward_rag.pipeline import RAGPipeline  # noqa: E402


st.set_page_config(page_title="交大西迁知识库", page_icon="📚", layout="wide")


def document_signature(settings: Settings) -> tuple[tuple[str, int, int], ...]:
    files = tuple(
        (path.as_posix(), path.stat().st_size, path.stat().st_mtime_ns)
        for path in discover_documents(settings.documents_dir)
    )
    configuration = (
        f"settings:{settings.chunk_size}:{settings.chunk_overlap}:{settings.embedding_model}",
        0,
        0,
    )
    return (configuration, *files)


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
            if item.chunk.source_url:
                st.link_button("打开原始网页", item.chunk.source_url)
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

        st.session_state.setdefault("qa_history", []).append(
            {"question": question, "method": method_label, "result": result}
        )

    history = st.session_state.setdefault("qa_history", [])
    if not history:
        st.info("输入问题后，回答和证据会保留在本次会话中；切换检索方式不会清空历史。")
    for record in reversed(history):
        result = record["result"]
        method_name = "BM25" if record["method"] == "bm25" else "BGE 向量"
        with st.chat_message("user"):
            st.write(record["question"])
            st.caption(f"检索方式：{method_name}")
        with st.chat_message("assistant"):
            if result.status == "answered":
                st.success(result.answer)
            elif result.status == "refused":
                st.warning(result.answer)
            else:
                st.error(f"工程错误：{result.answer}")
            if result.message:
                st.caption(result.message)
            st.caption(f"完整回答耗时：{result.elapsed_seconds:.3f} 秒")
            if result.citations:
                with st.expander("已校验引用", expanded=True):
                    for citation in result.citations:
                        st.markdown(
                            f"**[{citation.chunk_id}] {citation.document_name} · {citation.position}**"
                        )
                        if citation.source_url:
                            st.markdown(f"[打开原始网页]({citation.source_url})")
                        st.code(citation.text, language=None)
            with st.expander("Top-3 检索证据"):
                render_search_results(result.retrieval_results)

    with st.sidebar:
        st.caption(f"片段数：{len(knowledge_base.chunks)}")
        st.caption(f"Top-K：{settings.top_k}")
        st.caption(f"回答模型：{settings.llm_model}")
        st.caption("API：已配置" if settings.llm_api_key else "API：未配置")
        if st.button("清空本次问答历史"):
            st.session_state["qa_history"] = []
            st.rerun()


def documents_page() -> None:
    st.title("资料与索引")
    settings = Settings.from_env(ROOT)
    settings.documents_dir.mkdir(parents=True, exist_ok=True)
    paths = discover_documents(settings.documents_dir)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("资料文件", len(paths))
    col2.metric("支持格式", "MD / TXT / PDF")
    try:
        knowledge_base = get_knowledge_base(document_signature(settings))
        col3.metric("切分片段", len(knowledge_base.chunks))
        col4.metric("向量索引", "最新" if knowledge_base.vector.cache_is_valid else "需要重建")
    except Exception as exc:
        knowledge_base = None
        col3.error(f"读取失败：{exc}")
        col4.metric("向量索引", "不可用")

    document_rows = []
    if paths:
        chunk_counts: dict[str, int] = {}
        if knowledge_base is not None:
            for chunk in knowledge_base.chunks:
                chunk_counts[chunk.source_path] = chunk_counts.get(chunk.source_path, 0) + 1
        for path in paths:
            relative = path.relative_to(settings.documents_dir).as_posix()
            try:
                sections = load_document(path, settings.documents_dir)
                characters = sum(len(section.text) for section in sections)
                source_url = next((section.source_url for section in sections if section.source_url), "")
                status = "正常" if sections else "无正文"
            except Exception as exc:
                characters = 0
                source_url = ""
                status = f"解析失败：{exc}"
            document_rows.append(
                {
                    "文件": relative,
                    "格式": path.suffix.lower(),
                    "正文字符": characters,
                    "片段": chunk_counts.get(relative, 0),
                    "大小（KB）": round(path.stat().st_size / 1024, 1),
                    "状态": status,
                    "原始网页": source_url,
                }
            )
        st.dataframe(
            pd.DataFrame(document_rows),
            use_container_width=True,
            hide_index=True,
        )
        if knowledge_base is not None:
            catalog = [chunk.to_dict() for chunk in knowledge_base.chunks]
            st.download_button(
                "下载片段清单（用于填写必要证据组）",
                data=json.dumps(catalog, ensure_ascii=False, indent=2).encode("utf-8"),
                file_name="chunk_catalog.json",
                mime="application/json",
            )

        selected_name = st.selectbox("预览资料", [row["文件"] for row in document_rows])
        selected_path = settings.documents_dir / selected_name
        try:
            selected_sections = load_document(selected_path, settings.documents_dir)
        except Exception as exc:
            st.error(f"无法预览：{exc}")
        else:
            preview_text = "\n\n".join(section.text for section in selected_sections)
            selected_url = next(
                (section.source_url for section in selected_sections if section.source_url), None
            )
            if selected_url:
                st.link_button("打开该资料的原始网页", selected_url)
            st.text_area("正文预览（前 3000 字符）", preview_text[:3000], height=240, disabled=True)

        with st.expander("删除本地资料"):
            st.warning("删除后不可从应用内恢复；通过采集脚本生成的资料可重新获取。")
            confirm_delete = st.checkbox(f"确认删除 {selected_name}")
            if st.button("删除所选资料", disabled=not confirm_delete):
                resolved = selected_path.resolve()
                root = settings.documents_dir.resolve()
                if not resolved.is_relative_to(root) or resolved == root:
                    st.error("删除目标不在资料目录内，操作已取消。")
                else:
                    resolved.unlink()
                    get_knowledge_base.clear()
                    st.success(f"已删除：{selected_name}")
                    st.rerun()

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
        "本项目允许使用网上公开的西迁主题资料。请保留标题、发布机构、日期和 URL，"
        "清除导航页脚并核对事实；不要把评测题或标准答案放入知识库。"
    )


def _read_uploaded_result(upload) -> list[dict]:
    payload = json.loads(upload.getvalue().decode("utf-8-sig"))
    if not isinstance(payload, list):
        raise ValueError("结果 JSON 顶层必须是数组。")
    return payload


def _rate_text(numerator: int, denominator: int, rate: float | None) -> str:
    return "未生成" if rate is None else f"{numerator}/{denominator} = {rate:.1%}"


def render_metrics(rows: list[dict]) -> None:
    metrics = calculate_metrics(rows)
    table = []
    for method, values in metrics.items():
        table.append(
            {
                "方法": "BM25" if method == "bm25" else "BGE 向量",
                "运行进度": values["result_progress"],
                "评分进度": values["scoring_progress"],
                "陷阱题评分": values["trap_scoring_progress"],
                "工程错误": values["engineering_error_count"],
                "临时正确率": _rate_text(
                    values["answer_correct_count"],
                    values["answer_judged_count"],
                    values["temporary_answer_accuracy"],
                ),
                "正式正确率": _rate_text(
                    values["answer_correct_count"],
                    20,
                    values["official_answer_accuracy"],
                ),
                "正式幻觉率": _rate_text(
                    values["hallucination_count"],
                    5,
                    values["official_hallucination_rate"],
                ),
                "Top-3 正式命中率": _rate_text(
                    values["retrieval_hit_count"],
                    15,
                    values["official_retrieval_hit_rate"],
                ),
            }
        )
    st.dataframe(pd.DataFrame(table), use_container_width=True, hide_index=True)
    if not all(value["official_metrics_ready"] for value in metrics.values()):
        st.info("正式指标尚未生成：每种方法必须完成 20 条无工程错误的结果，并完成 20/20 人工正确性和 5/5 陷阱题评分。")


def manual_scoring_panel(settings: Settings) -> None:
    st.subheader("人工逐题评分")
    result_dir = settings.eval_dir / "results"
    local_files = sorted(result_dir.glob("**/combined_raw.json"), reverse=True)
    local_files += sorted(result_dir.glob("**/annotated.json"), reverse=True)
    labels = [path.relative_to(ROOT).as_posix() for path in local_files]
    selected_label = st.selectbox("选择本机结果", ["不选择"] + labels)
    uploaded = st.file_uploader("或上传结果 JSON", type=["json"], key="scoring_result")

    token = None
    rows = None
    source_path = None
    if uploaded is not None:
        token = f"upload:{uploaded.name}:{len(uploaded.getvalue())}"
        rows = _read_uploaded_result(uploaded)
    elif selected_label != "不选择":
        source_path = ROOT / selected_label
        token = f"local:{source_path}:{source_path.stat().st_mtime_ns}"
        rows = json.loads(source_path.read_text(encoding="utf-8-sig"))

    if rows is None or token is None:
        st.caption("运行评测或上传结果后，可在这里逐题查看证据并评分，无需手改 JSON。")
        return
    if st.session_state.get("scoring_source_token") != token:
        st.session_state["scoring_source_token"] = token
        st.session_state["scoring_rows"] = rows
        if source_path is not None:
            st.session_state["scoring_target"] = source_path.parent / "annotated.json"
        else:
            stamp = datetime.now().strftime("manual-%Y%m%d-%H%M%S")
            st.session_state["scoring_target"] = result_dir / stamp / "annotated.json"

    scoring_rows = st.session_state["scoring_rows"]
    method = st.radio("评分方法", ["bm25", "vector"], horizontal=True)
    candidates = [row for row in scoring_rows if row.get("retrieval_method") == method]
    if not candidates:
        st.warning("该结果中没有所选检索方法的数据。")
        return
    selected_id = st.selectbox(
        "选择题目",
        [row.get("question_id", "") for row in candidates],
        format_func=lambda value: next(
            (
                f"{item.get('question_id')} · {item.get('question_type')} · {item.get('question')}"
                for item in candidates
                if item.get("question_id") == value
            ),
            value,
        ),
    )
    row = next(item for item in candidates if item.get("question_id") == selected_id)
    st.markdown(f"**问题：** {row.get('question', '')}")
    st.markdown(f"**参考答案：** {row.get('reference_answer', '')}")
    status = row.get("status", "")
    if row.get("execution_ok", status in {"answered", "refused"}):
        st.markdown(f"**系统答案：** {row.get('answer', '')}")
    else:
        st.error(f"工程错误，修复后应重跑，不能算作正确拒答：{row.get('engineering_error') or row.get('message')}")
    st.caption(f"状态：{status}；检索命中：{row.get('retrieval_hit')}；耗时：{float(row.get('elapsed_seconds', 0)):.3f} 秒")
    with st.expander("Top-3 与依据", expanded=True):
        st.markdown("**人工填写的必要依据：**")
        for quote in row.get("expected_evidence_quotes", []):
            st.code(quote, language=None)
        st.markdown("**实际 Top-3：**")
        for item in row.get("retrieval_results", []):
            chunk = item.get("chunk", {})
            st.markdown(
                f"**{item.get('rank')} · {chunk.get('chunk_id')} · {chunk.get('document_name')} · {chunk.get('position')} · {float(item.get('score', 0)):.4f}**"
            )
            if chunk.get("source_url"):
                st.markdown(f"[原始网页]({chunk['source_url']})")
            st.code(chunk.get("text", ""), language=None)

    correct_options = {"未评分": None, "正确": True, "错误": False}
    reverse_correct = {None: "未评分", True: "正确", False: "错误"}
    correct_label = st.radio(
        "答案正确性",
        list(correct_options),
        index=list(correct_options).index(reverse_correct.get(row.get("human_correct"), "未评分")),
        horizontal=True,
        key=f"correct-{token}-{method}-{selected_id}",
    )
    hallucination_label = "未评分"
    if row.get("question_type") == "trap":
        hallucination_options = {"未评分": None, "没有编造": False, "发生编造": True}
        reverse_hallucination = {None: "未评分", False: "没有编造", True: "发生编造"}
        hallucination_label = st.radio(
            "陷阱题是否产生幻觉",
            list(hallucination_options),
            index=list(hallucination_options).index(
                reverse_hallucination.get(row.get("human_hallucination"), "未评分")
            ),
            horizontal=True,
            key=f"hallucination-{token}-{method}-{selected_id}",
        )
    notes = st.text_area(
        "评分说明",
        value=row.get("human_notes", ""),
        key=f"notes-{token}-{method}-{selected_id}",
    )
    if st.button("保存本题评分", type="primary"):
        row["human_correct"] = correct_options[correct_label]
        if row.get("question_type") == "trap":
            row["human_hallucination"] = {
                "未评分": None,
                "没有编造": False,
                "发生编造": True,
            }[hallucination_label]
        row["human_notes"] = notes.strip()
        target = st.session_state["scoring_target"]
        save_checkpoint(scoring_rows, target)
        st.success(f"已保存到 {target.relative_to(ROOT)}")

    render_metrics(scoring_rows)
    st.download_button(
        "下载当前人工标注 JSON",
        data=json.dumps(scoring_rows, ensure_ascii=False, indent=2).encode("utf-8"),
        file_name="annotated.json",
        mime="application/json",
    )


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
            progress = st.progress(0.0, text="准备评测……")
            run_id = datetime.now().strftime("run-%Y%m%d-%H%M%S")
            checkpoint = settings.eval_dir / "results" / run_id / "checkpoint.json"

            def persist(rows, completed, total):
                save_checkpoint(rows, checkpoint)
                progress.progress(completed / total, text=f"已完成 {completed}/{total}，结果已写入检查点")

            rows = run_evaluation(questions, pipeline, progress_callback=persist)
            progress.progress(100, text="评测运行完成")
            artifacts = save_result_bundle(
                rows,
                settings.eval_dir / "results",
                build_experiment_config(settings, knowledge_base.chunks),
                run_id=run_id,
            )
            st.success(f"评测文件已保存到 {artifacts['combined'].parent.relative_to(ROOT)}")
            st.caption("中断后可用 scripts/run_eval.py --run-id 上述编号 --resume 续跑；工程错误可加 --retry-errors。")
            render_metrics(rows)
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
    try:
        manual_scoring_panel(settings)
    except Exception as exc:
        st.error(f"评分结果读取失败：{exc}")


page = st.sidebar.radio("导航", ["知识库问答", "资料与索引", "评测"])
if page == "知识库问答":
    qa_page()
elif page == "资料与索引":
    documents_page()
else:
    evaluation_page()

