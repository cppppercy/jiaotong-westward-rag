# 交通大学西迁 RAG 问答助手

一个面向课程实验和作品集展示的中文知识库问答系统。系统先从西迁主题资料中检索证据，再调用 OpenAI 兼容模型回答；证据不足时明确回复“文档中没有相关信息”。

当前实现与实机验证进度见 [`docs/PROJECT_STATUS.md`](docs/PROJECT_STATUS.md)。
本人需要完成的题目、评分、报告和录屏步骤见 [`docs/USER_TASKS.md`](docs/USER_TASKS.md)。

## 功能

- 导入 Markdown、TXT 和文本型 PDF。
- 对同一切分结果切换 Jieba + BM25 与 `BAAI/bge-small-zh-v1.5` 向量检索。
- 展示 Top-3、检索分数、片段编号、来源和位置。
- 以独立阈值进行检索预拒答，再由模型判断证据是否充分。
- 模型只能返回片段编号，引用由程序白名单校验并展示原文。
- 批量运行 20 题 × 2 种检索方式，保留逐题原始结果与耗时。
- 仅对人工填写过的字段计算答案正确率和幻觉率，不自动伪造评分。

## 项目结构

```text
app.py                         Streamlit 三页界面
src/westward_rag/              文档、检索、回答与评测核心
data/documents/                本机正式资料（默认不提交）
data/templates/                资料核对记录模板
data/sources.md                官方候选来源清单
data/corpus_manifest.json      本地语料来源与字数统计（不含正文）
data/CORPUS_READING_GUIDE.md   正式语料阅读导引
data/index/                    本地向量索引（自动生成）
eval/questions.template.json   20 题空白模板
eval/results/                  原始评测结果（自动生成）
docs/                          报告、视频和 AI 错误记录模板
tests/                         自动化测试
```

## 环境

- Python 3.10—3.13
- Windows、macOS 或 Linux
- 一个支持 OpenAI Chat Completions 接口的 API
- 首次向量检索需联网下载 BGE 模型；模型下载后可以从本地缓存加载

## 安装与启动

Windows 可直接运行：

```powershell
.\start.ps1
```

也可以手动执行：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
Copy-Item .env.example .env
# 编辑 .env，填写 API 配置
.\.venv\Scripts\python.exe -m streamlit run app.py
```

macOS/Linux 将虚拟环境解释器路径替换为 `.venv/bin/python`。

## 配置

将 `.env.example` 复制为 `.env`：

```dotenv
LLM_API_KEY=你的密钥
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4o-mini
EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5
TOP_K=3
BM25_MIN_SCORE=1.5
VECTOR_MIN_SIMILARITY=0.45
```

`.env` 已被 Git 忽略。不同 API 服务商可填写各自兼容地址和模型名。

## 使用流程

1. 运行 `python scripts/fetch_corpus.py` 获取并清洗官方公开资料，或自行将有权使用的 `.md`、`.txt`、文本型 `.pdf` 放入 `data/documents/`。
2. 阅读 `data/CORPUS_READING_GUIDE.md` 和原始网页，核对年份、人数、统计范围与亲历者表述；不要求另写原创学习笔记。
3. 在“资料与索引”页查看解析结果并重建向量索引。
4. 在“知识库问答”页切换 BM25/向量检索，检查回答、引用和 Top-3。
5. 复制 `eval/questions.template.json` 为 `eval/questions.json`，由本人完成 20 题、标准答案、原文依据与必要证据组。
6. 在“评测”页校验并运行，随后对结果填写 `human_correct`、`human_hallucination` 和 `human_notes`。
7. 重新载入人工标注后的结果查看指标，按真实数据完成报告。

命令行评测：

```powershell
.\.venv\Scripts\python.exe scripts\run_eval.py --questions eval\questions.json
```

运行中断后使用同一个运行编号续跑；工程错误修复后可只重跑错误项：

```powershell
.\.venv\Scripts\python.exe scripts\run_eval.py --questions eval\questions.json --run-id run-YYYYMMDD-HHMMSS --resume --retry-errors
```

每次完整运行会分别输出 `bm25_raw.json`、`vector_raw.json`、合并结果、人工评分 CSV、指标摘要和实验配置。实验配置记录语料摘要、切分参数、阈值、模型、提示词版本、时间和代码提交号。

## 评测口径

- `in_document`：10 题，文档中可直接回答。
- `trap`：5 题，知识库没有答案，预期拒答。
- `multi_hop`：5 题，至少需要两处证据。
- 答案正确率：仅统计已填写布尔值 `human_correct` 的题目。
- 幻觉率：仅统计陷阱题中已填写 `human_hallucination` 的题目。
- 检索命中率：严格按 `required_evidence_groups` 判断。组间是“且”，必须全部覆盖；同一组中的片段编号是等价证据，命中任意一个即可。文档名相同但片段无关不算命中。
- 首次回答平均延迟：从提交问题到完整结果返回，包含检索和模型调用时间。

正式实验时，两种检索必须使用相同资料、切分参数、Top-K、回答模型、提示词和人工评分标准。

## 安全与局限

- RAG 和阈值只能降低幻觉风险，不能保证答案必然正确。
- BM25 与向量分数含义不同，因此分别设置阈值。默认值是首版工程起点，应使用独立调试题校准，并在报告中记录最终值。
- PDF 仅支持包含文本层的文件，不包含 OCR。
- 程序只校验模型引用了真实检索片段，最终答案是否被证据充分支持仍需人工核查。
- 不要将 API 密钥、个人隐私、无权公开的资料或正式评测答案意外提交到公共仓库。

## 测试

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest -q
```

课程最终提交还需要本人完成实验报告、真实 AI 错误记录和 3—5 分钟真机演示。仓库中的模板仅提供结构，不代表这些工作已经完成。

