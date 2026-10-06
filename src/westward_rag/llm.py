from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Protocol

from openai import OpenAI

from .models import SearchResult


REFUSAL_TEXT = "文档中没有相关信息"


@dataclass(frozen=True)
class GeneratedAnswer:
    answer: str
    citation_ids: list[str]
    refused: bool


class AnswerGenerator(Protocol):
    def generate(self, question: str, evidence: list[SearchResult]) -> GeneratedAnswer: ...


SYSTEM_PROMPT = f"""你是一个严格依据资料回答的知识库助手。
规则：
1. 只能使用用户消息中提供的证据，不得使用外部知识、常识或猜测补全。
2. 若证据不能完整支持答案，必须拒答，answer 精确写为“{REFUSAL_TEXT}”，refused 写 true，citations 写空数组。
3. 若能够回答，只引用真正支持结论的片段编号，不得创造编号、文档名或原文。
4. 输出合法 JSON，不要使用 Markdown 代码块，也不要输出 JSON 之外的文字。
JSON 格式：{{"answer":"简洁中文回答","citations":["C-..."],"refused":false}}
"""


def _extract_json(text: str) -> dict:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise ValueError("模型没有返回可解析的 JSON。")
        value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("模型返回的 JSON 必须是对象。")
    return value


class OpenAICompatibleGenerator:
    def __init__(self, api_key: str, base_url: str, model: str):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def generate(self, question: str, evidence: list[SearchResult]) -> GeneratedAnswer:
        blocks = []
        for result in evidence:
            chunk = result.chunk
            blocks.append(
                f"[{chunk.chunk_id}] 文档：{chunk.document_name}；位置：{chunk.position}\n{chunk.text}"
            )
        user_prompt = f"问题：{question}\n\n证据：\n" + "\n\n".join(blocks)
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        content = response.choices[0].message.content or ""
        payload = _extract_json(content)
        answer = str(payload.get("answer", "")).strip()
        citation_ids = payload.get("citations", [])
        if not isinstance(citation_ids, list) or not all(
            isinstance(item, str) for item in citation_ids
        ):
            raise ValueError("模型返回的 citations 字段格式错误。")
        refused = bool(payload.get("refused", False)) or answer == REFUSAL_TEXT
        return GeneratedAnswer(answer=answer, citation_ids=citation_ids, refused=refused)

