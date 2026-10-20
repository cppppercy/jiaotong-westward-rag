from __future__ import annotations

import hashlib
import re

from .models import Chunk, LoadedSection


PARAGRAPH_BREAK = re.compile(r"\n\s*\n+")


def _split_long_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    if len(text) <= chunk_size:
        return [text]
    pieces: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            boundary = max(text.rfind("。", start, end), text.rfind("；", start, end))
            if boundary > start + chunk_size // 2:
                end = boundary + 1
        piece = text[start:end].strip()
        if piece:
            pieces.append(piece)
        if end >= len(text):
            break
        start = max(start + 1, end - overlap)
    return pieces


def chunk_sections(
    sections: list[LoadedSection], chunk_size: int = 600, overlap: int = 80
) -> list[Chunk]:
    if chunk_size < 100 or not 0 <= overlap < chunk_size:
        raise ValueError("切分参数无效。")

    chunks: list[Chunk] = []
    doc_counter: dict[str, int] = {}
    for section in sections:
        paragraphs = [part.strip() for part in PARAGRAPH_BREAK.split(section.text) if part.strip()]
        atomic: list[str] = []
        for paragraph in paragraphs:
            atomic.extend(_split_long_text(paragraph, chunk_size, overlap))

        current = ""
        merged: list[str] = []
        for paragraph in atomic:
            candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
            if current and len(candidate) > chunk_size:
                merged.append(current)
                prefix = current[-overlap:].strip() if overlap else ""
                current = f"{prefix}\n\n{paragraph}".strip() if prefix else paragraph
                if len(current) > chunk_size:
                    merged.extend(_split_long_text(current, chunk_size, overlap)[:-1])
                    current = _split_long_text(current, chunk_size, overlap)[-1]
            else:
                current = candidate
        if current:
            merged.append(current)

        for text in merged:
            index = doc_counter.get(section.source_path, 0) + 1
            doc_counter[section.source_path] = index
            identity = f"{section.source_path}|{section.page}|{index}|{text}".encode("utf-8")
            chunk_id = f"C-{hashlib.sha1(identity).hexdigest()[:10]}"
            position = f"第 {section.page} 页 / 片段 {index}" if section.page else f"片段 {index}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    source_path=section.source_path,
                    document_name=section.document_name,
                    text=text,
                    position=position,
                    source_url=section.source_url,
                )
            )
    return chunks

