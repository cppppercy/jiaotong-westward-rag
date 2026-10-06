from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader

from .models import LoadedSection


SUPPORTED_SUFFIXES = {".md", ".txt", ".pdf"}
IGNORED_DOCUMENT_NAMES = {"README.md"}


def discover_documents(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_SUFFIXES
        and path.name not in IGNORED_DOCUMENT_NAMES
        and not path.name.startswith(".")
    )


def load_document(path: Path, root: Path | None = None) -> list[LoadedSection]:
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(f"不支持的文件格式：{suffix}")
    source_path = path.relative_to(root).as_posix() if root and path.is_relative_to(root) else path.name
    if suffix == ".pdf":
        return _load_pdf(path, source_path)
    text = path.read_text(encoding="utf-8-sig").strip()
    return [LoadedSection(source_path, path.name, text)] if text else []


def _load_pdf(path: Path, source_path: str) -> list[LoadedSection]:
    reader = PdfReader(str(path))
    sections: list[LoadedSection] = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            sections.append(
                LoadedSection(
                    source_path=source_path,
                    document_name=path.name,
                    text=text,
                    page=page_number,
                )
            )
    return sections


def load_directory(directory: Path) -> list[LoadedSection]:
    sections: list[LoadedSection] = []
    for path in discover_documents(directory):
        sections.extend(load_document(path, directory))
    return sections
