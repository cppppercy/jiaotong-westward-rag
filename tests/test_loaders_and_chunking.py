from pathlib import Path

from reportlab.pdfgen import canvas

from westward_rag.chunking import chunk_sections
from westward_rag.loaders import discover_documents, load_directory


def test_loads_markdown_text_and_pdf(tmp_path: Path) -> None:
    (tmp_path / "note.md").write_text("# 标题\n\n西迁学习笔记。", encoding="utf-8")
    (tmp_path / "facts.txt").write_text("第一批师生乘专列前往西安。", encoding="utf-8")
    pdf_path = tmp_path / "source.pdf"
    pdf = canvas.Canvas(str(pdf_path))
    pdf.drawString(72, 720, "PDF source content")
    pdf.save()

    paths = discover_documents(tmp_path)
    assert [path.suffix for path in paths] == [".txt", ".md", ".pdf"]
    sections = load_directory(tmp_path)
    assert {section.document_name for section in sections} == {
        "note.md",
        "facts.txt",
        "source.pdf",
    }
    assert next(section for section in sections if section.document_name == "source.pdf").page == 1


def test_chunk_ids_are_stable_and_metadata_is_kept(tmp_path: Path) -> None:
    content = (
        "---\nsource_url: https://example.com/source\n---\n"
        + "第一段。" * 40
        + "\n\n"
        + "第二段。" * 40
    )
    (tmp_path / "note.md").write_text(content, encoding="utf-8")
    sections = load_directory(tmp_path)

    first = chunk_sections(sections, chunk_size=120, overlap=20)
    second = chunk_sections(sections, chunk_size=120, overlap=20)

    assert len(first) > 2
    assert [item.chunk_id for item in first] == [item.chunk_id for item in second]
    assert all(item.document_name == "note.md" for item in first)
    assert all(item.position.startswith("片段") for item in first)
    assert all(len(item.text) <= 140 for item in first)
    assert all(item.source_url == "https://example.com/source" for item in first)

