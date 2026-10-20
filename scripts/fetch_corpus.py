from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

import requests
from bs4 import BeautifulSoup, Tag


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data" / "documents"
DEFAULT_MANIFEST = ROOT / "data" / "corpus_manifest.json"
DEFAULT_GUIDE = ROOT / "data" / "CORPUS_READING_GUIDE.md"


@dataclass(frozen=True)
class Source:
    source_id: str
    filename: str
    title: str
    publisher: str
    url: str
    topic: str


SOURCES = (
    Source(
        "S14",
        "01_校门沧桑话交大.md",
        "校门沧桑话交大",
        "西安交通大学西迁博物馆",
        "https://jdxqmuseum.xjtu.edu.cn/info/1046/1377.htm",
        "迁校时间、校园建设与校门沿革",
    ),
    Source(
        "S7",
        "03_西迁往事知多少.md",
        "交大西迁口述史：西迁往事知多少",
        "西安交通大学西迁精神专题网（来源：中国人才）",
        "https://xq.xjtu.edu.cn/info/1008/2992.htm",
        "多位亲历者口述与教学延续",
    ),
    Source(
        "S3",
        "04_西迁精神的由来与内涵.md",
        "“西迁精神”的由来与内涵",
        "西安交通大学新闻网（转载人民政协报）",
        "https://news.xjtu.edu.cn/info/1014/114806.htm",
        "精神内涵、形成过程与建校事例",
    ),
    Source(
        "S4",
        "05_交大西迁的故事.md",
        "交大西迁的故事",
        "西安交通大学西迁精神专题网",
        "https://xq.xjtu.edu.cn/info/1008/3192.htm",
        "迁校叙事和关键史料线索",
    ),
    Source(
        "S5",
        "06_西迁精神的光辉.md",
        "西迁精神的光辉",
        "西安交通大学西迁博物馆",
        "https://jdxqmuseum.xjtu.edu.cn/info/1046/1385.htm",
        "西迁影响与精神价值",
    ),
    Source(
        "S15",
        "07_钱学森的母校情缘.md",
        "钱学森的母校情缘",
        "西安交通大学西迁博物馆",
        "https://jdxqmuseum.xjtu.edu.cn/info/1046/1460.htm",
        "钱学森与迁校决定、母校建设",
    ),
    Source(
        "S11",
        "08_定名西安交通大学.txt",
        "定名“西安交通大学”",
        "西安交通大学西迁博物馆",
        "https://jdxqmuseum.xjtu.edu.cn/info/1046/1322.htm",
        "迁校方案调整、组织沿革与统计范围",
    ),
    Source(
        "S12",
        "09_我家的西迁漫记.txt",
        "我家的西迁漫记",
        "西安交通大学西迁博物馆",
        "https://jdxqmuseum.xjtu.edu.cn/info/1046/1896.htm",
        "家属生活、个人经历与社会背景",
    ),
    Source(
        "S13",
        "10_扎根西北办好社会主义的西安交通大学.txt",
        "扎根西北，办好社会主义的西安交通大学",
        "西安交通大学西迁博物馆",
        "https://jdxqmuseum.xjtu.edu.cn/info/1046/2188.htm",
        "西迁后的办学、人才培养与科研建设",
    ),
)


CONTENT_SELECTORS = (
    ".TRS_Editor",
    ".v_news_content",
    "#vsb_content",
    ".article-content",
    ".article_content",
    ".news_content",
    "article",
)
DROP_TEXT = re.compile(
    r"^(上一篇|下一篇|返回列表|关闭窗口|打印本页|责任编辑|分享到|扫一扫|点击量|当前位置)"
)
DATE_PATTERN = re.compile(r"(?:19|20)\d{2}[-年./]\d{1,2}[-月./]\d{1,2}日?")
HAN_PATTERN = re.compile(r"[\u4e00-\u9fff]")


def _get_html(source: Source, timeout: int) -> str:
    response = requests.get(
        source.url,
        timeout=timeout,
        headers={"User-Agent": "jiaotong-westward-rag/0.2 educational corpus fetcher"},
    )
    response.raise_for_status()
    response.encoding = response.apparent_encoding or response.encoding
    return response.text


def _main_content(soup: BeautifulSoup) -> Tag:
    candidates: list[Tag] = []
    for selector in CONTENT_SELECTORS:
        candidates.extend(item for item in soup.select(selector) if isinstance(item, Tag))
    if not candidates:
        candidates = [item for item in soup.find_all(["main", "section", "div"]) if isinstance(item, Tag)]
    if not candidates:
        raise ValueError("没有找到正文容器。")
    return max(candidates, key=lambda item: len(item.get_text(" ", strip=True)))


def _clean_paragraphs(container: Tag, expected_title: str) -> list[str]:
    for element in container.select("script, style, nav, form, iframe, .share, .footer, .prev-next"):
        element.decompose()
    paragraphs: list[str] = []
    seen: set[str] = set()
    for node in container.find_all(["h1", "h2", "h3", "h4", "p", "li"]):
        text = re.sub(r"\s+", " ", node.get_text(" ", strip=True)).strip()
        if not text or text == expected_title or DROP_TEXT.search(text):
            continue
        if len(text) < 8 and not node.name.startswith("h"):
            continue
        fingerprint = re.sub(r"\s+", "", text)
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        prefix = "## " if node.name.startswith("h") else ""
        paragraphs.append(prefix + text)
    if len("".join(paragraphs)) < 500:
        fallback = [
            re.sub(r"\s+", " ", line).strip()
            for line in container.get_text("\n", strip=True).splitlines()
        ]
        paragraphs = [line for line in fallback if len(line) >= 8 and not DROP_TEXT.search(line)]
    return paragraphs


def _published_at(soup: BeautifulSoup) -> str:
    text = soup.get_text(" ", strip=True)[:4000]
    match = DATE_PATTERN.search(text)
    return match.group(0) if match else "未自动识别，请人工核对原页"


def _render_document(source: Source, published_at: str, paragraphs: list[str]) -> str:
    metadata = (
        "---\n"
        f'title: "{source.title}"\n'
        f'publisher: "{source.publisher}"\n'
        f'published_at: "{published_at}"\n'
        f'source_url: "{source.url}"\n'
        f'retrieved_at: "{date.today().isoformat()}"\n'
        "---\n\n"
    )
    return metadata + f"# {source.title}\n\n" + "\n\n".join(paragraphs).strip() + "\n"


def fetch_source(source: Source, output_dir: Path, timeout: int) -> dict:
    html = _get_html(source, timeout)
    soup = BeautifulSoup(html, "html.parser")
    container = _main_content(soup)
    paragraphs = _clean_paragraphs(container, source.title)
    if not paragraphs:
        raise ValueError("清洗后没有正文。")
    document = _render_document(source, _published_at(soup), paragraphs)
    target = output_dir / source.filename
    target.write_text(document, encoding="utf-8")
    body = "\n".join(paragraphs)
    if len(body) < 500 or "版权所有：西安交通大学" in body:
        raise ValueError(f"正文过短或疑似只抓到页脚（{len(body)} 字符）。")
    return {
        **asdict(source),
        "status": "ok",
        "path": target.relative_to(ROOT).as_posix(),
        "characters": len(body),
        "han_characters": len(HAN_PATTERN.findall(body)),
        "paragraphs": len(paragraphs),
        "sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
        "retrieved_at": date.today().isoformat(),
        "_body": body,
    }


def _write_public_files(rows: list[dict], manifest_path: Path, guide_path: Path) -> None:
    duplicate_candidates = []
    successful = [row for row in rows if row.get("status") == "ok"]
    for index, left in enumerate(successful):
        left_chars = re.sub(r"\s+", "", left.get("_body", ""))
        left_shingles = {left_chars[offset : offset + 24] for offset in range(max(0, len(left_chars) - 23))}
        for right in successful[index + 1 :]:
            right_chars = re.sub(r"\s+", "", right.get("_body", ""))
            right_shingles = {
                right_chars[offset : offset + 24] for offset in range(max(0, len(right_chars) - 23))
            }
            union = left_shingles | right_shingles
            similarity = len(left_shingles & right_shingles) / len(union) if union else 0.0
            if similarity >= 0.35:
                duplicate_candidates.append(
                    {
                        "left": left["source_id"],
                        "right": right["source_id"],
                        "shingle_jaccard": round(similarity, 4),
                    }
                )
    public_rows = [
        {key: value for key, value in row.items() if not key.startswith("_")}
        for row in rows
    ]
    summary = {
        "generated_at": date.today().isoformat(),
        "notice": "正文位于被 Git 忽略的 data/documents；本文件仅记录来源和统计。",
        "successful_documents": sum(row.get("status") == "ok" for row in rows),
        "total_characters": sum(row.get("characters", 0) for row in rows),
        "total_han_characters": sum(row.get("han_characters", 0) for row in rows),
        "possible_duplicate_pairs": duplicate_candidates,
        "documents": public_rows,
    }
    manifest_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# 正式语料阅读导引",
        "",
        "> 正文不提交到公共仓库。运行 `python scripts/fetch_corpus.py` 可从原始页面重新获取。",
        "",
        f"成功获取：{summary['successful_documents']} 篇；正文字符：{summary['total_characters']}；汉字：{summary['total_han_characters']}。",
        "",
        "| 编号 | 主题 | 文件 | 正文字符 | 汉字 | 状态 |",
        "|---|---|---|---:|---:|---|",
    ]
    for row in public_rows:
        lines.append(
            f"| {row['source_id']} | {row['topic']} | {row['filename']} | "
            f"{row.get('characters', 0)} | {row.get('han_characters', 0)} | {row['status']} |"
        )
    lines.extend(
        [
            "",
            "## 建议阅读顺序",
            "",
            "1. 先读迁校时间、总体叙事和组织沿革（01、05、08）。",
            "2. 再读口述、人物与家属经历（03、07、09）。",
            "3. 核对西迁后的办学和统计口径（08、10）。",
            "4. 最后阅读精神内涵和当代阐释（04、06）。",
            "",
            "遇到年份、人数或统计范围不一致时，保留各文原有表述并回到 URL 核对，不要自动合并成一个数字。",
        ]
    )
    guide_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="获取并清洗交大西迁官方公开网页正文。")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--guide", type=Path, default=DEFAULT_GUIDE)
    parser.add_argument("--timeout", type=int, default=30)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for source in SOURCES:
        try:
            row = fetch_source(source, args.output, args.timeout)
            print(f"[ok] {source.source_id} {source.title}: {row['characters']} 字符")
        except Exception as exc:
            row = {
                **asdict(source),
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "characters": 0,
                "han_characters": 0,
            }
            print(f"[error] {source.source_id} {source.title}: {row['error']}", file=sys.stderr)
        rows.append(row)
    hashes = [row["sha256"] for row in rows if row.get("sha256")]
    if len(hashes) != len(set(hashes)):
        print("警告：检测到正文完全重复，请人工检查 manifest。", file=sys.stderr)
    _write_public_files(rows, args.manifest, args.guide)
    failures = sum(row["status"] != "ok" for row in rows)
    print(f"完成：成功 {len(rows) - failures}/{len(rows)}，统计：{args.manifest}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
