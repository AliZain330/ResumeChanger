from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List

from docx import Document


def _is_bullet(paragraph) -> bool:
    try:
        ppr = paragraph._p.pPr
        return ppr is not None and ppr.numPr is not None
    except Exception:
        return False


def _extract_docx_blocks(docx_path: str) -> List[Dict[str, str]]:
    document = Document(docx_path)
    blocks: List[Dict[str, str]] = []
    counter = 1

    for paragraph in document.paragraphs:
        text = paragraph.text.strip()
        if not text:
            continue

        block_id = f"b{counter:04d}"
        counter += 1

        kind = "bullet" if _is_bullet(paragraph) else "paragraph"
        blocks.append({"id": block_id, "kind": kind, "text": text})

    return blocks


def _normalize_line(text: str) -> str:
    return " ".join(text.strip().split())


def _extract_pdf_blocks(pdf_path: str) -> List[Dict[str, str]]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("PDF support requires pypdf. Install with: pip install pypdf") from exc

    reader = PdfReader(pdf_path)
    blocks: List[Dict[str, str]] = []
    counter = 1

    bullet_prefix = re.compile(r"^(?:[-*•]\s+|\d+[\.\)]\s+)")
    for page in reader.pages:
        page_text = page.extract_text() or ""
        for raw_line in page_text.splitlines():
            line = _normalize_line(raw_line)
            if not line:
                continue
            kind = "bullet" if bullet_prefix.match(line) else "paragraph"
            clean_text = bullet_prefix.sub("", line).strip() if kind == "bullet" else line
            if not clean_text:
                continue
            block_id = f"b{counter:04d}"
            counter += 1
            blocks.append({"id": block_id, "kind": kind, "text": clean_text})

    return blocks


def extract_blocks(input_path: str) -> List[Dict[str, str]]:
    suffix = Path(input_path).suffix.lower()
    if suffix == ".docx":
        return _extract_docx_blocks(input_path)
    if suffix == ".pdf":
        return _extract_pdf_blocks(input_path)
    raise ValueError(f"Unsupported file type: {suffix}")


def _replace_paragraph_text_preserving_runs(paragraph, new_text: str) -> None:
    if not paragraph.runs:
        paragraph.add_run(new_text)
        return
    if len(paragraph.runs) == 1:
        paragraph.runs[0].text = new_text
        return

    run_lengths = [max(len(run.text), 1) for run in paragraph.runs]
    total = sum(run_lengths)
    if total <= 0:
        paragraph.runs[0].text = new_text
        for run in paragraph.runs[1:]:
            run.text = ""
        return

    new_len = len(new_text)
    boundaries = [0]
    consumed = 0
    for length in run_lengths[:-1]:
        consumed += length
        boundaries.append(round((consumed / total) * new_len))
    boundaries.append(new_len)

    for idx, run in enumerate(paragraph.runs):
        start = boundaries[idx]
        end = boundaries[idx + 1]
        run.text = new_text[start:end]


def _apply_docx_updates(input_docx_path: str, output_docx_path: str, updates: Dict[str, str]) -> None:
    document = Document(input_docx_path)
    counter = 1

    for paragraph in document.paragraphs:
        text = paragraph.text.strip()
        if not text:
            continue

        block_id = f"b{counter:04d}"
        counter += 1

        if block_id not in updates:
            continue

        _replace_paragraph_text_preserving_runs(paragraph, updates[block_id])

    document.save(output_docx_path)


def _apply_pdf_updates_to_docx(input_pdf_path: str, output_docx_path: str, updates: Dict[str, str]) -> None:
    blocks = _extract_pdf_blocks(input_pdf_path)
    document = Document()

    for block in blocks:
        text = updates.get(block["id"], block["text"])
        if block["kind"] == "bullet":
            document.add_paragraph(text, style="List Bullet")
        else:
            document.add_paragraph(text)

    document.save(output_docx_path)


def apply_block_updates(
    input_docx_path: str,
    output_docx_path: str,
    updates: Dict[str, str],
) -> None:
    suffix = Path(input_docx_path).suffix.lower()
    if suffix == ".docx":
        _apply_docx_updates(input_docx_path, output_docx_path, updates)
        return
    if suffix == ".pdf":
        _apply_pdf_updates_to_docx(input_docx_path, output_docx_path, updates)
        return
    raise ValueError(f"Unsupported file type: {suffix}")
