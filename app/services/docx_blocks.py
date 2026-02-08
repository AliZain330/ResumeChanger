from __future__ import annotations

from typing import Dict, List

from docx import Document


def _is_bullet(paragraph) -> bool:
    try:
        ppr = paragraph._p.pPr
        return ppr is not None and ppr.numPr is not None
    except Exception:
        return False


def extract_blocks(docx_path: str) -> List[Dict[str, str]]:
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


def apply_block_updates(
    input_docx_path: str,
    output_docx_path: str,
    updates: Dict[str, str],
) -> None:
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

        new_text = updates[block_id]
        if paragraph.runs:
            paragraph.runs[0].text = new_text
            for run in paragraph.runs[1:]:
                run.text = ""
        else:
            paragraph.add_run(new_text)

    document.save(output_docx_path)
