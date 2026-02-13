from __future__ import annotations

import re
from typing import List, Set, Tuple


_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_PHONE_RE = re.compile(r"\+?\d[\d\s().-]{7,}\d")
_URL_RE = re.compile(r"(https?://\S+|\b(?:linkedin\.com|github\.com)\b)", re.IGNORECASE)
_COMMON_SECTION_HEADINGS = {
    "summary",
    "professional summary",
    "skills",
    "technical skills",
    "experience",
    "work experience",
    "education",
    "projects",
    "certifications",
    "awards",
}


def is_private_block(text: str) -> bool:
    if _EMAIL_RE.search(text):
        return True
    if _PHONE_RE.search(text):
        return True
    if _URL_RE.search(text):
        return True
    return False


def protect_top_blocks(blocks: List[dict], n: int = 5) -> Set[str]:
    protected: Set[str] = set()
    count = 0
    for block in blocks:
        text = block.get("text", "").strip()
        if not text:
            continue
        block_id = block.get("id")
        if block_id:
            protected.add(block_id)
            count += 1
        if count >= n:
            break
    return protected


def looks_like_heading(text: str) -> bool:
    trimmed = text.strip()
    if len(trimmed) > 60:
        return False

    normalized = re.sub(r"\s+", " ", trimmed).strip(" :.-").lower()
    if normalized in _COMMON_SECTION_HEADINGS:
        return True

    letters = [ch for ch in trimmed if ch.isalpha()]
    if letters and all(ch.isupper() for ch in letters) and len(trimmed.split()) <= 5:
        return True

    return trimmed.endswith(":") and len(trimmed.split()) <= 4


def choose_rewrite_candidates(blocks: List[dict]) -> Tuple[List[dict], Set[str]]:
    protected_ids = protect_top_blocks(blocks, n=3)

    for block in blocks:
        text = block.get("text", "")
        block_id = block.get("id")
        if not block_id:
            continue

        if is_private_block(text) or looks_like_heading(text):
            protected_ids.add(block_id)

    rewrite_blocks = [block for block in blocks if block.get("id") not in protected_ids]
    return rewrite_blocks, protected_ids
