from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Set, Tuple

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RewriteCandidateBlock:
    id: str
    kind: str
    text: str
    original_word_count: int


def _get_status_code(exc: Exception) -> int | None:
    for attr in ("status_code", "code", "status", "http_status"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
    response = getattr(exc, "response", None)
    if response is not None:
        status_code = getattr(response, "status_code", None)
        if isinstance(status_code, int):
            return status_code
    return None


def _should_retry(status_code: int | None) -> bool:
    if status_code is None:
        return False
    return status_code in {429, 500, 502, 503, 504}


def _safe_response_summary(response: object) -> dict:
    candidates = getattr(response, "candidates", None)
    return {
        "response_type": type(response).__name__,
        "candidates_count": len(candidates) if isinstance(candidates, list) else None,
    }


def _extract_text_fallback(response: object) -> str:
    candidates = getattr(response, "candidates", None)
    if not candidates:
        return ""
    first = candidates[0]
    content = getattr(first, "content", None)
    parts = getattr(content, "parts", None) if content is not None else None
    if not parts:
        return ""
    texts = [getattr(part, "text", "") for part in parts if getattr(part, "text", "")]
    return "".join(texts).strip()


def _call_llm_with_retries(
    prompt: str,
    api_key: str,
    model: str,
    prompt_length: int,
    blocks_count: int,
) -> Tuple[str, Dict[str, object]]:
    client = genai.Client(api_key=api_key)
    backoffs = [0.5, 1.0, 2.0]

    for attempt in range(len(backoffs)):
        try:
            logger.info(
                "Gemini request start. model=%s blocks=%s prompt_length=%s",
                model,
                blocks_count,
                prompt_length,
            )
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(response_mime_type="application/json"),
            )
            text = response.text or ""
            candidates = getattr(response, "candidates", None)
            has_candidates = bool(candidates)
            finish_reason = None
            if isinstance(candidates, list) and candidates:
                finish_reason = getattr(candidates[0], "finish_reason", None)
            logger.info(
                "Gemini response received. model=%s prompt_length=%s response_text_len=%s empty=%s has_candidates=%s finish_reason=%s",
                model,
                prompt_length,
                len(text),
                not text.strip(),
                has_candidates,
                finish_reason,
            )
            if not text.strip():
                fallback_text = _extract_text_fallback(response)
                logger.warning(
                    "Gemini empty response text; fallback used. model=%s prompt_length=%s fallback_text_len=%s summary=%s",
                    model,
                    prompt_length,
                    len(fallback_text),
                    _safe_response_summary(response),
                )
                if not fallback_text.strip():
                    raise RuntimeError("Empty response text")
                return fallback_text, {
                    "response_text_len": len(fallback_text),
                    "has_candidates": has_candidates,
                    "finish_reason": finish_reason,
                }
            return text, {
                "response_text_len": len(text),
                "has_candidates": has_candidates,
                "finish_reason": finish_reason,
            }
        except Exception as exc:
            status_code = _get_status_code(exc)
            logger.error(
                "Gemini request failed. error_type=%s message=%s model=%s prompt_length=%s status_code=%s",
                type(exc).__name__,
                str(exc),
                model,
                prompt_length,
                status_code,
            )
            if _should_retry(status_code) and attempt < len(backoffs) - 1:
                time.sleep(backoffs[attempt])
                continue
            raise


def _word_count(text: str) -> int:
    return len([word for word in text.strip().split() if word])


def _validate_and_filter_updates(
    updates: Dict[str, str],
    blocks_by_id: Dict[str, RewriteCandidateBlock],
) -> Tuple[Dict[str, object], int]:
    filtered: Dict[str, str] = {}
    skipped: List[str] = []
    dropped_wordcount = 0

    for block_id, new_text in updates.items():
        if block_id not in blocks_by_id or not isinstance(new_text, str):
            skipped.append(block_id)
            continue

        block = blocks_by_id[block_id]
        original_count = int(block.original_word_count)
        new_count = _word_count(new_text)

        if block.kind == "bullet":
            if not (original_count - 3 <= new_count <= original_count + 3):
                skipped.append(block_id)
                dropped_wordcount += 1
                logger.info(
                    "Wordcount drop. id=%s kind=%s old=%s new=%s",
                    block_id,
                    block.kind,
                    original_count,
                    new_count,
                )
                continue
        else:
            if not (original_count - 10 <= new_count <= original_count + 10):
                skipped.append(block_id)
                dropped_wordcount += 1
                logger.info(
                    "Wordcount drop. id=%s kind=%s old=%s new=%s",
                    block_id,
                    block.kind,
                    original_count,
                    new_count,
                )
                continue

        if new_text.strip() == block.text.strip():
            skipped.append(block_id)
            continue

        filtered[block_id] = new_text

    return {"updates": filtered, "skipped": sorted(set(skipped))}, dropped_wordcount


class GeminiRewriter:
    def __init__(self, api_key: str, model: str) -> None:
        self._api_key = api_key
        self._model = model

    def _build_prompt(self, job_description: str, blocks: Iterable[RewriteCandidateBlock]) -> str:
        blocks_payload = [
            {
                "id": block.id,
                "kind": block.kind,
                "text": block.text,
                "original_word_count": block.original_word_count,
            }
            for block in blocks
        ]

        return (
            "You are rewriting resume blocks to better fit the job description.\n"
            "Return STRICT JSON only in this exact shape:\n"
            '{"updates": {"b0007": "new text"}, "skipped": ["b0019"]}\n'
            "Rules:\n"
            "- Rewrite ONLY the ids provided.\n"
            "- Protected blocks are not included and must not be modified.\n"
            "- Do NOT add new personal info or change contact info.\n"
            "- Do NOT invent facts, metrics, awards, dates, companies, or skills.\n"
            "- Do NOT mention the job description explicitly.\n"
            "- Keep bullet length within +/- 3 words of original_word_count.\n"
            "- Keep paragraph length within +/- 10 words of original_word_count.\n"
            "- If a block cannot be improved, put its id in skipped.\n"
            "- Include only actually changed blocks in updates; unchanged go to skipped.\n\n"
            f"Job description:\n{job_description}\n\n"
            f"Blocks:\n{json.dumps(blocks_payload, ensure_ascii=True)}\n"
        )

    def rewrite_blocks(
        self,
        job_description: str,
        blocks: List[RewriteCandidateBlock],
    ) -> Dict[str, object]:
        if not blocks:
            return {"updates": {}, "skipped": []}

        blocks_by_id = {block.id: block for block in blocks}
        allowed_ids: Set[str] = set(blocks_by_id.keys())
        prompt = self._build_prompt(job_description, blocks)
        prompt_length = len(prompt)

        for attempt in range(2):
            response, response_meta = _call_llm_with_retries(
                prompt,
                api_key=self._api_key,
                model=self._model,
                prompt_length=prompt_length,
                blocks_count=len(blocks),
            )
            try:
                payload = json.loads(response)
            except json.JSONDecodeError:
                payload = None

            if isinstance(payload, dict) and isinstance(payload.get("updates"), dict):
                raw_updates = payload.get("updates", {})
                raw_skipped = payload.get("skipped", [])
                updates_raw_count = len(raw_updates)
                skipped_raw_count = len(raw_skipped) if isinstance(raw_skipped, list) else 0
                filtered_updates = {
                    block_id: text
                    for block_id, text in raw_updates.items()
                    if block_id in allowed_ids
                }
                dropped_unknown_id = updates_raw_count - len(filtered_updates)
                result, dropped_wordcount = _validate_and_filter_updates(filtered_updates, blocks_by_id)

                logger.info(
                    "Gemini rewrite stats: candidates_sent=%s parsed_json_ok=%s updates_raw_count=%s "
                    "skipped_raw_count=%s updates_dropped_due_to_unknown_id=%s updates_dropped_due_to_wordcount=%s "
                    "final_updates_count=%s final_skipped_count=%s",
                    len(blocks),
                    True,
                    updates_raw_count,
                    skipped_raw_count,
                    dropped_unknown_id,
                    dropped_wordcount,
                    len(result["updates"]),
                    len(result["skipped"]),
                )

                if result["updates"] or result["skipped"]:
                    return result

            logger.info(
                "Gemini rewrite stats: candidates_sent=%s parsed_json_ok=%s updates_raw_count=%s "
                "skipped_raw_count=%s updates_dropped_due_to_unknown_id=%s updates_dropped_due_to_wordcount=%s "
                "final_updates_count=%s final_skipped_count=%s",
                len(blocks),
                False,
                0,
                0,
                0,
                0,
                0,
                len(allowed_ids),
            )

            prompt = self._build_prompt(job_description, blocks) + "\nReturn ONLY valid JSON. No extra text."

        return {"updates": {}, "skipped": list(allowed_ids)}

    def smoke_test(self) -> Dict[str, object]:
        prompt = 'Return valid JSON only: {"ok": true}'
        prompt_length = len(prompt)
        try:
            text, response_meta = _call_llm_with_retries(
                prompt,
                api_key=self._api_key,
                model=self._model,
                prompt_length=prompt_length,
                blocks_count=0,
            )
            return {
                "success": True,
                "model": self._model,
                "response_text_len": response_meta.get("response_text_len", len(text)),
            }
        except Exception as exc:
            status_code = _get_status_code(exc)
            return {
                "success": False,
                "model": self._model,
                "response_text_len": 0,
                "error": f"{type(exc).__name__}: {str(exc)}",
                "status_code": status_code,
            }
