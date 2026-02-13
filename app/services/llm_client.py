from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Set, Tuple

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

_CONNECTOR_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "by",
    "for",
    "from",
    "in",
    "into",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
    "that",
    "this",
    "these",
    "those",
    "is",
    "be",
    "been",
    "being",
    "was",
    "were",
    "it",
    "its",
    "their",
    "my",
    "our",
    "your",
}


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


def _should_fallback_model(exc: Exception) -> bool:
    status_code = _get_status_code(exc)
    if status_code in {400, 404}:
        return True
    message = str(exc).lower()
    return "model" in message and ("not found" in message or "unsupported" in message)


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


def _normalize_rewrite_text(text: str) -> str:
    return " ".join(text.replace("\r", " ").replace("\n", " ").split())


def _sentence_count(text: str) -> int:
    return len(re.findall(r"[.!?]+", text))


def _tokenize_words(text: str) -> List[str]:
    return re.findall(r"[a-z0-9+#./-]+", text.lower())


def _token_root(token: str) -> str:
    for suffix in ("ing", "ed", "ly", "es", "s"):
        if len(token) > 5 and token.endswith(suffix):
            return token[: -len(suffix)]
    return token


def _extract_numeric_markers(text: str) -> Set[str]:
    return set(re.findall(r"\b\d[\d,.\-%/]*\b", text))


def _extract_target_role(job_description: str) -> str:
    patterns = [
        r"(?im)\b(?:role|position|job title)\s*[:\-]\s*([^\n,.;]{3,80})",
        r"(?im)\b(?:hiring|seeking|looking for)\s+(?:an?\s+)?([A-Za-z][A-Za-z0-9/&\-\s]{2,80}?)(?:\s+(?:role|position))\b",
        r"(?im)\b(?:as|for)\s+(?:an?\s+)?([A-Za-z][A-Za-z0-9/&\-\s]{2,80}?)(?:\s+at|\s+with|\s*,|\s*\.)",
    ]
    for pattern in patterns:
        match = re.search(pattern, job_description)
        if match:
            role = " ".join(match.group(1).split()).strip(" :.-")
            if 2 <= len(role.split()) <= 8:
                return role
    return ""


def _likely_summary_block_ids(blocks: List[RewriteCandidateBlock]) -> Set[str]:
    summary_ids: Set[str] = set()
    for block in blocks:
        text = block.text.lower()
        if any(k in text for k in ("summary", "profile", "objective")):
            summary_ids.add(block.id)

    # Fallback heuristic: first longer paragraph is usually the summary.
    if not summary_ids:
        for block in blocks:
            if block.kind == "paragraph" and block.original_word_count >= 20:
                summary_ids.add(block.id)
                break
    return summary_ids


def _role_phrase(text: str, target_role: str) -> str:
    if not target_role:
        return text
    if target_role.lower() in text.lower():
        return text
    if text.endswith("."):
        return f"{text[:-1]}, targeting {target_role} roles."
    return f"{text}, targeting {target_role} roles."


def _coerce_str_list(value: object, min_len: int = 0, max_len: int = 120) -> List[str]:
    if not isinstance(value, list):
        return []
    out: List[str] = []
    for item in value:
        if isinstance(item, str):
            clean = _normalize_rewrite_text(item)
            if min_len <= len(clean) <= max_len:
                out.append(clean)
    return out


def _coerce_evidence_map(value: object) -> Dict[str, List[str]]:
    if not isinstance(value, dict):
        return {}
    mapped: Dict[str, List[str]] = {}
    for key, val in value.items():
        if not isinstance(key, str):
            continue
        lines = _coerce_str_list(val if isinstance(val, list) else [], min_len=1, max_len=180)
        if lines:
            mapped[_normalize_rewrite_text(key)] = lines
    return mapped


def _validate_and_filter_updates(
    updates: Dict[str, str],
    blocks_by_id: Dict[str, RewriteCandidateBlock],
    job_description: str,
    summary_block_ids: Set[str],
    target_role: str,
) -> Tuple[Dict[str, object], int]:
    filtered: Dict[str, str] = {}
    skipped: List[str] = []
    dropped_wordcount = 0

    for block_id, new_text in updates.items():
        if block_id not in blocks_by_id or not isinstance(new_text, str):
            skipped.append(block_id)
            continue

        clean_text = _normalize_rewrite_text(new_text)
        block = blocks_by_id[block_id]
        if block_id in summary_block_ids and target_role:
            clean_text = _role_phrase(clean_text, target_role)
        original_count = int(block.original_word_count)
        new_count = _word_count(clean_text)

        if block.kind == "bullet":
            if not (original_count - 8 <= new_count <= original_count + 8):
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
            # Guard against turning bullet-style content into paragraph-style prose.
            if _sentence_count(clean_text) > max(1, _sentence_count(block.text) + 1):
                skipped.append(block_id)
                continue
        else:
            if not (original_count - 20 <= new_count <= original_count + 20):
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

        source_tokens = set(_tokenize_words(block.text)) | set(_tokenize_words(job_description))
        source_roots = {_token_root(token) for token in source_tokens}
        update_tokens = _tokenize_words(clean_text)
        unknown_content_tokens = [
            token
            for token in update_tokens
            if len(token) > 3
            and token not in _CONNECTOR_WORDS
            and _token_root(token) not in source_roots
        ]
        if len(unknown_content_tokens) > max(3, int(len(update_tokens) * 0.2)):
            skipped.append(block_id)
            continue

        source_numbers = _extract_numeric_markers(block.text) | _extract_numeric_markers(job_description)
        update_numbers = _extract_numeric_markers(clean_text)
        if any(number not in source_numbers for number in update_numbers):
            skipped.append(block_id)
            continue

        if clean_text.strip() == block.text.strip():
            skipped.append(block_id)
            continue

        filtered[block_id] = clean_text

    return {"updates": filtered, "skipped": sorted(set(skipped))}, dropped_wordcount


class GeminiRewriter:
    def __init__(self, api_key: str, model: str) -> None:
        self._api_key = api_key
        self._model = model

    def _model_candidates(self) -> List[str]:
        fallback = ["gemini-2.5-flash", "gemini-2.0-flash"]
        candidates: List[str] = []
        for model_name in [self._model, *fallback]:
            if model_name and model_name not in candidates:
                candidates.append(model_name)
        return candidates

    def _generate_with_fallback(self, prompt: str, prompt_length: int, blocks_count: int) -> Tuple[str, Dict[str, object]]:
        last_exc: Exception | None = None
        for model_name in self._model_candidates():
            try:
                text, response_meta = _call_llm_with_retries(
                    prompt,
                    api_key=self._api_key,
                    model=model_name,
                    prompt_length=prompt_length,
                    blocks_count=blocks_count,
                )
                response_meta["model_used"] = model_name
                return text, response_meta
            except Exception as exc:
                last_exc = exc
                if not _should_fallback_model(exc):
                    raise
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("No Gemini models available for request")

    def _build_prompt(self, job_description: str, blocks: Iterable[RewriteCandidateBlock]) -> str:
        return self._build_prompt_with_instructions(job_description, blocks, "")

    def _build_prompt_with_instructions(
        self,
        job_description: str,
        blocks: Iterable[RewriteCandidateBlock],
        special_instructions: str,
    ) -> str:
        blocks_payload = [
            {
                "id": block.id,
                "kind": block.kind,
                "text": block.text,
                "original_word_count": block.original_word_count,
            }
            for block in blocks
        ]

        instructions_text = _normalize_rewrite_text(special_instructions)

        return (
            "You are rewriting resume blocks to better fit the job description.\n"
            "Return STRICT JSON only in this exact shape:\n"
            '{'
            '"updates": {"b0007": "new text"}, '
            '"skipped": ["b0019"], '
            '"must_match_signals": ["..."], '
            '"nice_to_have_signals": ["..."], '
            '"culture_cues": ["..."], '
            '"evidence_map": {"signal": ["exact CV line 1", "exact CV line 2"]}, '
            '"positioning_angle": "...", '
            '"gap_list": ["..."]'
            "}\n"
            "Rules:\n"
            "- Rewrite ONLY the ids provided.\n"
            "- Protected blocks are not included and must not be modified.\n"
            "- Executive summary must mention the target role from the job description.\n"
            "- Use British English spelling.\n"
            "- Keep ATS optimisation natural; mirror relevant keywords but do not keyword stuff.\n"
            "- Do NOT add new personal info or change contact info.\n"
            "- NON NEGOTIABLE: Do not fabricate anything.\n"
            "- Do not add employers, degrees, titles, certifications, numbers, tools, achievements, dates, or responsibilities that are not present in the source CV blocks or explicitly provided by the user.\n"
            "- Use ONLY facts already present in the original blocks.\n"
            "- You may expand wording and emphasize keywords from the job description, but do not introduce new achievements, tools, dates, companies, or skills.\n"
            "- If the job requires something missing (for example Kubernetes), you may ONLY:\n"
            "  1) mention as 'familiar with' only if CV already implies equivalent experience, or\n"
            "  2) put it in gap_list only.\n"
            "- Do not insert gap_list items into resume updates.\n"
            "- Keep the CV truthful, coherent, and aligned to candidate seniority and timeline.\n"
            "- Every bullet must be evidence based. Prefer outcomes, scope, and metrics.\n"
            "- If metrics are missing, do not invent numbers; express impact without fabricating metrics.\n"
            "- Do NOT mention the job description explicitly.\n"
            "- Keep each update as a single line (no newlines, no tabs).\n"
            "- Keep bullet blocks as bullet-style points; do not turn bullet content into paragraph prose.\n"
            "- Keep bullet length within +/- 8 words of original_word_count.\n"
            "- Keep paragraph length within +/- 20 words of original_word_count.\n"
            "- Identify top 8 to 12 must_match_signals from the posting.\n"
            "- Identify top 6 to 10 nice_to_have_signals from the posting.\n"
            "- Identify culture_cues (ownership, cross-functional, fast paced, regulated, research heavy, client facing, etc.).\n"
            "- For each must_match signal, fill evidence_map with exact source CV lines that prove it; if no evidence, add that signal to gap_list.\n"
            "- Choose ONE positioning_angle (single concise phrase).\n"
            "- If a block cannot be improved, put its id in skipped.\n"
            "- Include only actually changed blocks in updates; unchanged go to skipped.\n\n"
            f"Job description:\n{job_description}\n\n"
            f"Special instructions:\n{instructions_text or 'None'}\n\n"
            f"Blocks:\n{json.dumps(blocks_payload, ensure_ascii=True)}\n"
        )

    def rewrite_blocks(
        self,
        job_description: str,
        blocks: List[RewriteCandidateBlock],
        special_instructions: str = "",
    ) -> Dict[str, object]:
        if not blocks:
            return {
                "updates": {},
                "skipped": [],
                "analysis": {
                    "must_match_signals": [],
                    "nice_to_have_signals": [],
                    "culture_cues": [],
                    "evidence_map": {},
                    "positioning_angle": "",
                },
                "gap_list": [],
            }

        blocks_by_id = {block.id: block for block in blocks}
        allowed_ids: Set[str] = set(blocks_by_id.keys())
        summary_block_ids = _likely_summary_block_ids(blocks)
        target_role = _extract_target_role(job_description)
        prompt = self._build_prompt_with_instructions(job_description, blocks, special_instructions)
        prompt_length = len(prompt)

        for attempt in range(2):
            response, response_meta = self._generate_with_fallback(
                prompt=prompt,
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
                must_match_signals = _coerce_str_list(payload.get("must_match_signals", []), min_len=2, max_len=120)[:12]
                nice_to_have_signals = _coerce_str_list(payload.get("nice_to_have_signals", []), min_len=2, max_len=120)[:10]
                culture_cues = _coerce_str_list(payload.get("culture_cues", []), min_len=2, max_len=120)[:10]
                evidence_map = _coerce_evidence_map(payload.get("evidence_map", {}))
                positioning_angle = _normalize_rewrite_text(payload.get("positioning_angle", "")) if isinstance(payload.get("positioning_angle", ""), str) else ""
                gap_list = _coerce_str_list(payload.get("gap_list", []), min_len=2, max_len=160)[:20]
                updates_raw_count = len(raw_updates)
                skipped_raw_count = len(raw_skipped) if isinstance(raw_skipped, list) else 0
                filtered_updates = {
                    block_id: text
                    for block_id, text in raw_updates.items()
                    if block_id in allowed_ids
                }
                dropped_unknown_id = updates_raw_count - len(filtered_updates)
                result, dropped_wordcount = _validate_and_filter_updates(
                    filtered_updates,
                    blocks_by_id,
                    job_description=job_description,
                    summary_block_ids=summary_block_ids,
                    target_role=target_role,
                )
                result["analysis"] = {
                    "must_match_signals": must_match_signals,
                    "nice_to_have_signals": nice_to_have_signals,
                    "culture_cues": culture_cues,
                    "evidence_map": evidence_map,
                    "positioning_angle": positioning_angle,
                }
                result["gap_list"] = gap_list

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

            prompt = self._build_prompt_with_instructions(job_description, blocks, special_instructions) + "\nReturn ONLY valid JSON. No extra text."

        return {
            "updates": {},
            "skipped": list(allowed_ids),
            "analysis": {
                "must_match_signals": [],
                "nice_to_have_signals": [],
                "culture_cues": [],
                "evidence_map": {},
                "positioning_angle": "",
            },
            "gap_list": [],
        }

    def extract_job_description_from_image(self, image_bytes: bytes, mime_type: str) -> str:
        if not image_bytes:
            raise ValueError("Empty image file")

        client = genai.Client(api_key=self._api_key)
        prompt = (
            "Extract the full job description text from this image.\n"
            "Return plain text only. Preserve bullet points and line ordering where possible.\n"
            "Do not summarize and do not add content."
        )

        last_exc: Exception | None = None
        for model_name in self._model_candidates():
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=[
                        types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                        prompt,
                    ],
                )
                text = (response.text or "").strip()
                if not text:
                    text = _extract_text_fallback(response)
                text = text.strip()
                if text:
                    return text
            except Exception as exc:
                last_exc = exc
                if not _should_fallback_model(exc):
                    raise
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("Could not extract job description from image")

    def smoke_test(self) -> Dict[str, object]:
        prompt = 'Return valid JSON only: {"ok": true}'
        prompt_length = len(prompt)
        try:
            text, response_meta = self._generate_with_fallback(
                prompt=prompt,
                prompt_length=prompt_length,
                blocks_count=0,
            )
            return {
                "success": True,
                "model": response_meta.get("model_used", self._model),
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
