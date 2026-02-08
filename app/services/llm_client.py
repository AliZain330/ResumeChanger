from __future__ import annotations

import json
from typing import Dict, List

from google import genai


def call_llm(prompt: str, api_key: str, model: str) -> str:
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(model=model, contents=prompt)
    return response.text or ""


def _word_count(text: str) -> int:
    return len([word for word in text.strip().split() if word])


def _validate_and_filter_updates(
    updates: Dict[str, str],
    blocks_by_id: Dict[str, dict],
) -> Dict[str, object]:
    filtered: Dict[str, str] = {}
    skipped: List[str] = []

    for block_id, new_text in updates.items():
        if block_id not in blocks_by_id:
            skipped.append(block_id)
            continue
        if not isinstance(new_text, str):
            skipped.append(block_id)
            continue

        block = blocks_by_id[block_id]
        original_count = int(block.get("original_word_count", 0))
        new_count = _word_count(new_text)
        kind = block.get("kind", "paragraph")

        if kind == "bullet":
            if not (original_count - 3 <= new_count <= original_count + 3):
                skipped.append(block_id)
                continue
        else:
            if not (original_count - 10 <= new_count <= original_count + 10):
                skipped.append(block_id)
                continue

        filtered[block_id] = new_text

    return {"updates": filtered, "skipped": sorted(set(skipped))}


def rewrite_blocks(
    job_description: str,
    blocks: List[dict],
    api_key: str,
    model: str,
) -> Dict[str, object]:
    blocks_by_id = {block["id"]: block for block in blocks}
    blocks_payload = [
        {
            "id": block["id"],
            "kind": block.get("kind", "paragraph"),
            "text": block.get("text", ""),
            "original_word_count": block.get("original_word_count", 0),
        }
        for block in blocks
    ]

    base_prompt = (
        "You are rewriting resume blocks to better fit the job description.\n"
        "Return STRICT JSON only in this exact shape:\n"
        '{"updates": {"b0007": "new text", "b0012": "new text"}, "skipped": ["b0019"]}\n'
        "Rules:\n"
        "- Do NOT add new personal info or change contact info.\n"
        "- Do NOT invent metrics, awards, companies, or dates.\n"
        "- Do NOT change headings or section titles.\n"
        "- Tailor content to the job description.\n"
        "- Keep each bullet within +/- 3 words of original_word_count.\n"
        "- Keep each paragraph within +/- 10 words of original_word_count.\n"
        "- Return updates ONLY for IDs provided in the blocks list.\n\n"
        f"Job description:\n{job_description}\n\n"
        f"Blocks:\n{json.dumps(blocks_payload, ensure_ascii=True)}\n"
    )

    prompt = base_prompt
    for attempt in range(2):
        response = call_llm(prompt, api_key=api_key, model=model)
        try:
            payload = json.loads(response)
        except json.JSONDecodeError:
            payload = None

        if isinstance(payload, dict) and isinstance(payload.get("updates"), dict):
            result = _validate_and_filter_updates(payload.get("updates", {}), blocks_by_id)
            if result["updates"]:
                return result
            return {"updates": {}, "skipped": result["skipped"]}

        prompt = base_prompt + "\nReturn ONLY valid JSON. No extra text."

    return {"updates": {}, "skipped": ["invalid_response"]}
