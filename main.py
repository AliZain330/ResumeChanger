import logging
import os
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.services.docx_blocks import apply_block_updates, extract_blocks
from app.services.privacy_guard import choose_rewrite_candidates
from app.services.llm_client import GeminiRewriter, RewriteCandidateBlock
from config import get_settings

app = FastAPI()

ROOT_DIR = Path(__file__).resolve().parent
APP_DIR = ROOT_DIR / "app"
UPLOADS_DIR = APP_DIR / "uploads"
OUTPUTS_DIR = APP_DIR / "outputs"

UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=str(APP_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(APP_DIR / "templates"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("resumechanger")

vertex_env_value = os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "").strip()
vertex_env_set = vertex_env_value.lower() in {"1", "true", "yes"}
if vertex_env_set:
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "0"
    logger.warning("GOOGLE_GENAI_USE_VERTEXAI was set; forcing AI Studio mode (api key).")

settings = get_settings()
has_gemini_key = bool(settings.gemini_api_key)
has_google_key = bool(os.getenv("GOOGLE_API_KEY", "").strip())
logger.info(
    "Gemini config: GEMINI_API_KEY present=%s GOOGLE_API_KEY present=%s GEMINI_MODEL=%s vertex_mode_env_set=%s vertex_mode_env_value=%s",
    has_gemini_key,
    has_google_key,
    settings.gemini_model,
    vertex_env_set,
    vertex_env_value or "unset",
)
if has_google_key:
    logger.warning("GOOGLE_API_KEY is set but ignored. Using GEMINI_API_KEY only.")

rewriter = GeminiRewriter(api_key=settings.gemini_api_key, model=settings.gemini_model)


@app.get("/", response_class=HTMLResponse)
def show_form(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})


@app.post("/extract", response_class=HTMLResponse)
async def extract(request: Request, file: UploadFile = File(...), job_description: str = Form(...)):
    file_id = uuid4().hex
    input_path = UPLOADS_DIR / f"{file_id}.docx"
    input_path.write_bytes(await file.read())

    blocks = extract_blocks(str(input_path))
    rewrite_candidates, protected_ids = choose_rewrite_candidates(blocks)
    return templates.TemplateResponse(
        "blocks.html",
        {
            "request": request,
            "blocks": blocks,
            "file_id": file_id,
            "job_description": job_description,
            "protected_ids": protected_ids,
            "rewrite_count": len(rewrite_candidates),
        },
    )


@app.post("/tailor")
def tailor(request: Request, file_id: str = Form(...), job_description: str = Form(...)):
    input_path = UPLOADS_DIR / f"{file_id}.docx"
    output_path = OUTPUTS_DIR / f"{file_id}_tailored.docx"

    blocks = extract_blocks(str(input_path))
    rewrite_candidates, protected_ids = choose_rewrite_candidates(blocks)

    candidate_blocks = [
        RewriteCandidateBlock(
            id=block["id"],
            kind=block.get("kind", "paragraph"),
            text=block.get("text", ""),
            original_word_count=len(block.get("text", "").split()),
        )
        for block in rewrite_candidates
    ]

    updates = {}
    skipped = []
    try:
        result = rewriter.rewrite_blocks(job_description, candidate_blocks)
        updates = result.get("updates", {})
        skipped = result.get("skipped", [])
    except Exception as exc:
        logger.error("Gemini rewrite failed: %s", str(exc))
        updates = {}
        skipped = [block.id for block in candidate_blocks]

    safe_updates = {block_id: text for block_id, text in updates.items() if block_id not in protected_ids}
    apply_block_updates(str(input_path), str(output_path), safe_updates)

    warning = ""
    if not safe_updates:
        warning = "AI returned no valid updates, original kept"

    return templates.TemplateResponse(
        "result.html",
        {
            "request": request,
            "file_id": file_id,
            "total_blocks": len(blocks),
            "protected_count": len(protected_ids),
            "candidates_count": len(candidate_blocks),
            "rewritten_count": len(safe_updates),
            "skipped_count": len(skipped),
            "warning": warning,
        },
    )


@app.get("/download/{file_id}")
def download(file_id: str):
    output_path = OUTPUTS_DIR / f"{file_id}_tailored.docx"
    if not output_path.exists():
        raise HTTPException(status_code=404, detail="Tailored file not found.")
    return FileResponse(
        str(output_path),
        filename="tailored.docx",
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/debug/gemini")
def debug_gemini():
    current = get_settings()
    temp_rewriter = GeminiRewriter(
        api_key=current.gemini_api_key,
        model=current.gemini_model,
    )
    return temp_rewriter.smoke_test()
