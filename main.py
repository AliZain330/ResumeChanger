import logging
import os
import secrets
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Deque, Dict
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
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
auth_enabled = bool(settings.app_username and settings.app_password)
logger.info(
    "Gemini config: GEMINI_API_KEY present=%s GOOGLE_API_KEY present=%s GEMINI_MODEL=%s vertex_mode_env_set=%s vertex_mode_env_value=%s auth_enabled=%s rate_limit_per_minute=%s file_retention_hours=%s",
    has_gemini_key,
    has_google_key,
    settings.gemini_model,
    vertex_env_set,
    vertex_env_value or "unset",
    auth_enabled,
    settings.rate_limit_per_minute,
    settings.file_retention_hours,
)
if has_google_key:
    logger.warning("GOOGLE_API_KEY is set but ignored. Using GEMINI_API_KEY only.")

rewriter = GeminiRewriter(api_key=settings.gemini_api_key, model=settings.gemini_model)
_request_windows: Dict[str, Deque[float]] = defaultdict(deque)


def _prune_old_files() -> None:
    threshold_seconds = settings.file_retention_hours * 3600
    now = time.time()
    for folder in (UPLOADS_DIR, OUTPUTS_DIR):
        for path in folder.glob("*"):
            if not path.is_file():
                continue
            age = now - path.stat().st_mtime
            if age > threshold_seconds:
                try:
                    path.unlink()
                except OSError:
                    logger.warning("Failed to delete expired file: %s", path)


def _get_basic_auth_parts(request: Request) -> tuple[str, str] | None:
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Basic "):
        return None
    import base64

    token = auth_header.removeprefix("Basic ").strip()
    if not token:
        return None
    try:
        raw = base64.b64decode(token).decode("utf-8")
    except Exception:
        return None
    if ":" not in raw:
        return None
    username, password = raw.split(":", 1)
    return username, password


def _auth_failed_response() -> Response:
    return Response(
        status_code=401,
        headers={"WWW-Authenticate": 'Basic realm="ResumeChanger"'},
    )


@app.middleware("http")
async def add_security_guards(request: Request, call_next):
    if request.url.path.startswith("/static"):
        return await call_next(request)

    if auth_enabled:
        auth_parts = _get_basic_auth_parts(request)
        if not auth_parts:
            return _auth_failed_response()
        username, password = auth_parts
        if not (
            secrets.compare_digest(username, settings.app_username)
            and secrets.compare_digest(password, settings.app_password)
        ):
            return _auth_failed_response()

    client_host = request.client.host if request.client else "unknown"
    now = time.time()
    window = _request_windows[client_host]
    while window and (now - window[0]) > 60:
        window.popleft()
    if len(window) >= settings.rate_limit_per_minute:
        return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded. Try again shortly."})
    window.append(now)

    return await call_next(request)


@app.get("/", response_class=HTMLResponse)
def show_form(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})


@app.post("/extract", response_class=HTMLResponse)
async def extract(request: Request, file: UploadFile = File(...), job_description: str = Form(...)):
    _prune_old_files()
    ext = Path(file.filename or "").suffix.lower()
    if ext not in {".docx", ".pdf"}:
        raise HTTPException(status_code=400, detail="Unsupported file type. Upload .docx or .pdf.")

    file_id = uuid4().hex
    input_path = UPLOADS_DIR / f"{file_id}{ext}"
    input_path.write_bytes(await file.read())

    blocks = extract_blocks(str(input_path))
    rewrite_candidates, protected_ids = choose_rewrite_candidates(blocks)
    return templates.TemplateResponse(
        "blocks.html",
        {
            "request": request,
            "blocks": blocks,
            "file_id": file_id,
            "file_ext": ext,
            "job_description": job_description,
            "protected_ids": protected_ids,
            "rewrite_count": len(rewrite_candidates),
        },
    )


@app.post("/tailor")
def tailor(
    request: Request,
    file_id: str = Form(...),
    file_ext: str = Form(...),
    job_description: str = Form(...),
):
    _prune_old_files()
    ext = file_ext.lower().strip()
    if ext not in {".docx", ".pdf"}:
        raise HTTPException(status_code=400, detail="Unsupported source format.")
    input_path = UPLOADS_DIR / f"{file_id}{ext}"
    output_path = OUTPUTS_DIR / f"{file_id}_tailored.docx"
    if not input_path.exists():
        raise HTTPException(status_code=404, detail="Source file not found.")

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
            "source_format": ext,
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
    _prune_old_files()
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
    _prune_old_files()
    current = get_settings()
    temp_rewriter = GeminiRewriter(
        api_key=current.gemini_api_key,
        model=current.gemini_model,
    )
    return temp_rewriter.smoke_test()