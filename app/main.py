from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, Request, Form, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.services.docx_blocks import extract_blocks, apply_block_updates
from app.config import get_gemini_settings
from app.services.llm_client import rewrite_blocks
from app.services.privacy_guard import choose_rewrite_candidates

app = FastAPI()

APP_DIR = Path(__file__).resolve().parent
UPLOADS_DIR = APP_DIR / "uploads"
OUTPUTS_DIR = APP_DIR / "outputs"

UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=str(APP_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(APP_DIR / "templates"))


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
def tailor(
    request: Request,
    file_id: str = Form(...),
    job_description: str = Form(...),
):
    input_path = UPLOADS_DIR / f"{file_id}.docx"
    output_path = OUTPUTS_DIR / f"{file_id}_tailored.docx"

    blocks = extract_blocks(str(input_path))
    rewrite_candidates, protected_ids = choose_rewrite_candidates(blocks)
    for block in rewrite_candidates:
        block["original_word_count"] = len(block.get("text", "").split())

    updates = {}
    skipped = []
    try:
        api_key, model = get_gemini_settings()
        llm_result = rewrite_blocks(job_description, rewrite_candidates, api_key, model)
        updates = llm_result.get("updates", {})
        skipped = llm_result.get("skipped", [])
    except Exception:
        updates = {}
        skipped = []

    apply_block_updates(str(input_path), str(output_path), updates)

    warning = ""
    if not updates:
        warning = "AI returned no valid updates, original kept"

    return templates.TemplateResponse(
        "result.html",
        {
            "request": request,
            "file_id": file_id,
            "total_blocks": len(blocks),
            "protected_count": len(protected_ids),
            "rewrite_count": len(rewrite_candidates),
            "rewritten_count": len(updates),
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
