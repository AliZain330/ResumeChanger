from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

app = FastAPI()

app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")


@app.get("/", response_class=HTMLResponse)
def show_form(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})


@app.post("/preview", response_class=HTMLResponse)
def preview(
    request: Request,
    name: str = Form(...),
    email: str = Form(...),
    summary: str = Form(...),
    job_description: str = Form(...),
):
    resume = {
        "basics": {"name": name, "email": email},
        "summary": summary,
        "job_description": job_description,
    }
    return templates.TemplateResponse(
        "preview.html",
        {"request": request, "resume": resume},
    )


@app.get("/health")
def health():
    return {"status": "ok"}
