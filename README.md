## ResumeChanger

Web app that tailors resumes to a job description using Gemini.

### Features

- Upload `.docx` or `.pdf` resume files.
- Extract text blocks and classify them as paragraph/bullet.
- Protect likely sensitive/structural blocks (top lines, contact data, headings).
- Rewrite candidate blocks with Gemini and strict constraints.
- Download tailored output as `.docx` (PDF inputs are exported to DOCX).
- Optional basic auth and built-in request rate limiting.
- Automatic cleanup for old uploaded/output files.

### Setup

1. Install dependencies:
   - `pip install -r requirements.txt`
2. Create `.env` with at least:
   - `GEMINI_API_KEY=...`
   - Optional: `GEMINI_MODEL=gemini-2.5-flash`
3. Start app:
   - `uvicorn main:app --reload`

### Optional security and runtime settings

- `APP_BASIC_AUTH_USER` and `APP_BASIC_AUTH_PASS` enable HTTP Basic auth.
- `RATE_LIMIT_PER_MINUTE` sets per-IP request cap (default `30`).
- `FILE_RETENTION_HOURS` controls cleanup retention (default `24`).

### Endpoints

- `GET /` upload form
- `POST /extract` parse blocks and preview protection
- `POST /tailor` run Gemini tailoring and generate output
- `GET /download/{file_id}` download tailored DOCX
- `GET /health` health check
- `GET /debug/gemini` Gemini connectivity smoke test