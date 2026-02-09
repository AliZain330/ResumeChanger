# ResumeChanger

## Setup

Create a `.env` file in the project root:

```
GEMINI_API_KEY=your_key_here
GEMINI_MODEL=gemini-3-flash-preview
```

Install dependencies and run:

```
pip install -r requirements.txt
python -m uvicorn main:app
```

## Gemini Smoke Test

Use the debug endpoint to validate your Gemini API key and model:

- `GET http://127.0.0.1:8000/debug/gemini`

The response includes:

- `success` (true/false)
- `model`
- `response_text_len`
or an error message if it fails.
