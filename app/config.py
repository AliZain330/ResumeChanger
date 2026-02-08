import os

from dotenv import load_dotenv


load_dotenv()

LLM_API_KEY = os.getenv("LLM_API_KEY", "").strip()
LLM_MODEL = os.getenv("LLM_MODEL", "").strip()


def get_gemini_settings() -> tuple[str, str]:
    if not LLM_API_KEY:
        raise ValueError("LLM_API_KEY is missing. Set it in your .env file.")
    if not LLM_MODEL:
        raise ValueError("LLM_MODEL is missing. Set it in your .env file.")
    return LLM_API_KEY, LLM_MODEL
