import os
from dataclasses import dataclass
from pathlib import Path


_ROOT_DIR = Path(__file__).resolve().parent
_DOTENV_PATH = _ROOT_DIR / ".env"


def _load_env_file() -> dict[str, str]:
    """Read .env as raw bytes, strip BOM, parse key=value pairs."""
    if not _DOTENV_PATH.exists():
        return {}
    raw = _DOTENV_PATH.read_bytes()
    # Strip UTF-8 BOM
    if raw.startswith(b"\xef\xbb\xbf"):
        raw = raw[3:]
    # Strip UTF-16 LE BOM and decode
    if raw.startswith(b"\xff\xfe"):
        text = raw.decode("utf-16-le").lstrip("\ufeff")
    else:
        text = raw.decode("utf-8", errors="replace")

    env_vars: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip().lstrip("\ufeff")
        value = value.strip().strip("\"'")
        if key:
            env_vars[key] = value
    return env_vars


# Load .env into os.environ at import time
_file_env = _load_env_file()
for _k, _v in _file_env.items():
    os.environ.setdefault(_k, _v)


@dataclass(frozen=True)
class Settings:
    gemini_api_key: str
    gemini_model: str


def get_settings() -> Settings:
    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
    model = os.getenv("GEMINI_MODEL", "").strip()

    # Fallback: read directly from parsed file values
    if not gemini_key:
        gemini_key = _file_env.get("GEMINI_API_KEY", "").strip()
    if not model:
        model = _file_env.get("GEMINI_MODEL", "").strip()

    if not gemini_key:
        found_keys = sorted(_file_env.keys())
        raise ValueError(
            f"GEMINI_API_KEY is missing. .env at {_DOTENV_PATH} "
            f"(exists: {_DOTENV_PATH.exists()}). Keys found: {found_keys}"
        )
    if not model:
        model = "gemini-3-flash-preview"

    return Settings(gemini_api_key=gemini_key, gemini_model=model)
