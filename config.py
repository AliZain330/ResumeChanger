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
    app_username: str
    app_password: str
    rate_limit_per_minute: int
    file_retention_hours: int


def _read_int_env(name: str, default: int, min_value: int = 1) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(value, min_value)


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

    app_username = os.getenv("APP_BASIC_AUTH_USER", "").strip()
    app_password = os.getenv("APP_BASIC_AUTH_PASS", "").strip()
    rate_limit_per_minute = _read_int_env("RATE_LIMIT_PER_MINUTE", default=30, min_value=1)
    file_retention_hours = _read_int_env("FILE_RETENTION_HOURS", default=24, min_value=1)

    return Settings(
        gemini_api_key=gemini_key,
        gemini_model=model,
        app_username=app_username,
        app_password=app_password,
        rate_limit_per_minute=rate_limit_per_minute,
        file_retention_hours=file_retention_hours,
    )
