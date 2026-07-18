import os
from pathlib import Path


_SERVICE_NAME = "VisioALS"
_ELEVENLABS_ACCOUNT = "elevenlabs_api_key"


def _load_local_env_key() -> str:
    """Read the developer-only key from the git-ignored project .env file."""
    env_path = Path(__file__).resolve().with_name(".env")
    try:
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, value = line.split("=", 1)
            if name.strip() == "ELEVENLABS_API_KEY":
                return value.strip().strip('"').strip("'")
    except FileNotFoundError:
        pass
    except OSError as e:
        print(f"could not read local environment file: {e}")
    return ""


def load_elevenlabs_api_key() -> str:
    """Load the API key without storing it in the patient profile or config JSON."""
    env_key = os.environ.get("ELEVENLABS_API_KEY", "").strip()
    if env_key:
        return env_key

    local_key = _load_local_env_key()
    if local_key:
        return local_key

    try:
        import keyring
        return (keyring.get_password(_SERVICE_NAME, _ELEVENLABS_ACCOUNT) or "").strip()
    except Exception as e:
        print(f"credential store unavailable: {e}")
        return ""


def save_elevenlabs_api_key(api_key: str) -> bool:
    """Persist the API key in the operating system credential store."""
    key = (api_key or "").strip()
    if not key:
        return False

    try:
        import keyring
        keyring.set_password(_SERVICE_NAME, _ELEVENLABS_ACCOUNT, key)
        return True
    except Exception as e:
        print(f"could not save ElevenLabs key to credential store: {e}")
        return False
