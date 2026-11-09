"""Tweet text cleaning."""
import re

_URL = re.compile(r"https?://\S+|www\.\S+")
_USER = re.compile(r"@\w+")
_SPACES = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Lowercase, replace URLs/user mentions with tokens, strip '#', collapse spaces."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = _URL.sub(" http ", text)
    text = _USER.sub(" @user ", text)
    text = text.replace("#", "")
    return _SPACES.sub(" ", text).strip()
