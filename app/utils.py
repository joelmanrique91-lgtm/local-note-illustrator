from __future__ import annotations

import os
import sys
from pathlib import Path



def open_path(path: Path) -> None:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No existe la ruta: {path}")

    if sys.platform.startswith("win"):
        os.startfile(str(path))  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        os.system(f'open "{path}"')
    else:
        os.system(f'xdg-open "{path}"')



def safe_slug(text: str) -> str:
    allowed = "-_"
    cleaned = "".join(ch if ch.isalnum() or ch in allowed else "_" for ch in text.strip())
    cleaned = "_".join(filter(None, cleaned.split("_")))
    return cleaned[:80] or "documento"
