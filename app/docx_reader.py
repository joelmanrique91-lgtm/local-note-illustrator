from __future__ import annotations

import re
from pathlib import Path

from docx import Document



def read_docx_text(path: Path) -> str:
    if path.suffix.lower() != ".docx":
        raise ValueError(f"Archivo no soportado: {path}")

    doc = Document(str(path))
    chunks: list[str] = []
    for paragraph in doc.paragraphs:
        text = paragraph.text.strip()
        if text:
            chunks.append(text)

    normalized = [re.sub(r"\s+", " ", chunk).strip() for chunk in chunks]
    return "\n".join([chunk for chunk in normalized if chunk])
