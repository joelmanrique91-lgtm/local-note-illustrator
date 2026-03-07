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

    merged = "\n".join(chunks)
    merged = re.sub(r"\s+", " ", merged).strip()
    return merged
