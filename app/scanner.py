from __future__ import annotations

from pathlib import Path
from typing import List



def scan_docx_files(root: Path, include_subfolders: bool = True) -> List[Path]:
    if not root.exists() or not root.is_dir():
        raise NotADirectoryError(f"Carpeta inválida: {root}")

    pattern = "**/*.docx" if include_subfolders else "*.docx"
    files: list[Path] = []
    for path in sorted(root.glob(pattern)):
        if path.name.startswith("~$"):
            continue
        if path.is_file() and path.suffix.lower() == ".docx":
            files.append(path)
    return files
