from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from app.types import DocumentManifest, ImageManifest, RunManifest


class RunManifestWriter:
    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.run_manifest: RunManifest | None = None
        self.file_path: Path | None = None

    def start(
        self,
        selected_root_folder: Path,
        include_subfolders: bool,
        images_per_document: int,
        config_snapshot: dict[str, object],
        runtime_effective: dict[str, object] | None = None,
    ) -> RunManifest:
        manifest = RunManifest.create(
            selected_root_folder=selected_root_folder,
            include_subfolders=include_subfolders,
            images_per_document=images_per_document,
            config_snapshot=config_snapshot,
        )
        manifest.runtime_effective = runtime_effective
        self.run_manifest = manifest
        self.file_path = self.log_dir / f"run_manifest_{manifest.run_id}.json"
        self._persist()
        return manifest

    def add_document(self, document: DocumentManifest) -> None:
        if self.run_manifest is None:
            raise RuntimeError("Run manifest no inicializado")
        self.run_manifest.documents.append(document)
        self._persist()

    def append_output(
        self,
        document: DocumentManifest,
        image_index: int,
        output_path: Path,
        file_size_bytes: int | None = None,
        device_at_generation: str | None = None,
        dtype_at_generation: str | None = None,
        cuda_fallback_triggered: bool | None = None,
    ) -> None:
        document.outputs.append(
            ImageManifest(
                image_index=image_index,
                output_path=str(output_path),
                file_size_bytes=file_size_bytes,
                device_at_generation=device_at_generation,
                dtype_at_generation=dtype_at_generation,
                cuda_fallback_triggered=cuda_fallback_triggered,
            )
        )
        self._persist()

    def mark_document_error(self, document: DocumentManifest, error: str) -> None:
        document.error = error
        self._persist()

    def finish(self, status: str) -> Path:
        if self.run_manifest is None or self.file_path is None:
            raise RuntimeError("Run manifest no inicializado")
        self.run_manifest.status = status
        self.run_manifest.finished_at = datetime.now().isoformat(timespec="seconds")
        self._persist()
        return self.file_path

    def _persist(self) -> None:
        if self.run_manifest is None or self.file_path is None:
            return
        payload = asdict(self.run_manifest)
        with self.file_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
