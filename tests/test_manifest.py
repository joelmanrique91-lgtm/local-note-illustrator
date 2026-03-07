from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from app.manifest import RunManifestWriter
from app.types import DocumentManifest


class ManifestTests(unittest.TestCase):
    def test_manifest_write(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp)
            writer = RunManifestWriter(log_dir)
            run = writer.start(Path("/tmp/docs"), True, 1, {"model_id": "x"})
            doc = DocumentManifest(
                document_path="/tmp/docs/a.docx",
                source="openai",
                strategy_override="auto",
                strategy_suggested="conceptual",
                strategy_effective="conceptual",
                domain="technology_science",
                preset="balanced",
                seed=None,
                width=1024,
                height=1024,
                steps=30,
                guidance_scale=6.8,
                openai_status="success",
                openai_model="gpt-4.1-mini",
                prompt_source="openai",
            )
            writer.add_document(doc)
            writer.append_output(doc, 1, Path("/tmp/docs/a_img_01.jpg"))
            path = writer.finish("success")

            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["run_id"], run.run_id)
            self.assertEqual(payload["status"], "success")
            self.assertEqual(payload["documents"][0]["source"], "openai")


if __name__ == "__main__":
    unittest.main()
