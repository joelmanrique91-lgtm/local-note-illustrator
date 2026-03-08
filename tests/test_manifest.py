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
            run.runtime_effective = {"device": {"value": "cuda", "source": "runtime"}}
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
                semantic_adjustment_reason="political_news_simplified:secondary_subjects_capped",
                semantic_validation_status="simplified",
                openai_raw_payload={
                    "domain": "political_diplomatic_event",
                    "visual_strategy": "institutional",
                    "primary_subject": "Donald Trump and Pete Hegseth",
                    "secondary_subjects": ["delegates"],
                    "setting": "official conference room",
                    "composition_notes": "medium-wide framing",
                    "style_notes": "formal and neutral",
                },
                validated_prompt_main="senior government delegation, official conference room",
                final_positive_prompt="official delegation meeting, conference room",
                final_negative_prompt="blurry, low quality",
                sanitation_flags={
                    "political_guard_triggered": True,
                    "multi_name_sanitized": True,
                    "political_domain_equivalent_detected": True,
                    "anti_text_negative_applied": True,
                },
                runtime_effective={"prompt_source": {"value": "openai", "source": "runtime"}},
            )
            writer.add_document(doc)
            writer.append_output(
                doc,
                1,
                Path("/tmp/docs/a_img_01.jpg"),
                file_size_bytes=12345,
                device_at_generation="cuda",
                dtype_at_generation="float16",
                cuda_fallback_triggered=False,
            )
            path = writer.finish("success")

            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["run_id"], run.run_id)
            self.assertEqual(payload["status"], "success")
            self.assertEqual(payload["documents"][0]["source"], "openai")
            self.assertIn("official delegation meeting", payload["documents"][0]["final_positive_prompt"])
            self.assertEqual(payload["documents"][0]["final_negative_prompt"], "blurry, low quality")
            self.assertEqual(payload["documents"][0]["openai_raw_payload"]["domain"], "political_diplomatic_event")
            self.assertIn("senior government delegation", payload["documents"][0]["validated_prompt_main"])
            self.assertTrue(payload["documents"][0]["sanitation_flags"]["multi_name_sanitized"])
            self.assertEqual(payload["runtime_effective"]["device"]["value"], "cuda")
            self.assertEqual(payload["documents"][0]["outputs"][0]["file_size_bytes"], 12345)


if __name__ == "__main__":
    unittest.main()
