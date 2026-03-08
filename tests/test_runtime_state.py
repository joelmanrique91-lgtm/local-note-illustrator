from __future__ import annotations

import unittest
from pathlib import Path

from app.config import AppConfig
from app.runtime_state import RuntimeResolver, build_export_payload


class RuntimeStateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = AppConfig(
            model_id="stabilityai/stable-diffusion-xl-base-1.0",
            default_negative_prompt="blurry, watermark",
            default_num_images=1,
            default_steps=30,
            default_guidance_scale=6.8,
            default_width=1024,
            default_height=1024,
            output_format="jpg",
            log_dir=Path("logs"),
            app_log_file="app.log",
            force_cpu=False,
            default_preset="balanced",
            openai_enable=True,
            openai_api_key="test-key",
            openai_model="gpt-4.1-mini",
            openai_timeout_seconds=20,
            openai_max_retries=1,
            openai_prompt_intelligence_mode="required_with_safety_fallback",
            openai_max_input_chars=8000,
            openai_strict_schema=True,
            jpeg_quality=90,
            jpeg_subsampling=0,
            value_sources={
                "model_id": "env",
                "output_format": "default",
                "openai_enable": "env",
                "openai_prompt_intelligence_mode": "env",
                "openai_model": "env",
                "jpeg_quality": "default",
                "jpeg_subsampling": "default",
            },
        )

    def test_runtime_resolution_has_effective_values_and_sources(self) -> None:
        resolver = RuntimeResolver(self.config)
        runtime = resolver.resolve_run_runtime(
            preset_name="balanced",
            strategy_override="auto",
            seed=123,
            images_per_document=2,
            backend_state={
                "pipeline_class": "StableDiffusionXLPipeline",
                "device": "cuda",
                "dtype": "float16",
                "cuda_fallback_triggered": False,
            },
        )

        self.assertEqual(runtime.width.value, 1024)
        self.assertEqual(runtime.width.source, "preset")
        self.assertEqual(runtime.model_id.source, "env")
        self.assertEqual(runtime.seed.source, "gui")
        self.assertEqual(runtime.images_per_document.source, "gui")

    def test_export_payload_uses_runtime_source_of_truth(self) -> None:
        resolver = RuntimeResolver(self.config)
        runtime = resolver.resolve_run_runtime(
            preset_name="balanced",
            strategy_override="editorial_photo",
            seed=None,
            images_per_document=1,
            backend_state={
                "pipeline_class": "StableDiffusionXLPipeline",
                "device": "cpu",
                "dtype": "float32",
                "cuda_fallback_triggered": True,
            },
        )

        payload = build_export_payload(
            runtime=runtime,
            selected_root_folder="/tmp/docs",
            per_document=[
                {
                    "document_path": "/tmp/docs/a.docx",
                    "prompt_source": "openai",
                    "strategy_effective": "institutional",
                    "domain": "political_institutional",
                    "final_positive_prompt": "photojournalistic photograph, delegation at summit room",
                    "final_negative_prompt": "blurry, watermark",
                    "width": 1024,
                    "height": 1024,
                    "steps": 30,
                    "guidance_scale": 6.8,
                    "outputs": [
                        {
                            "image_index": 1,
                            "output_path": "/tmp/docs/a_img_01.jpg",
                            "file_size_bytes": 123456,
                        }
                    ],
                }
            ],
            torch_version="2.x",
            diffusers_version="0.x",
            generated_at="2026-01-01T00:00:00",
        )

        self.assertIn("runtime_effective", payload)
        self.assertEqual(payload["runtime_effective"]["device"]["value"], "cpu")
        self.assertEqual(payload["runtime_effective"]["cuda_fallback_triggered"]["value"], True)
        self.assertEqual(payload["documents"][0]["domain"], "political_institutional")
        self.assertEqual(payload["documents"][0]["outputs"][0]["file_size_bytes"], 123456)


if __name__ == "__main__":
    unittest.main()
