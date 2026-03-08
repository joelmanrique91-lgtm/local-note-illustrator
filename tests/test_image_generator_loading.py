from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from app.config import AppConfig
from app.image_generator import ImageGenerator, ModelLoadError


class _FakePipeline:
    def to(self, _device: str):
        return self

    def enable_attention_slicing(self) -> None:
        return None


class ImageGeneratorLoadingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = AppConfig(
            model_id="SG161222/RealVisXL_V5.0",
            default_negative_prompt="blurry",
            default_num_images=1,
            default_steps=30,
            default_guidance_scale=6.8,
            default_width=1024,
            default_height=1024,
            output_format="jpg",
            log_dir=Path("logs"),
            app_log_file="app.log",
            force_cpu=True,
            default_preset="balanced",
            openai_enable=True,
            openai_api_key="",
            openai_model="gpt-4.1-mini",
            openai_timeout_seconds=20,
            openai_max_retries=1,
            openai_prompt_intelligence_mode="required_with_safety_fallback",
            openai_max_input_chars=8000,
            openai_strict_schema=True,
            jpeg_quality=90,
            jpeg_subsampling=0,
            value_sources={},
        )
        self.logger = Mock()

    def test_cuda_load_tries_fp16_variant_then_falls_back_without_variant(self) -> None:
        generator = ImageGenerator(self.config, self.logger)
        fake_pipe = _FakePipeline()

        first_error = Exception("Error no file named diffusion_pytorch_model.fp16.safetensors for variant fp16")
        with patch("app.image_generator.StableDiffusionXLPipeline.from_pretrained", side_effect=[first_error, fake_pipe]) as mocked:
            loaded = generator._initialize_pipeline_on_device("cuda")

        self.assertIs(loaded, fake_pipe)
        self.assertEqual(mocked.call_count, 2)

        first_call_kwargs = mocked.call_args_list[0].kwargs
        second_call_kwargs = mocked.call_args_list[1].kwargs

        self.assertTrue(first_call_kwargs["use_safetensors"])
        self.assertEqual(first_call_kwargs["variant"], "fp16")
        self.assertEqual(second_call_kwargs["use_safetensors"], True)
        self.assertNotIn("variant", second_call_kwargs)

    def test_load_error_message_is_clear_for_hf_auth(self) -> None:
        generator = ImageGenerator(self.config, self.logger)
        with patch(
            "app.image_generator.StableDiffusionXLPipeline.from_pretrained",
            side_effect=Exception("401 Unauthorized from huggingface_hub"),
        ):
            with self.assertRaises(ModelLoadError) as ctx:
                generator._initialize_pipeline_on_device("cpu")

        self.assertIn("No hay acceso al repositorio en Hugging Face", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
