from __future__ import annotations

import logging
import unittest
from unittest.mock import patch

from app.config import AppConfig
from app.prompt_intelligence import resolve_prompt_plan
from app.types import PromptIntelligenceResult


def build_config() -> AppConfig:
    from pathlib import Path

    return AppConfig(
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
    )


class PromptIntelligenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = build_config()
        self.logger = logging.getLogger("test_prompt_intelligence")

    def test_openai_success(self) -> None:
        fake = PromptIntelligenceResult(
            source="openai",
            domain="technology_science",
            visual_strategy="conceptual",
            human_closeup_risk=2,
            avoid_close_ups=False,
            prompt_main="editorial lab scene, realistic",
            prompt_variants=["editorial lab scene, alternate angle"],
            negative_prompt="deformed anatomy",
            composition_notes="wide composition",
            style_notes="neutral grading",
            confidence=0.88,
            fallback_reason=None,
            raw_schema_version="openai.v1",
        )

        with patch("app.prompt_intelligence.OpenAIPromptAssistant.generate_prompt_intelligence", return_value=fake):
            res = resolve_prompt_plan(
                text="Informe sobre laboratorio e investigación",
                strategy_override="auto",
                config=self.config,
                logger=self.logger,
                variants=2,
            )

        self.assertEqual(res.prompt_plan.source, "openai")
        self.assertEqual(res.openai_status, "success")
        self.assertEqual(res.intelligence.fallback_reason, None)

    def test_timeout_fallback(self) -> None:
        from app.llm_assistant import LlmAssistantTimeoutError

        with patch(
            "app.prompt_intelligence.OpenAIPromptAssistant.generate_prompt_intelligence",
            side_effect=LlmAssistantTimeoutError("timeout"),
        ):
            res = resolve_prompt_plan(
                text="Documento cualquiera",
                strategy_override="auto",
                config=self.config,
                logger=self.logger,
                variants=1,
            )

        self.assertEqual(res.prompt_plan.source, "local_fallback")
        self.assertEqual(res.openai_status, "fallback")
        self.assertTrue((res.intelligence.fallback_reason or "").startswith("timeout:"))

    def test_schema_invalid_fallback(self) -> None:
        from app.llm_assistant import LlmAssistantSchemaError

        with patch(
            "app.prompt_intelligence.OpenAIPromptAssistant.generate_prompt_intelligence",
            side_effect=LlmAssistantSchemaError("bad schema"),
        ):
            res = resolve_prompt_plan(
                text="Documento cualquiera",
                strategy_override="auto",
                config=self.config,
                logger=self.logger,
                variants=1,
            )

        self.assertEqual(res.prompt_plan.source, "local_fallback")
        self.assertIn("schema_invalid", res.intelligence.fallback_reason or "")

    def test_manual_strategy_override(self) -> None:
        fake = PromptIntelligenceResult(
            source="openai",
            domain="technology_science",
            visual_strategy="conceptual",
            human_closeup_risk=2,
            avoid_close_ups=False,
            prompt_main="editorial lab scene, realistic",
            prompt_variants=[],
            negative_prompt="deformed anatomy",
            composition_notes="wide composition",
            style_notes="neutral grading",
            confidence=0.88,
            fallback_reason=None,
            raw_schema_version="openai.v1",
        )

        with patch("app.prompt_intelligence.OpenAIPromptAssistant.generate_prompt_intelligence", return_value=fake):
            res = resolve_prompt_plan(
                text="Informe sobre laboratorio e investigación",
                strategy_override="industrial",
                config=self.config,
                logger=self.logger,
                variants=1,
            )

        self.assertEqual(res.prompt_plan.strategy_effective, "industrial")
        self.assertEqual(res.intelligence.visual_strategy, "industrial")


if __name__ == "__main__":
    unittest.main()
