from __future__ import annotations

import logging
import unittest

from app.config import AppConfig
from app.llm_assistant import LlmAssistantSchemaError, OpenAIPromptAssistant


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
        jpeg_quality=90,
        jpeg_subsampling=0,
        value_sources={},
    )


class LlmAssistantRenderReadyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.assistant = OpenAIPromptAssistant(config=build_config(), logger=logging.getLogger("test_llm"))

    def test_render_ready_payload_is_valid(self) -> None:
        payload = {
            "domain": "technology_science",
            "visual_strategy": "conceptual",
            "human_closeup_risk": 2,
            "confidence": 0.86,
            "primary_subject": "research scientist",
            "secondary_subjects": ["lab technician"],
            "primary_action": "reviewing diagnostic monitor",
            "setting": "laboratory control room",
            "visible_objects": ["instrument panel", "safety glass"],
            "framing": "medium-wide framing",
            "mood": "focused and neutral",
            "realism_notes": "editorial documentary realism",
            "avoid_close_ups": True,
            "avoid_identity_claims": True,
            "avoid_multi_person_overload": True,
            "prompt_variants": ["alternate angle near monitor"],
            "negative_prompt": "low quality",
        }

        result = self.assistant._validate_payload(payload)

        self.assertEqual(result.raw_schema_version, "openai.v2.render_ready")
        self.assertEqual(result.semantic_validation_status, "validated")
        self.assertIn("research scientist", result.prompt_main)

    def test_abstract_payload_is_rejected(self) -> None:
        payload = {
            "domain": "political_institutional",
            "visual_strategy": "institutional",
            "human_closeup_risk": 3,
            "confidence": 0.7,
            "primary_subject": "delegation",
            "secondary_subjects": ["officials"],
            "primary_action": "discussing geopolitics",
            "setting": "policy environment",
            "visible_objects": ["symbols"],
            "framing": "wide",
            "mood": "serious",
            "realism_notes": "editorial",
            "avoid_close_ups": True,
            "avoid_identity_claims": True,
            "avoid_multi_person_overload": True,
            "prompt_variants": [],
            "negative_prompt": "low quality",
        }

        with self.assertRaises(LlmAssistantSchemaError):
            self.assistant._validate_payload(payload)

    def test_political_payload_is_simplified_when_overloaded(self) -> None:
        payload = {
            "domain": "political_institutional",
            "visual_strategy": "institutional",
            "human_closeup_risk": 4,
            "confidence": 0.82,
            "primary_subject": "Emmanuel Macron and Olaf Scholz",
            "secondary_subjects": ["leader one", "leader two", "leader three", "leader four"],
            "primary_action": "formal diplomatic handshake",
            "setting": "state hall",
            "visible_objects": ["national flags", "delegation banners", "conference badges", "table microphones", "nameplates", "backdrop"],
            "framing": "",
            "mood": "",
            "realism_notes": "",
            "avoid_close_ups": True,
            "avoid_identity_claims": True,
            "avoid_multi_person_overload": True,
            "prompt_variants": [],
            "negative_prompt": "low quality",
        }

        result = self.assistant._validate_payload(payload)

        self.assertEqual(result.semantic_validation_status, "simplified")
        self.assertIn("political_news_simplified", result.semantic_adjustment_reason or "")
        self.assertIn("senior government delegation", result.prompt_main)
        self.assertIn("state hall", result.prompt_main)
        self.assertNotIn("official conference room", result.prompt_main)

    def test_single_named_person_is_not_over_generalized(self) -> None:
        payload = {
            "domain": "political_institutional",
            "visual_strategy": "institutional",
            "human_closeup_risk": 4,
            "confidence": 0.82,
            "primary_subject": "Emmanuel Macron",
            "secondary_subjects": ["foreign minister"],
            "primary_action": "arriving for delegation meeting",
            "setting": "state hall",
            "visible_objects": ["press microphones", "conference desk"],
            "framing": "medium-wide framing",
            "mood": "formal",
            "realism_notes": "documentary realism",
            "avoid_close_ups": True,
            "avoid_identity_claims": True,
            "avoid_multi_person_overload": True,
            "prompt_variants": [],
            "negative_prompt": "low quality",
        }

        result = self.assistant._validate_payload(payload)

        self.assertEqual(result.semantic_validation_status, "validated")
        self.assertIn("Emmanuel Macron", result.prompt_main)
        self.assertNotIn("senior government delegation", result.prompt_main)

    def test_physical_political_setting_is_not_overwritten(self) -> None:
        payload = {
            "domain": "political_institutional",
            "visual_strategy": "institutional",
            "human_closeup_risk": 4,
            "confidence": 0.82,
            "primary_subject": "national delegation",
            "secondary_subjects": ["advisors"],
            "primary_action": "formal handshake",
            "setting": "state hall",
            "visible_objects": ["conference desk", "microphones"],
            "framing": "medium-wide framing",
            "mood": "formal",
            "realism_notes": "editorial documentary realism",
            "avoid_close_ups": True,
            "avoid_identity_claims": True,
            "avoid_multi_person_overload": True,
            "prompt_variants": [],
            "negative_prompt": "low quality",
        }

        result = self.assistant._validate_payload(payload)

        self.assertIn("state hall", result.prompt_main)
        self.assertNotIn("official conference room", result.prompt_main)

    def test_political_news_alias_triggers_same_policy(self) -> None:
        payload = {
            "domain": "political_news",
            "visual_strategy": "institutional",
            "human_closeup_risk": 4,
            "confidence": 0.82,
            "primary_subject": "Emmanuel Macron and Olaf Scholz",
            "secondary_subjects": ["leader one", "leader two", "leader three"],
            "primary_action": "formal diplomatic handshake",
            "setting": "state hall",
            "visible_objects": ["national flags", "delegation banners", "conference badges", "microphones", "nameplates", "backdrop", "media wall"],
            "framing": "",
            "mood": "",
            "realism_notes": "",
            "avoid_close_ups": True,
            "avoid_identity_claims": True,
            "avoid_multi_person_overload": True,
            "prompt_variants": [],
            "negative_prompt": "low quality",
        }

        result = self.assistant._validate_payload(payload)

        self.assertEqual(result.semantic_validation_status, "simplified")
        self.assertIn("political_news_simplified", result.semantic_adjustment_reason or "")


    def test_political_diplomatic_domain_alias_triggers_same_policy(self) -> None:
        payload = {
            "domain": "political_diplomatic_event",
            "visual_strategy": "institutional",
            "human_closeup_risk": 4,
            "confidence": 0.82,
            "primary_subject": "Emmanuel Macron and Olaf Scholz",
            "secondary_subjects": ["Donald Trump", "Pete Hegseth", "leader three"],
            "primary_action": "formal diplomatic handshake",
            "setting": "state hall",
            "visible_objects": ["national flags", "delegation banners", "conference badges", "microphones", "nameplates", "backdrop", "media wall"],
            "framing": "",
            "mood": "",
            "realism_notes": "",
            "avoid_close_ups": True,
            "avoid_identity_claims": True,
            "avoid_multi_person_overload": True,
            "prompt_variants": [],
            "negative_prompt": "low quality",
        }

        result = self.assistant._validate_payload(payload)

        self.assertEqual(result.semantic_validation_status, "simplified")
        self.assertIn("political_news_simplified", result.semantic_adjustment_reason or "")
        self.assertIn("senior government delegation", result.prompt_main)
        self.assertIsNotNone(result.openai_raw_payload)

    def test_simplify_return_shape_matches_unpack_contract(self) -> None:
        simplified = self.assistant._simplify_political_payload(
            domain="political_institutional",
            primary_subject="delegation",
            secondary_subjects=["a", "b", "c"],
            primary_action="formal handshake",
            setting="policy environment",
            visible_objects=["flag", "banner", "microphones"],
            framing="",
            mood="",
            realism_notes="",
        )

        self.assertEqual(len(simplified), 10)
        (
            primary_subject,
            secondary_subjects,
            primary_action,
            setting,
            visible_objects,
            framing,
            mood,
            realism_notes,
            is_simplified,
            simplification_reason,
        ) = simplified
        self.assertTrue(is_simplified)
        self.assertTrue(bool(simplification_reason))
        self.assertTrue(primary_subject)
        self.assertTrue(setting)


if __name__ == "__main__":
    unittest.main()
