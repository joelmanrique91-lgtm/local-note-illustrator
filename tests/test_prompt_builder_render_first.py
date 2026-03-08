from __future__ import annotations

import unittest

from app.prompt_builder import PROMPT_MAX_CHARS, build_local_fallback_result, compose_prompt_plan
from app.types import PromptIntelligenceResult


class PromptBuilderRenderFirstTests(unittest.TestCase):
    def _intelligence(self, **kwargs) -> PromptIntelligenceResult:
        base = PromptIntelligenceResult(
            source="openai",
            domain="technology_science",
            visual_strategy="conceptual",
            human_closeup_risk=2,
            avoid_close_ups=False,
            prompt_main="news scene in research lab, scientist team, working with instruments, laboratory environment",
            prompt_variants=[],
            negative_prompt="deformed anatomy",
            composition_notes="wide shot, clear foreground subject",
            style_notes="neutral color grading, editorial photography",
            confidence=0.9,
            fallback_reason=None,
            raw_schema_version="openai.v1",
        )
        return PromptIntelligenceResult(**{**base.__dict__, **kwargs})

    def test_render_first_prioritizes_semantic_core_before_style(self) -> None:
        intelligence = self._intelligence()
        plan = compose_prompt_plan(intelligence, base_negative_prompt="blurry", variants=1)

        prompt = plan.positive_prompts[0]
        self.assertIn("news scene in research lab", prompt)
        self.assertIn("scientist team", prompt)
        self.assertLess(prompt.index("scientist team"), prompt.index("neutral color grading"))

    def test_length_budget_trims_lower_priority_segments_first(self) -> None:
        long_style = ", ".join(["neutral editorial styling"] * 50)
        intelligence = self._intelligence(style_notes=long_style)

        plan = compose_prompt_plan(intelligence, base_negative_prompt="blurry", variants=1)
        prompt = plan.positive_prompts[0]

        self.assertLessEqual(len(prompt), PROMPT_MAX_CHARS)
        self.assertIn("news scene in research lab", prompt)

    def test_abstract_phrases_are_filtered_from_notes(self) -> None:
        intelligence = self._intelligence(
            composition_notes="high-stakes setting, clear foreground subject",
            style_notes="credible editorial realism, balanced editorial framing, neutral color grading",
        )

        plan = compose_prompt_plan(intelligence, base_negative_prompt="blurry", variants=1)
        prompt = plan.positive_prompts[0].lower()

        self.assertNotIn("high-stakes setting", prompt)
        self.assertNotIn("credible editorial realism", prompt)
        self.assertNotIn("balanced editorial framing", prompt)
        self.assertIn("neutral color grading", prompt)

    def test_filters_abstract_phrases_from_prompt_main(self) -> None:
        intelligence = self._intelligence(
            prompt_main=(
                "official briefing room, institutional significance, officials speaking at podium, "
                "policy environment, microphones and flags"
            )
        )

        plan = compose_prompt_plan(intelligence, base_negative_prompt="blurry", variants=1)
        prompt = plan.positive_prompts[0].lower()

        self.assertNotIn("institutional significance", prompt)
        self.assertNotIn("policy environment", prompt)
        self.assertIn("officials speaking at podium", prompt)

    def test_semantic_core_preserved_when_keywords_do_not_match(self) -> None:
        intelligence = self._intelligence(
            prompt_main=(
                "night newsroom with damaged ceiling lights, red emergency table map, "
                "city district wall board, analog siren console"
            ),
            style_notes="stylized newsroom lighting",
        )

        plan = compose_prompt_plan(intelligence, base_negative_prompt="blurry", variants=1)
        prompt = plan.positive_prompts[0]

        self.assertIn("night newsroom with damaged ceiling lights", prompt)
        self.assertIn("red emergency table map", prompt)

    def test_sports_prompt_respects_budget_after_domain_guardrails(self) -> None:
        intelligence = self._intelligence(
            domain="sports_transfers",
            visual_strategy="editorial_photo",
            prompt_main=(
                "player signing moment at training ground, coach and player handshake, "
                "stadium corridor background, photographers near tunnel"
            ),
            style_notes=", ".join(["neutral editorial styling"] * 40),
        )

        plan = compose_prompt_plan(intelligence, base_negative_prompt="blurry", variants=1)
        prompt = plan.positive_prompts[0]

        self.assertLessEqual(len(prompt), PROMPT_MAX_CHARS)
        self.assertIn("player signing moment at training ground", prompt)

    def test_valid_openai_plan_remains_compatible(self) -> None:
        intelligence = self._intelligence(
            prompt_variants=["news scene in research lab, scientist team, working with instruments, alternate angle"]
        )

        plan = compose_prompt_plan(intelligence, base_negative_prompt="blurry", variants=2)

        self.assertEqual(plan.source, "openai")
        self.assertEqual(len(plan.positive_prompts), 2)

    def test_local_fallback_plan_remains_compatible(self) -> None:
        intelligence = build_local_fallback_result(
            text="Informe sobre innovación en laboratorio científico y nuevos ensayos.",
            strategy_override="auto",
            base_negative_prompt="blurry",
            variants=1,
            fallback_reason="test",
        )

        plan = compose_prompt_plan(intelligence, base_negative_prompt="blurry", variants=1)

        self.assertEqual(plan.source, "local_fallback")
        self.assertEqual(len(plan.positive_prompts), 1)
        self.assertTrue(plan.positive_prompts[0])


if __name__ == "__main__":
    unittest.main()
