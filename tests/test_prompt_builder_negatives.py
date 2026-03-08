from __future__ import annotations

import unittest

from app.prompt_builder import BASE_NEGATIVE_TERMS, compose_prompt_plan
from app.types import PromptIntelligenceResult


class PromptBuilderNegativePromptTests(unittest.TestCase):
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

    def test_base_negative_terms_are_minimal_universal_core(self) -> None:
        self.assertEqual(
            BASE_NEGATIVE_TERMS,
            [
                "blurry",
                "blurry details",
                "low quality",
                "jpeg artifacts",
                "unreadable text",
                "watermark",
                "logo",
                "deformed anatomy",
                "malformed body",
                "broken proportions",
                "extra limbs",
                "deformed hands",
                "extra fingers",
                "duplicate face",
                "asymmetrical eyes",
                "crossed eyes",
                "plastic skin",
                "uncanny",
            ],
        )

    def test_non_sports_domain_does_not_include_sports_negatives(self) -> None:
        intelligence = self._intelligence(domain="technology_science")

        plan = compose_prompt_plan(intelligence, base_negative_prompt="blurry", variants=1)

        self.assertNotIn("badge", plan.negative_prompt)
        self.assertNotIn("soccer emblem collage", plan.negative_prompt)

    def test_sports_domain_includes_pruned_sports_negatives(self) -> None:
        intelligence = self._intelligence(domain="sports_transfers", visual_strategy="editorial_photo")

        plan = compose_prompt_plan(intelligence, base_negative_prompt="blurry", variants=1)

        self.assertIn("badge", plan.negative_prompt)
        self.assertIn("soccer emblem collage", plan.negative_prompt)
        terms = [term.strip().lower() for term in plan.negative_prompt.split(",")]
        self.assertNotIn("poster", terms)
        self.assertNotIn("infographic", terms)
        self.assertNotIn("readable text", terms)

    def test_infographic_strategy_avoids_infographic_negative_conflict(self) -> None:
        intelligence = self._intelligence(
            visual_strategy="infographic_like",
            prompt_main="editorial infographic-like composition with chart panel and institutional context",
            negative_prompt="infographic, infographic board",
        )

        plan = compose_prompt_plan(intelligence, base_negative_prompt="blurry", variants=1)

        self.assertNotIn("infographic", plan.negative_prompt)
        self.assertIn("blurry", plan.negative_prompt)

    def test_negative_terms_deduplicate_normalized_variants(self) -> None:
        intelligence = self._intelligence(negative_prompt="JPEG artifacts,   jpeg artifact, LOW-QUALITY")

        plan = compose_prompt_plan(intelligence, base_negative_prompt=" Low Quality ", variants=1)
        lowered = plan.negative_prompt.lower()

        self.assertEqual(lowered.count("jpeg artifacts"), 1)
        self.assertEqual(lowered.count("low quality"), 1)

    def test_political_group_scene_suppresses_conflicting_group_negatives(self) -> None:
        intelligence = self._intelligence(
            domain="political_institutional",
            visual_strategy="institutional",
            prompt_main="summit conference with delegation group at official press room",
            negative_prompt="chaotic crowd, large groups, extreme close-up portrait",
        )

        plan = compose_prompt_plan(intelligence, base_negative_prompt="blurry", variants=1)
        lowered = plan.negative_prompt.lower()

        self.assertNotIn("chaotic crowd", lowered)
        self.assertNotIn("large groups", lowered)
        self.assertNotIn("extreme close-up portrait", lowered)
        self.assertIn("deformed hands", lowered)

    def test_sports_scene_suppresses_group_conflicts_but_keeps_structural_terms(self) -> None:
        intelligence = self._intelligence(
            domain="sports_transfers",
            visual_strategy="editorial_photo",
            prompt_main="team signing at stadium with group of players and staff",
            negative_prompt="chaotic crowd, crowded scenes",
        )

        plan = compose_prompt_plan(intelligence, base_negative_prompt="blurry", variants=1)
        lowered = plan.negative_prompt.lower()

        self.assertNotIn("chaotic crowd", lowered)
        self.assertNotIn("crowded scenes", lowered)
        self.assertIn("badge", lowered)
        self.assertIn("extra fingers", lowered)

    def test_disaster_scene_allows_documentary_wide_group_context(self) -> None:
        intelligence = self._intelligence(
            domain="conflict_disaster_crisis",
            visual_strategy="documentary_wide",
            prompt_main="rescue team and evacuees in wide emergency response scene",
            negative_prompt="large group, chaotic crowd",
        )

        plan = compose_prompt_plan(intelligence, base_negative_prompt="blurry", variants=1)
        lowered = plan.negative_prompt.lower()

        self.assertNotIn("large group", lowered)
        self.assertNotIn("chaotic crowd", lowered)
        self.assertIn("watermark", lowered)


if __name__ == "__main__":
    unittest.main()
