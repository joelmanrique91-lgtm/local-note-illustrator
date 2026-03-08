from __future__ import annotations

from dataclasses import dataclass

from app.config import AppConfig
from app.llm_assistant import (
    LlmAssistantAuthError,
    LlmAssistantConfigError,
    LlmAssistantError,
    LlmAssistantNetworkError,
    LlmAssistantSchemaError,
    LlmAssistantTimeoutError,
    OpenAIPromptAssistant,
)
from app.prompt_builder import build_local_fallback_result, compose_prompt_plan
from app.strategy import classify_domain
from app.types import PromptIntelligenceResult, PromptPlan


@dataclass(frozen=True)
class PromptResolution:
    intelligence: PromptIntelligenceResult
    prompt_plan: PromptPlan
    openai_status: str


def _apply_manual_override(intelligence: PromptIntelligenceResult, strategy_override: str) -> PromptIntelligenceResult:
    if strategy_override == "auto":
        return intelligence

    return PromptIntelligenceResult(
        source=intelligence.source,
        domain=intelligence.domain,
        visual_strategy=strategy_override,
        human_closeup_risk=intelligence.human_closeup_risk,
        avoid_close_ups=intelligence.avoid_close_ups,
        prompt_main=intelligence.prompt_main,
        prompt_variants=intelligence.prompt_variants,
        negative_prompt=intelligence.negative_prompt,
        composition_notes=intelligence.composition_notes,
        style_notes=intelligence.style_notes,
        confidence=intelligence.confidence,
        fallback_reason=intelligence.fallback_reason,
        raw_schema_version=intelligence.raw_schema_version,
        strategy_adjustment_reason=intelligence.strategy_adjustment_reason,
        semantic_adjustment_reason=intelligence.semantic_adjustment_reason,
        semantic_validation_status=intelligence.semantic_validation_status,
    )


def _enforce_sports_visual_policy(intelligence: PromptIntelligenceResult, text: str, logger) -> PromptIntelligenceResult:
    domain = intelligence.domain
    if domain == "technical_generic":
        inferred = classify_domain(text)
        if inferred == "sports_transfers":
            domain = inferred

    if domain != "sports_transfers":
        return intelligence

    allowed = {"editorial_photo", "documentary_wide"}
    if intelligence.visual_strategy in allowed:
        return PromptIntelligenceResult(
            source=intelligence.source,
            domain=domain,
            visual_strategy=intelligence.visual_strategy,
            human_closeup_risk=intelligence.human_closeup_risk,
            avoid_close_ups=intelligence.avoid_close_ups,
            prompt_main=intelligence.prompt_main,
            prompt_variants=intelligence.prompt_variants,
            negative_prompt=intelligence.negative_prompt,
            composition_notes=intelligence.composition_notes,
            style_notes=intelligence.style_notes,
            confidence=intelligence.confidence,
            fallback_reason=intelligence.fallback_reason,
            raw_schema_version=intelligence.raw_schema_version,
            strategy_adjustment_reason=intelligence.strategy_adjustment_reason,
            semantic_adjustment_reason=intelligence.semantic_adjustment_reason,
            semantic_validation_status=intelligence.semantic_validation_status,
        )

    reason = f"domain_policy:sports_transfers_forced_from_{intelligence.visual_strategy}"
    logger.warning(
        "Policy adjustment: sports domain strategy corrected from %s to editorial_photo",
        intelligence.visual_strategy,
    )
    return PromptIntelligenceResult(
        source=intelligence.source,
        domain=domain,
        visual_strategy="editorial_photo",
        human_closeup_risk=max(intelligence.human_closeup_risk, 4),
        avoid_close_ups=True,
        prompt_main=intelligence.prompt_main,
        prompt_variants=intelligence.prompt_variants,
        negative_prompt=intelligence.negative_prompt,
        composition_notes=intelligence.composition_notes,
        style_notes=intelligence.style_notes,
        confidence=intelligence.confidence,
        fallback_reason=intelligence.fallback_reason,
        raw_schema_version=intelligence.raw_schema_version,
        strategy_adjustment_reason=reason,
        semantic_adjustment_reason=intelligence.semantic_adjustment_reason,
        semantic_validation_status=intelligence.semantic_validation_status,
    )


def _resolve_local_fallback(
    text: str,
    strategy_override: str,
    config: AppConfig,
    variants: int,
    fallback_reason: str,
    openai_status: str = "fallback",
) -> PromptResolution:
    intelligence = build_local_fallback_result(
        text=text,
        strategy_override=strategy_override,
        base_negative_prompt=config.default_negative_prompt,
        variants=variants,
        fallback_reason=fallback_reason,
    )
    intelligence = _enforce_sports_visual_policy(intelligence, text=text, logger=none_logger)
    plan = compose_prompt_plan(
        intelligence=intelligence,
        base_negative_prompt=config.default_negative_prompt,
        variants=variants,
    )
    return PromptResolution(intelligence=intelligence, prompt_plan=plan, openai_status=openai_status)


class _NoLogger:
    def warning(self, *_args, **_kwargs):
        return None


none_logger = _NoLogger()


def resolve_prompt_plan(
    text: str,
    strategy_override: str,
    config: AppConfig,
    logger,
    variants: int,
) -> PromptResolution:
    mode = config.openai_prompt_intelligence_mode

    if not config.openai_enable or mode == "disabled":
        logger.info("Prompt intelligence local: OPENAI deshabilitado o modo disabled")
        return _resolve_local_fallback(
            text=text,
            strategy_override=strategy_override,
            config=config,
            variants=variants,
            fallback_reason="openai_disabled",
            openai_status="skipped",
        )

    assistant = OpenAIPromptAssistant(config=config, logger=logger)
    try:
        intelligence = assistant.generate_prompt_intelligence(text=text, variants=variants)
        intelligence = _enforce_sports_visual_policy(intelligence, text=text, logger=logger)
        intelligence = _apply_manual_override(intelligence, strategy_override)
        plan = compose_prompt_plan(
            intelligence=intelligence,
            base_negative_prompt=config.default_negative_prompt,
            variants=variants,
        )
        return PromptResolution(intelligence=intelligence, prompt_plan=plan, openai_status="success")
    except LlmAssistantConfigError as exc:
        logger.warning("OpenAI config issue, fallback local: %s", exc)
        if mode == "required_strict":
            raise
        return _resolve_local_fallback(
            text=text,
            strategy_override=strategy_override,
            config=config,
            variants=variants,
            fallback_reason=f"config:{exc}",
        )
    except LlmAssistantTimeoutError as exc:
        logger.warning("OpenAI timeout, fallback local: %s", exc)
        return _resolve_local_fallback(
            text=text,
            strategy_override=strategy_override,
            config=config,
            variants=variants,
            fallback_reason=f"timeout:{exc}",
        )
    except LlmAssistantNetworkError as exc:
        logger.warning("OpenAI network, fallback local: %s", exc)
        return _resolve_local_fallback(
            text=text,
            strategy_override=strategy_override,
            config=config,
            variants=variants,
            fallback_reason=f"network:{exc}",
        )
    except LlmAssistantAuthError as exc:
        logger.warning("OpenAI auth, fallback local: %s", exc)
        if mode == "required_strict":
            raise
        return _resolve_local_fallback(
            text=text,
            strategy_override=strategy_override,
            config=config,
            variants=variants,
            fallback_reason=f"auth:{exc}",
        )
    except LlmAssistantSchemaError as exc:
        logger.warning("OpenAI schema invalid, fallback local: %s", exc)
        return _resolve_local_fallback(
            text=text,
            strategy_override=strategy_override,
            config=config,
            variants=variants,
            fallback_reason=f"schema_invalid:{exc}",
        )
    except LlmAssistantError as exc:
        logger.warning("OpenAI generic error, fallback local: %s", exc)
        return _resolve_local_fallback(
            text=text,
            strategy_override=strategy_override,
            config=config,
            variants=variants,
            fallback_reason=f"openai_error:{exc}",
        )
