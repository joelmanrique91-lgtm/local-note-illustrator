from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from app.config import AppConfig
from app.prompt_builder import build_document_context
from app.types import PromptIntelligenceResult

ALLOWED_STRATEGIES = {
    "editorial_photo",
    "conceptual",
    "infographic_like",
    "industrial",
    "institutional",
    "documentary_wide",
}

CONCRETE_SETTING_HINTS = {
    "room",
    "table",
    "office",
    "podium",
    "conference",
    "briefing",
    "hall",
    "street",
    "airport",
    "stadium",
    "building",
    "meeting",
    "corridor",
    "lab",
}

ABSTRACT_ACTION_TERMS = {
    "debating policy",
    "discussing geopolitics",
    "representing diplomacy",
    "symbolizing",
    "analyzing",
    "reflecting",
}

IDENTITY_FORCE_HINTS = {
    "exact likeness",
    "photorealistic portrait of",
    "exact face",
    "identical face",
}

FLAG_OVERLOAD_HINTS = {"flag", "flags", "banner", "emblem", "crest"}
POLITICAL_DOMAINS = {"political_institutional", "political_news"}
ABSTRACT_POLITICAL_SETTING_HINTS = {
    "policy environment",
    "institutional significance",
    "political climate",
    "geopolitical tension",
    "diplomatic context",
}


class LlmAssistantError(RuntimeError):
    error_type = "unexpected"


class LlmAssistantConfigError(LlmAssistantError):
    error_type = "config"


class LlmAssistantTimeoutError(LlmAssistantError):
    error_type = "timeout"


class LlmAssistantNetworkError(LlmAssistantError):
    error_type = "network"


class LlmAssistantAuthError(LlmAssistantError):
    error_type = "auth"


class LlmAssistantSchemaError(LlmAssistantError):
    error_type = "schema_invalid"


@dataclass(frozen=True)
class LlmResponseEnvelope:
    result: PromptIntelligenceResult
    openai_status: str


class OpenAIPromptAssistant:
    def __init__(self, config: AppConfig, logger):
        self.config = config
        self.logger = logger

    def generate_prompt_intelligence(self, text: str, variants: int = 1) -> PromptIntelligenceResult:
        if not self.config.openai_enable:
            raise LlmAssistantConfigError("OPENAI_ENABLE=false")
        if not self.config.openai_api_key.strip():
            raise LlmAssistantConfigError("OPENAI_API_KEY faltante")

        payload = self._call_openai(text=text, variants=variants)
        return self._validate_payload(payload)

    def _call_openai(self, text: str, variants: int) -> dict[str, Any]:
        try:
            from openai import APITimeoutError, AuthenticationError, OpenAI
            from openai import APIConnectionError  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise LlmAssistantConfigError(f"SDK OpenAI no disponible: {exc}") from exc

        client = OpenAI(api_key=self.config.openai_api_key, timeout=self.config.openai_timeout_seconds)

        context = build_document_context(text, max_chars=self.config.openai_max_input_chars)
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(context=context, variants=variants)

        last_exc: Exception | None = None
        attempts = max(1, self.config.openai_max_retries + 1)
        for attempt in range(1, attempts + 1):
            try:
                response = client.chat.completions.create(
                    model=self.config.openai_model,
                    temperature=0.2,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    response_format={"type": "json_object"},
                )
                content = response.choices[0].message.content if response.choices else None
                if not content:
                    raise LlmAssistantSchemaError("OpenAI no devolvió contenido")
                try:
                    data = json.loads(content)
                except json.JSONDecodeError as exc:
                    raise LlmAssistantSchemaError("OpenAI devolvió JSON inválido") from exc
                return data
            except APITimeoutError as exc:
                last_exc = exc
                self.logger.warning("OpenAI timeout (intento %s/%s)", attempt, attempts)
            except APIConnectionError as exc:
                last_exc = exc
                self.logger.warning("OpenAI network error (intento %s/%s)", attempt, attempts)
            except AuthenticationError as exc:
                raise LlmAssistantAuthError("OpenAI auth error") from exc
            except LlmAssistantSchemaError:
                raise
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                self.logger.warning("OpenAI error inesperado (intento %s/%s): %s", attempt, attempts, exc)

        if last_exc is None:
            raise LlmAssistantError("OpenAI error desconocido")

        msg = str(last_exc).lower()
        if "timeout" in msg:
            raise LlmAssistantTimeoutError(str(last_exc)) from last_exc
        if "connection" in msg or "network" in msg:
            raise LlmAssistantNetworkError(str(last_exc)) from last_exc
        raise LlmAssistantError(str(last_exc)) from last_exc

    def _build_system_prompt(self) -> str:
        return (
            "You are a prompt-intelligence assistant for local SDXL editorial rendering. "
            "Return strict JSON only. No markdown. "
            "Return concrete render-ready scene fields, not editorial prose. "
            "Prefer one main subject plus supporting context, realistic action, and physically plausible staging. "
            "Avoid exact likeness claims for real people and avoid overloaded scenes with many leaders/flags/symbols. "
            "For political or diplomatic topics, prefer robust compositions such as official delegation meeting, "
            "senior officials at conference table, podium briefing, handshake, or delegation arrival."
        )

    def _build_user_prompt(self, context: str, variants: int) -> str:
        return (
            "Analyze this document and produce JSON with these exact keys: "
            "domain, visual_strategy, human_closeup_risk, confidence, "
            "primary_subject, secondary_subjects, primary_action, setting, visible_objects, framing, mood, realism_notes, "
            "avoid_close_ups, avoid_identity_claims, avoid_multi_person_overload, "
            "prompt_variants, negative_prompt. "
            f"prompt_variants should contain up to {max(1, min(variants, 2))} concise renderable variants. "
            "secondary_subjects and visible_objects must be arrays of short strings. "
            "visual_strategy must be one of: editorial_photo, conceptual, infographic_like, "
            "industrial, institutional, documentary_wide. "
            "human_closeup_risk in range 0..10 and confidence in range 0..1. "
            "Avoid abstract political language and avoid impossible crowd complexity. "
            f"Document context:\n{context}"
        )

    def _coerce_string_list(self, value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        clean: list[str] = []
        for item in value:
            text = str(item).strip()
            if text:
                clean.append(text)
        return clean

    def _semantic_check_render_ready(self, primary_subject: str, primary_action: str, setting: str) -> None:
        if not primary_subject:
            raise LlmAssistantSchemaError("render_ready_invalid:primary_subject_missing")
        if not primary_action:
            raise LlmAssistantSchemaError("render_ready_invalid:primary_action_missing")
        if primary_action.lower() in ABSTRACT_ACTION_TERMS:
            raise LlmAssistantSchemaError("render_ready_invalid:abstract_action")
        lowered_setting = setting.lower()
        if not setting or not any(token in lowered_setting for token in CONCRETE_SETTING_HINTS):
            raise LlmAssistantSchemaError("render_ready_invalid:non_concrete_setting")

    def _identity_force_detected(self, *fields: str) -> bool:
        merged = " ".join(fields).lower()
        return any(hint in merged for hint in IDENTITY_FORCE_HINTS)

    def _contains_multi_named_people(self, text: str) -> bool:
        matches = re.findall(r"\b[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+\b", text)
        return len(matches) >= 2

    def _is_abstract_political_setting(self, setting: str) -> bool:
        lowered = setting.lower().strip()
        if not lowered:
            return True
        return any(hint in lowered for hint in ABSTRACT_POLITICAL_SETTING_HINTS)

    def _simplify_political_payload(
        self,
        domain: str,
        primary_subject: str,
        secondary_subjects: list[str],
        primary_action: str,
        setting: str,
        visible_objects: list[str],
        framing: str,
        mood: str,
        realism_notes: str,
    ) -> tuple[str, list[str], str, str, list[str], str, str, str, bool, str | None]:
        if domain not in POLITICAL_DOMAINS:
            return (
                primary_subject,
                secondary_subjects,
                primary_action,
                setting,
                visible_objects,
                framing,
                mood,
                realism_notes,
                False,
                None,
            )

        simplification_reason: str | None = None
        simplified = False

        if len(secondary_subjects) > 2:
            secondary_subjects = secondary_subjects[:2]
            simplified = True
            simplification_reason = "political_news_simplified:secondary_subjects_capped"

        flag_objects = [item for item in visible_objects if any(h in item.lower() for h in FLAG_OVERLOAD_HINTS)]
        if len(flag_objects) > 2 or len(visible_objects) > 6:
            visible_objects = visible_objects[:4]
            simplified = True
            simplification_reason = simplification_reason or "political_news_simplified:symbol_overload_reduced"

        if self._contains_multi_named_people(primary_subject):
            primary_subject = "senior government delegation"
            simplified = True
            simplification_reason = simplification_reason or "political_news_simplified:identity_generalized"

        if self._is_abstract_political_setting(setting):
            setting = "official conference room"
            simplified = True
            simplification_reason = simplification_reason or "political_news_simplified:setting_hardened"

        framing = framing or "medium-wide framing"
        mood = mood or "formal and neutral"
        realism_notes = realism_notes or "documentary realism, no exact likeness"

        return (
            primary_subject,
            secondary_subjects,
            primary_action,
            setting,
            visible_objects,
            framing,
            mood,
            realism_notes,
            simplified,
            simplification_reason,
        )

    def _compose_prompt_main(
        self,
        primary_subject: str,
        secondary_subjects: list[str],
        primary_action: str,
        setting: str,
        visible_objects: list[str],
        framing: str,
        mood: str,
        realism_notes: str,
    ) -> str:
        segments = [primary_subject, primary_action, setting]
        if secondary_subjects:
            segments.append("supporting subjects: " + ", ".join(secondary_subjects[:2]))
        if visible_objects:
            segments.append("visible context: " + ", ".join(visible_objects[:4]))
        if framing:
            segments.append(framing)
        if mood:
            segments.append(mood)
        if realism_notes:
            segments.append(realism_notes)
        return ", ".join(segment.strip() for segment in segments if segment.strip())

    def _validate_payload(self, payload: dict[str, Any]) -> PromptIntelligenceResult:
        required = {
            "domain": str,
            "visual_strategy": str,
            "human_closeup_risk": int,
            "confidence": (int, float),
            "primary_subject": str,
            "secondary_subjects": list,
            "primary_action": str,
            "setting": str,
            "visible_objects": list,
            "framing": str,
            "mood": str,
            "realism_notes": str,
            "avoid_close_ups": bool,
            "avoid_identity_claims": bool,
            "avoid_multi_person_overload": bool,
            "prompt_variants": list,
            "negative_prompt": str,
        }
        for key, expected_type in required.items():
            if key not in payload:
                raise LlmAssistantSchemaError(f"Falta campo requerido: {key}")
            if not isinstance(payload[key], expected_type):
                raise LlmAssistantSchemaError(f"Tipo inválido para {key}")

        strategy = payload["visual_strategy"].strip()
        if strategy not in ALLOWED_STRATEGIES:
            if self.config.openai_strict_schema:
                raise LlmAssistantSchemaError(f"visual_strategy no soportada: {strategy}")
            strategy = "conceptual"

        domain = str(payload["domain"]).strip() or "technical_generic"
        primary_subject = str(payload["primary_subject"]).strip()
        secondary_subjects = self._coerce_string_list(payload.get("secondary_subjects"))
        primary_action = str(payload["primary_action"]).strip()
        setting = str(payload["setting"]).strip()
        visible_objects = self._coerce_string_list(payload.get("visible_objects"))
        framing = str(payload["framing"]).strip()
        mood = str(payload["mood"]).strip()
        realism_notes = str(payload["realism_notes"]).strip()

        self._semantic_check_render_ready(primary_subject, primary_action, setting)

        if len(secondary_subjects) > 4:
            raise LlmAssistantSchemaError("render_ready_invalid:too_many_secondary_subjects")
        if self._identity_force_detected(primary_subject, realism_notes, primary_action, setting):
            raise LlmAssistantSchemaError("render_ready_invalid:identity_likeness_forced")

        (
            primary_subject,
            secondary_subjects,
            primary_action,
            setting,
            visible_objects,
            framing,
            mood,
            realism_notes,
            simplified,
            simplification_reason,
        ) = self._simplify_political_payload(
            domain=domain,
            primary_subject=primary_subject,
            secondary_subjects=secondary_subjects,
            primary_action=primary_action,
            setting=setting,
            visible_objects=visible_objects,
            framing=framing,
            mood=mood,
            realism_notes=realism_notes,
        )

        prompt_main = self._compose_prompt_main(
            primary_subject=primary_subject,
            secondary_subjects=secondary_subjects,
            primary_action=primary_action,
            setting=setting,
            visible_objects=visible_objects,
            framing=framing,
            mood=mood,
            realism_notes=realism_notes,
        )

        risk = max(0, min(int(payload["human_closeup_risk"]), 10))
        confidence = float(payload["confidence"])
        confidence = max(0.0, min(confidence, 1.0))

        variants_raw = payload.get("prompt_variants", [])
        variants = [str(item).strip() for item in variants_raw if str(item).strip()]
        composition_notes = ", ".join(part for part in [setting, framing, "avoid close-up faces" if payload["avoid_close_ups"] else ""] if part)
        style_notes = ", ".join(part for part in [mood, realism_notes, "no exact likeness of real people" if payload["avoid_identity_claims"] else ""] if part)

        semantic_status = "simplified" if simplified else "validated"

        return PromptIntelligenceResult(
            source="openai",
            domain=domain,
            visual_strategy=strategy,
            human_closeup_risk=risk,
            avoid_close_ups=bool(payload["avoid_close_ups"]),
            prompt_main=prompt_main,
            prompt_variants=variants,
            negative_prompt=str(payload["negative_prompt"]).strip(),
            composition_notes=composition_notes,
            style_notes=style_notes,
            confidence=confidence,
            fallback_reason=None,
            raw_schema_version="openai.v2.render_ready",
            semantic_adjustment_reason=simplification_reason,
            semantic_validation_status=semantic_status,
        )
