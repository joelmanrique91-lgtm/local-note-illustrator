from __future__ import annotations

import json
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
            "Return strict JSON only. No markdown. You must infer domain, visual strategy, risk, "
            "main prompt, variants, negative prompt, composition notes and style notes. "
            "Prefer realistic editorial visuals and safe SDXL compositions. "
            "Avoid impossible scenes, avoid extreme close-up faces when risk is high, avoid malformed anatomy, "
            "avoid plastic skin, avoid chaotic crowds unless truly required by context. "
            "For sports transfer/club/player news, strongly prefer realistic sports editorial coverage with "
            "human presence and training/signing/stadium context. Never produce badges, crests, shields, poster layouts, "
            "infographic boards, visible typography, fake text, jersey logo close-ups, or collage of club symbols."
        )

    def _build_user_prompt(self, context: str, variants: int) -> str:
        return (
            "Analyze this news/document context and produce JSON with exact keys: "
            "domain, visual_strategy, human_closeup_risk, avoid_close_ups, prompt_main, prompt_variants, "
            "negative_prompt, composition_notes, style_notes, confidence. "
            f"prompt_variants should contain up to {max(1, min(variants, 2))} useful variants. "
            "visual_strategy must be one of: editorial_photo, conceptual, infographic_like, "
            "industrial, institutional, documentary_wide. "
            "human_closeup_risk in range 0..10 and confidence in range 0..1. "
            "Use concise English prompts optimized for SDXL local generation. "
            "If context is sports transfer/club/player domain, choose editorial_photo or documentary_wide unless very strong statistical-document evidence. "
            "In sports domain include explicit no-text and no-badge restrictions. "
            f"Document context:\n{context}"
        )

    def _validate_payload(self, payload: dict[str, Any]) -> PromptIntelligenceResult:
        required = {
            "domain": str,
            "visual_strategy": str,
            "human_closeup_risk": int,
            "avoid_close_ups": bool,
            "prompt_main": str,
            "prompt_variants": list,
            "negative_prompt": str,
            "composition_notes": str,
            "style_notes": str,
            "confidence": (int, float),
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

        prompt_main = payload["prompt_main"].strip()
        if not prompt_main:
            raise LlmAssistantSchemaError("prompt_main vacío")

        risk = max(0, min(int(payload["human_closeup_risk"]), 10))
        confidence = float(payload["confidence"])
        confidence = max(0.0, min(confidence, 1.0))

        variants_raw = payload.get("prompt_variants", [])
        variants = [str(item).strip() for item in variants_raw if str(item).strip()]

        return PromptIntelligenceResult(
            source="openai",
            domain=str(payload["domain"]).strip() or "technical_generic",
            visual_strategy=strategy,
            human_closeup_risk=risk,
            avoid_close_ups=bool(payload["avoid_close_ups"]),
            prompt_main=prompt_main,
            prompt_variants=variants,
            negative_prompt=str(payload["negative_prompt"]).strip(),
            composition_notes=str(payload["composition_notes"]).strip(),
            style_notes=str(payload["style_notes"]).strip(),
            confidence=confidence,
            fallback_reason=None,
            raw_schema_version="openai.v1",
        )
