from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class InferencePreset:
    name: str
    steps: int
    guidance_scale: float
    width: int
    height: int
    default_num_images: int


INFERENCE_PRESETS: dict[str, InferencePreset] = {
    "speed": InferencePreset(
        name="speed", steps=22, guidance_scale=6.0, width=832, height=832, default_num_images=1
    ),
    "balanced": InferencePreset(
        name="balanced",
        steps=30,
        guidance_scale=6.8,
        width=1024,
        height=1024,
        default_num_images=1,
    ),
    "quality": InferencePreset(
        name="quality", steps=40, guidance_scale=7.2, width=1152, height=1152, default_num_images=1
    ),
    "editorial_safe": InferencePreset(
        name="editorial_safe",
        steps=34,
        guidance_scale=6.2,
        width=1024,
        height=768,
        default_num_images=1,
    ),
}


@dataclass(frozen=True)
class AppConfig:
    model_id: str
    default_negative_prompt: str
    default_num_images: int
    default_steps: int
    default_guidance_scale: float
    default_width: int
    default_height: int
    output_format: str
    log_dir: Path
    app_log_file: str
    force_cpu: bool
    default_preset: str
    openai_enable: bool
    openai_api_key: str
    openai_model: str
    openai_timeout_seconds: int
    openai_max_retries: int
    openai_prompt_intelligence_mode: str
    openai_max_input_chars: int
    openai_strict_schema: bool


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _normalize_dimension(value: int) -> int:
    value = max(256, value)
    return (value // 64) * 64


def get_preset(name: str) -> InferencePreset:
    return INFERENCE_PRESETS.get(name, INFERENCE_PRESETS["balanced"])


def load_config() -> AppConfig:
    load_dotenv(override=False)

    output_format = os.getenv("OUTPUT_FORMAT", "jpg").lower()
    if output_format not in {"png", "jpg", "jpeg"}:
        output_format = "jpg"

    log_dir = Path(os.getenv("LOG_DIR", "logs"))

    preset_name = os.getenv("DEFAULT_PRESET", "balanced")
    preset = get_preset(preset_name)

    default_num_images = max(1, min(_env_int("DEFAULT_NUM_IMAGES", preset.default_num_images), 2))
    default_steps = max(10, _env_int("DEFAULT_STEPS", preset.steps))
    default_guidance_scale = max(1.0, _env_float("DEFAULT_GUIDANCE_SCALE", preset.guidance_scale))
    default_width = _normalize_dimension(_env_int("DEFAULT_WIDTH", preset.width))
    default_height = _normalize_dimension(_env_int("DEFAULT_HEIGHT", preset.height))
    openai_timeout_seconds = max(5, _env_int("OPENAI_TIMEOUT_SECONDS", 20))
    openai_max_retries = max(0, _env_int("OPENAI_MAX_RETRIES", 1))
    openai_mode = os.getenv(
        "OPENAI_PROMPT_INTELLIGENCE_MODE", "required_with_safety_fallback"
    ).strip()
    openai_max_input_chars = max(1000, _env_int("OPENAI_MAX_INPUT_CHARS", 8000))

    return AppConfig(
        model_id=os.getenv("MODEL_ID", "stabilityai/stable-diffusion-xl-base-1.0"),
        default_negative_prompt=os.getenv(
            "DEFAULT_NEGATIVE_PROMPT",
            "distorted face, malformed anatomy, deformed hands, extra fingers, extra limbs, "
            "duplicate face, crossed eyes, asymmetrical eyes, blurry details, low quality, "
            "plastic skin, uncanny, oversaturated colors, artificial lighting, unreadable text, "
            "watermark, logo",
        ),
        default_num_images=default_num_images,
        default_steps=default_steps,
        default_guidance_scale=default_guidance_scale,
        default_width=default_width,
        default_height=default_height,
        output_format=output_format,
        log_dir=log_dir,
        app_log_file=os.getenv("APP_LOG_FILE", "app.log"),
        force_cpu=_env_bool("FORCE_CPU", False),
        default_preset=preset.name,
        openai_enable=_env_bool("OPENAI_ENABLE", True),
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        openai_timeout_seconds=openai_timeout_seconds,
        openai_max_retries=openai_max_retries,
        openai_prompt_intelligence_mode=openai_mode,
        openai_max_input_chars=openai_max_input_chars,
        openai_strict_schema=_env_bool("OPENAI_STRICT_SCHEMA", True),
    )
