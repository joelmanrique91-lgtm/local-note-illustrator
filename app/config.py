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
    "preview_fast": InferencePreset(
        name="preview_fast",
        steps=12,
        guidance_scale=6.0,
        width=512,
        height=512,
        default_num_images=1,
    ),
    "dev_fast": InferencePreset(
        name="dev_fast",
        steps=20,
        guidance_scale=6.5,
        width=640,
        height=640,
        default_num_images=1,
    ),
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
    jpeg_quality: int
    jpeg_subsampling: int
    value_sources: dict[str, str]


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

    def _source(name: str) -> str:
        return "env" if os.getenv(name) is not None else "default"

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
    jpeg_quality = max(1, min(_env_int("JPEG_QUALITY", 90), 100))
    jpeg_subsampling = max(0, min(_env_int("JPEG_SUBSAMPLING", 0), 2))

    value_sources = {
        "model_id": _source("MODEL_ID"),
        "default_negative_prompt": _source("DEFAULT_NEGATIVE_PROMPT"),
        "default_num_images": _source("DEFAULT_NUM_IMAGES"),
        "default_steps": _source("DEFAULT_STEPS"),
        "default_guidance_scale": _source("DEFAULT_GUIDANCE_SCALE"),
        "default_width": _source("DEFAULT_WIDTH"),
        "default_height": _source("DEFAULT_HEIGHT"),
        "output_format": _source("OUTPUT_FORMAT"),
        "log_dir": _source("LOG_DIR"),
        "app_log_file": _source("APP_LOG_FILE"),
        "force_cpu": _source("FORCE_CPU"),
        "default_preset": _source("DEFAULT_PRESET"),
        "openai_enable": _source("OPENAI_ENABLE"),
        "openai_api_key": _source("OPENAI_API_KEY"),
        "openai_model": _source("OPENAI_MODEL"),
        "openai_timeout_seconds": _source("OPENAI_TIMEOUT_SECONDS"),
        "openai_max_retries": _source("OPENAI_MAX_RETRIES"),
        "openai_prompt_intelligence_mode": _source("OPENAI_PROMPT_INTELLIGENCE_MODE"),
        "openai_max_input_chars": _source("OPENAI_MAX_INPUT_CHARS"),
        "openai_strict_schema": _source("OPENAI_STRICT_SCHEMA"),
        "jpeg_quality": _source("JPEG_QUALITY"),
        "jpeg_subsampling": _source("JPEG_SUBSAMPLING"),
    }

    return AppConfig(
        model_id=os.getenv("MODEL_ID", "SG161222/RealVisXL_V5.0"),
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
        jpeg_quality=jpeg_quality,
        jpeg_subsampling=jpeg_subsampling,
        value_sources=value_sources,
    )
