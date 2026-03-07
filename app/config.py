from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


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


def load_config() -> AppConfig:
    load_dotenv(override=False)

    output_format = os.getenv("OUTPUT_FORMAT", "jpg").lower()
    if output_format not in {"png", "jpg", "jpeg"}:
        output_format = "jpg"

    log_dir = Path(os.getenv("LOG_DIR", "logs"))

    return AppConfig(
        model_id=os.getenv("MODEL_ID", "stabilityai/stable-diffusion-xl-base-1.0"),
        default_negative_prompt=os.getenv(
            "DEFAULT_NEGATIVE_PROMPT",
            "cartoon, anime, illustration, low quality, blurry, distorted face, bad anatomy, deformed hands, extra fingers, extra limbs, text, watermark, logo",
        ),
        default_num_images=max(1, min(_env_int("DEFAULT_NUM_IMAGES", 1), 2)),
        default_steps=max(10, _env_int("DEFAULT_STEPS", 30)),
        default_guidance_scale=max(1.0, _env_float("DEFAULT_GUIDANCE_SCALE", 7.5)),
        default_width=max(256, _env_int("DEFAULT_WIDTH", 1024)),
        default_height=max(256, _env_int("DEFAULT_HEIGHT", 1024)),
        output_format=output_format,
        log_dir=log_dir,
        app_log_file=os.getenv("APP_LOG_FILE", "app.log"),
        force_cpu=_env_bool("FORCE_CPU", False),
    )
