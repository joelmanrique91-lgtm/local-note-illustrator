from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import torch
from diffusers import StableDiffusionPipeline

from app.config import AppConfig
from app.utils import safe_slug


class ImageGenerator:
    def __init__(self, config: AppConfig, logger):
        self.config = config
        self.logger = logger
        self.pipeline: Optional[StableDiffusionPipeline] = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32

    def _load_pipeline(self) -> None:
        if self.pipeline is not None:
            return

        self.logger.info(
            "Cargando pipeline Diffusers (model=%s, device=%s)",
            self.config.model_id,
            self.device,
        )
        start = time.perf_counter()
        pipe = StableDiffusionPipeline.from_pretrained(
            self.config.model_id,
            torch_dtype=self.dtype,
            safety_checker=None,
        )
        pipe = pipe.to(self.device)
        if self.device == "cuda":
            pipe.enable_attention_slicing()
        self.pipeline = pipe
        self.logger.info("Pipeline cargado en %.2fs", time.perf_counter() - start)

    def _next_output_path(self, docx_path: Path, index: int, fmt: str) -> Path:
        base = safe_slug(docx_path.stem)
        candidate = docx_path.parent / f"{base}_img_{index:02d}.{fmt}"
        if not candidate.exists():
            return candidate

        suffix = 1
        while True:
            candidate = docx_path.parent / f"{base}_img_{index:02d}_{suffix:02d}.{fmt}"
            if not candidate.exists():
                return candidate
            suffix += 1

    def generate(
        self,
        docx_path: Path,
        positive_prompt: str,
        negative_prompt: str,
        image_index: int,
        steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> Path:
        self._load_pipeline()
        assert self.pipeline is not None

        steps = steps or self.config.default_steps
        guidance_scale = guidance_scale or self.config.default_guidance_scale
        width = width or self.config.default_width
        height = height or self.config.default_height
        fmt = self.config.output_format

        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        start = time.perf_counter()
        result = self.pipeline(
            prompt=positive_prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            generator=generator,
        )
        image = result.images[0]
        output = self._next_output_path(docx_path, image_index, fmt)
        image.save(output)
        elapsed = time.perf_counter() - start
        self.logger.info("Imagen generada: %s (%.2fs)", output, elapsed)
        return output
