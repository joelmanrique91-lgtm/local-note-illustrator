from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import torch
from diffusers import StableDiffusionXLPipeline

from app.config import AppConfig
from app.utils import safe_slug


class ImageGenerator:
    CUDA_FALLBACK_PATTERNS = (
        "no kernel image is available for execution on the device",
        "cuda error",
        "cuda initialization",
        "invalid device function",
        "device-side assert triggered",
        "sm_",
    )

    def __init__(self, config: AppConfig, logger):
        self.config = config
        self.logger = logger
        self.pipeline: Optional[StableDiffusionXLPipeline] = None
        self.device = "cpu"
        self.dtype = torch.float32
        self._choose_initial_device()

    def _is_cuda_related_error(self, exc: Exception) -> bool:
        msg = str(exc).lower()
        return any(pattern in msg for pattern in self.CUDA_FALLBACK_PATTERNS)

    def _choose_initial_device(self) -> None:
        cuda_available = torch.cuda.is_available()
        gpu_name = "N/A"
        if cuda_available:
            try:
                gpu_name = torch.cuda.get_device_name(0)
            except Exception as exc:  # noqa: BLE001
                gpu_name = f"error leyendo GPU: {exc}"

        self.logger.info("torch version: %s", torch.__version__)
        self.logger.info("torch.cuda.is_available(): %s", cuda_available)
        self.logger.info("GPU detectada: %s", gpu_name)

        if self.config.force_cpu:
            self.device = "cpu"
            self.dtype = torch.float32
            self.logger.info("FORCE_CPU=true -> CUDA deshabilitado manualmente.")
            return

        if not cuda_available:
            self.device = "cpu"
            self.dtype = torch.float32
            self.logger.info("CUDA no disponible; se usará CPU.")
            return

        try:
            probe = torch.zeros(1, device="cuda")
            _ = (probe + 1).item()
            torch.cuda.synchronize()
            self.device = "cuda"
            self.dtype = torch.float16
            self.logger.info("Validación CUDA exitosa; se usará GPU.")
        except Exception as exc:  # noqa: BLE001
            self.device = "cpu"
            self.dtype = torch.float32
            self.logger.warning("CUDA no usable; fallback a CPU. Motivo: %s", exc)

    def _reset_pipeline(self) -> None:
        self.pipeline = None
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                pass

    def _initialize_pipeline_on_device(self, device: str) -> StableDiffusionXLPipeline:
        dtype = torch.float16 if device == "cuda" else torch.float32
        self.logger.info(
            "Cargando pipeline Diffusers (model=%s, device=%s, dtype=%s)",
            self.config.model_id,
            device,
            dtype,
        )
        start = time.perf_counter()
        pipe = StableDiffusionXLPipeline.from_pretrained(
            self.config.model_id,
            torch_dtype=dtype,
        )
        pipe = pipe.to(device)
        if device == "cuda":
            pipe.enable_attention_slicing()
        self.logger.info("Pipeline cargado en %.2fs (%s)", time.perf_counter() - start, device)
        return pipe

    def _load_pipeline(self) -> None:
        if self.pipeline is not None:
            return

        try:
            self.pipeline = self._initialize_pipeline_on_device(self.device)
            self.dtype = torch.float16 if self.device == "cuda" else torch.float32
            self.logger.info("Pipeline final en dispositivo: %s", self.device)
        except Exception as exc:  # noqa: BLE001
            if self.device == "cuda" and self._is_cuda_related_error(exc):
                self.logger.warning(
                    "Falló inicialización CUDA del pipeline; fallback a CPU. Motivo: %s", exc
                )
                self._reset_pipeline()
                self.device = "cpu"
                self.dtype = torch.float32
                self.pipeline = self._initialize_pipeline_on_device("cpu")
                self.logger.info("Pipeline final en dispositivo: cpu")
            else:
                raise

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

    def _run_generation(
        self,
        positive_prompt: str,
        negative_prompt: str,
        steps: int,
        guidance_scale: float,
        width: int,
        height: int,
        seed: Optional[int],
    ):
        assert self.pipeline is not None
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        return self.pipeline(
            prompt=positive_prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            generator=generator,
        )

    def _log_generation_start(
        self,
        width: int,
        height: int,
        steps: int,
        guidance_scale: float,
        negative_prompt: str,
        seed: Optional[int],
        preset_name: str,
        strategy_name: str,
    ) -> None:
        self.logger.info(
            "Inicio generación | model_id=%s | preset=%s | strategy=%s | device=%s | width=%s | height=%s | steps=%s | guidance_scale=%s | seed=%s | negative_prompt=%s | force_cpu=%s",
            self.config.model_id,
            preset_name,
            strategy_name,
            self.device,
            width,
            height,
            steps,
            guidance_scale,
            seed if seed is not None else "aleatoria",
            negative_prompt,
            self.config.force_cpu,
        )

    def get_runtime_parameters(
        self,
        negative_prompt: Optional[str] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> dict[str, object]:
        return {
            "model_id": self.config.model_id,
            "device": self.device,
            "force_cpu": self.config.force_cpu,
            "width": width if width is not None else self.config.default_width,
            "height": height if height is not None else self.config.default_height,
            "steps": steps if steps is not None else self.config.default_steps,
            "guidance_scale": (
                guidance_scale
                if guidance_scale is not None
                else self.config.default_guidance_scale
            ),
            "seed": seed,
            "negative_prompt": (
                negative_prompt
                if negative_prompt is not None
                else self.config.default_negative_prompt
            ),
        }

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
        preset_name: str = "custom",
        strategy_name: str = "auto",
    ) -> Path:
        self._load_pipeline()
        assert self.pipeline is not None

        steps = steps or self.config.default_steps
        guidance_scale = guidance_scale or self.config.default_guidance_scale
        width = width or self.config.default_width
        height = height or self.config.default_height

        self._log_generation_start(
            width=width,
            height=height,
            steps=steps,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt,
            seed=seed,
            preset_name=preset_name,
            strategy_name=strategy_name,
        )
        self.logger.info(
            "Generando imagen para '%s' (idx=%s) | steps=%s guidance=%.2f size=%sx%s",
            docx_path.name,
            image_index,
            steps,
            guidance_scale,
            width,
            height,
        )
        self.logger.info("Prompt used: %s", positive_prompt)
        start = time.perf_counter()

        try:
            result = self._run_generation(
                positive_prompt=positive_prompt,
                negative_prompt=negative_prompt,
                steps=steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height,
                seed=seed,
            )
        except Exception as exc:  # noqa: BLE001
            if self.device == "cuda" and self._is_cuda_related_error(exc):
                self.logger.warning(
                    "Fallo durante generación en CUDA; fallback a CPU y reintento. Motivo: %s",
                    exc,
                )
                self._reset_pipeline()
                self.device = "cpu"
                self.dtype = torch.float32
                self.pipeline = self._initialize_pipeline_on_device("cpu")
                result = self._run_generation(
                    positive_prompt=positive_prompt,
                    negative_prompt=negative_prompt,
                    steps=steps,
                    guidance_scale=guidance_scale,
                    width=width,
                    height=height,
                    seed=seed,
                )
            else:
                raise

        image = result.images[0]
        output_path = self._next_output_path(docx_path, image_index, self.config.output_format)
        if self.config.output_format in {"jpg", "jpeg"}:
            image = image.convert("RGB")
            image.save(output_path, format="JPEG", quality=90, subsampling=0)
        else:
            image.save(output_path)
        elapsed = time.perf_counter() - start
        self.logger.info("Imagen guardada: %s (%.2fs)", output_path, elapsed)
        return output_path
