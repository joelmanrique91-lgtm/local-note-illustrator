from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal, Optional

from app.config import AppConfig, InferencePreset, get_preset

ValueSource = Literal["preset", "env", "default", "gui", "runtime", "fallback", "derived"]


@dataclass(frozen=True)
class RuntimeValue:
    value: Any
    source: ValueSource


@dataclass(frozen=True)
class RuntimeSnapshot:
    model_id: RuntimeValue
    pipeline_class: RuntimeValue
    device: RuntimeValue
    dtype: RuntimeValue
    cuda_fallback_triggered: RuntimeValue
    preset: RuntimeValue
    width: RuntimeValue
    height: RuntimeValue
    steps: RuntimeValue
    guidance_scale: RuntimeValue
    seed: RuntimeValue
    strategy_override: RuntimeValue
    output_format: RuntimeValue
    jpeg_quality: RuntimeValue
    jpeg_subsampling: RuntimeValue
    images_per_document: RuntimeValue
    openai_enable: RuntimeValue
    openai_mode: RuntimeValue
    openai_model: RuntimeValue

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DocumentRuntimeSnapshot:
    runtime: RuntimeSnapshot
    strategy_effective: RuntimeValue
    prompt_source: RuntimeValue
    openai_status: RuntimeValue
    fallback_reason: RuntimeValue

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ImageRuntimeSnapshot:
    output_path: RuntimeValue
    file_size_bytes: RuntimeValue
    device_at_generation: RuntimeValue
    dtype_at_generation: RuntimeValue
    cuda_fallback_triggered: RuntimeValue

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RuntimeResolver:
    def __init__(self, config: AppConfig):
        self.config = config

    @staticmethod
    def _seed_source(seed: Optional[int]) -> ValueSource:
        return "gui" if seed is not None else "runtime"

    @staticmethod
    def _fallback_source(triggered: bool) -> ValueSource:
        return "fallback" if triggered else "runtime"

    def resolve_run_runtime(
        self,
        preset_name: str,
        strategy_override: str,
        seed: Optional[int],
        images_per_document: int,
        backend_state: dict[str, Any],
    ) -> RuntimeSnapshot:
        preset: InferencePreset = get_preset(preset_name)

        return RuntimeSnapshot(
            model_id=RuntimeValue(self.config.model_id, self.config.value_sources.get("model_id", "default")),
            pipeline_class=RuntimeValue(backend_state.get("pipeline_class", "StableDiffusionXLPipeline"), "runtime"),
            device=RuntimeValue(backend_state.get("device", "cpu"), "runtime"),
            dtype=RuntimeValue(backend_state.get("dtype", "float32"), "runtime"),
            cuda_fallback_triggered=RuntimeValue(
                bool(backend_state.get("cuda_fallback_triggered", False)),
                self._fallback_source(bool(backend_state.get("cuda_fallback_triggered", False))),
            ),
            preset=RuntimeValue(preset.name, "gui"),
            width=RuntimeValue(preset.width, "preset"),
            height=RuntimeValue(preset.height, "preset"),
            steps=RuntimeValue(preset.steps, "preset"),
            guidance_scale=RuntimeValue(preset.guidance_scale, "preset"),
            seed=RuntimeValue(seed if seed is not None else "random", self._seed_source(seed)),
            strategy_override=RuntimeValue(strategy_override, "gui"),
            output_format=RuntimeValue(self.config.output_format, self.config.value_sources.get("output_format", "default")),
            jpeg_quality=RuntimeValue(self.config.jpeg_quality, self.config.value_sources.get("jpeg_quality", "default")),
            jpeg_subsampling=RuntimeValue(
                self.config.jpeg_subsampling,
                self.config.value_sources.get("jpeg_subsampling", "default"),
            ),
            images_per_document=RuntimeValue(images_per_document, "gui"),
            openai_enable=RuntimeValue(self.config.openai_enable, self.config.value_sources.get("openai_enable", "default")),
            openai_mode=RuntimeValue(
                self.config.openai_prompt_intelligence_mode,
                self.config.value_sources.get("openai_prompt_intelligence_mode", "default"),
            ),
            openai_model=RuntimeValue(self.config.openai_model, self.config.value_sources.get("openai_model", "default")),
        )

    def resolve_document_runtime(
        self,
        run_runtime: RuntimeSnapshot,
        strategy_effective: str,
        prompt_source: str,
        openai_status: str,
        fallback_reason: Optional[str],
    ) -> DocumentRuntimeSnapshot:
        fallback_source: ValueSource = "fallback" if fallback_reason else "runtime"
        return DocumentRuntimeSnapshot(
            runtime=run_runtime,
            strategy_effective=RuntimeValue(strategy_effective, "derived"),
            prompt_source=RuntimeValue(prompt_source, "runtime"),
            openai_status=RuntimeValue(openai_status, "runtime"),
            fallback_reason=RuntimeValue(fallback_reason or "none", fallback_source),
        )

    def resolve_image_runtime(
        self,
        output_path: str,
        file_size_bytes: int,
        device_at_generation: str,
        dtype_at_generation: str,
        cuda_fallback_triggered: bool,
    ) -> ImageRuntimeSnapshot:
        return ImageRuntimeSnapshot(
            output_path=RuntimeValue(output_path, "runtime"),
            file_size_bytes=RuntimeValue(file_size_bytes, "runtime"),
            device_at_generation=RuntimeValue(device_at_generation, "runtime"),
            dtype_at_generation=RuntimeValue(dtype_at_generation, "runtime"),
            cuda_fallback_triggered=RuntimeValue(
                cuda_fallback_triggered,
                self._fallback_source(cuda_fallback_triggered),
            ),
        )


def build_export_payload(
    runtime: RuntimeSnapshot,
    selected_root_folder: str,
    per_document: list[dict[str, Any]],
    torch_version: str,
    diffusers_version: str,
    generated_at: str,
) -> dict[str, Any]:
    return {
        "timestamp": generated_at,
        "runtime_effective": runtime.to_dict(),
        "selected_root_folder": selected_root_folder,
        "torch_version": torch_version,
        "diffusers_version": diffusers_version,
        "documents": per_document,
    }
