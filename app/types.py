from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional
from uuid import uuid4


PromptSource = Literal["openai", "local_fallback"]


@dataclass(frozen=True)
class PromptIntelligenceResult:
    source: PromptSource
    domain: str
    visual_strategy: str
    human_closeup_risk: int
    avoid_close_ups: bool
    prompt_main: str
    prompt_variants: list[str]
    negative_prompt: str
    composition_notes: str
    style_notes: str
    confidence: float
    fallback_reason: Optional[str]
    raw_schema_version: str
    strategy_adjustment_reason: Optional[str] = None
    semantic_adjustment_reason: Optional[str] = None
    semantic_validation_status: Optional[str] = None


@dataclass(frozen=True)
class PromptPlan:
    positive_prompts: list[str]
    negative_prompt: str
    strategy_effective: str
    domain: str
    source: PromptSource
    strategy_adjustment_reason: Optional[str] = None
    semantic_adjustment_reason: Optional[str] = None
    semantic_validation_status: Optional[str] = None


@dataclass
class ImageManifest:
    image_index: int
    output_path: str


@dataclass
class DocumentManifest:
    document_path: str
    source: PromptSource
    strategy_override: str
    strategy_suggested: str
    strategy_effective: str
    domain: str
    preset: str
    seed: Optional[int]
    width: int
    height: int
    steps: int
    guidance_scale: float
    outputs: list[ImageManifest] = field(default_factory=list)
    error: Optional[str] = None
    openai_status: Optional[str] = None
    fallback_reason: Optional[str] = None
    openai_model: Optional[str] = None
    prompt_source: Optional[str] = None
    strategy_adjustment_reason: Optional[str] = None
    semantic_adjustment_reason: Optional[str] = None
    semantic_validation_status: Optional[str] = None
    final_positive_prompt: Optional[str] = None
    final_negative_prompt: Optional[str] = None


@dataclass
class RunManifest:
    run_id: str
    started_at: str
    finished_at: Optional[str]
    status: str
    selected_root_folder: str
    include_subfolders: bool
    images_per_document: int
    config_snapshot: dict[str, object]
    documents: list[DocumentManifest] = field(default_factory=list)

    @classmethod
    def create(
        cls,
        selected_root_folder: Path,
        include_subfolders: bool,
        images_per_document: int,
        config_snapshot: dict[str, object],
    ) -> "RunManifest":
        return cls(
            run_id=uuid4().hex,
            started_at=datetime.now().isoformat(timespec="seconds"),
            finished_at=None,
            status="running",
            selected_root_folder=str(selected_root_folder),
            include_subfolders=include_subfolders,
            images_per_document=images_per_document,
            config_snapshot=config_snapshot,
        )
