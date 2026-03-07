from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass


STOPWORDS = {
    "de",
    "la",
    "el",
    "en",
    "y",
    "a",
    "los",
    "las",
    "del",
    "con",
    "por",
    "para",
    "un",
    "una",
    "que",
    "se",
    "al",
    "es",
}


@dataclass
class PromptPack:
    positive_prompts: list[str]
    negative_prompt: str



def _extract_keywords(text: str, n: int = 6) -> list[str]:
    words = re.findall(r"[a-zA-ZáéíóúÁÉÍÓÚñÑ]{4,}", text.lower())
    words = [w for w in words if w not in STOPWORDS]
    if not words:
        return ["escena conceptual", "ilustración"]
    freq = Counter(words)
    return [w for w, _ in freq.most_common(n)]



def build_prompts(
    text: str,
    negative_prompt: str,
    variants: int = 1,
) -> PromptPack:
    variants = max(1, min(variants, 2))
    summary = text[:800] if text else "nota breve"
    keywords = _extract_keywords(text)

    base = (
        "high quality digital illustration, cinematic lighting, detailed, "
        f"main concept about {', '.join(keywords[:4])}. "
        f"Context: {summary}"
    )

    prompts = [base]
    if variants == 2:
        prompts.append(
            "editorial concept art, clean composition, storytelling scene, "
            f"focus on {', '.join(keywords[1:5])}, professional color grading"
        )

    return PromptPack(positive_prompts=prompts, negative_prompt=negative_prompt)
