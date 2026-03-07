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
        return ["documento", "escena realista"]
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

    style_direction = (
        "photojournalistic scene, professional photography, realistic lighting, "
        "detailed composition, natural skin tones, editorial image, "
        "realistic environment, high detail"
    )

    base = (
        f"{style_direction}. "
        f"Main subject based on: {', '.join(keywords[:4])}. "
        f"Context: {summary}"
    )

    prompts = [base]
    if variants == 2:
        prompts.append(
            f"{style_direction}. "
            f"Alternative editorial framing focused on {', '.join(keywords[1:5])}. "
            "Balanced colors, documentary realism, natural perspective."
        )

    return PromptPack(positive_prompts=prompts, negative_prompt=negative_prompt)
