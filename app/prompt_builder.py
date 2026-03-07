from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from app.strategy import StrategyProfile, analyze_visual_strategy

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
    "the",
    "and",
    "for",
    "with",
    "from",
    "this",
    "that",
}

JUNK_KEYWORDS = {
    "reunion",
    "reunión",
    "convoca",
    "discute",
    "discutir",
    "analiza",
    "tema",
    "evento",
    "nota",
    "importante",
    "nueva",
    "sobre",
    "durante",
    "acerca",
    "meeting",
    "discuss",
    "analysis",
    "topic",
}

TAG_PREFIXES = ("tags:", "tag:", "etiquetas:")

VISUAL_STRATEGY_DIRECTIVES = {
    "editorial_photo": "editorial documentary photo, medium-wide framing, context before portrait, natural color grading",
    "conceptual": "clean conceptual editorial visual, symbolic composition, no close portraits, calm neutral palette",
    "infographic_like": "editorial infographic-like composition, data-driven visual language, screens, charts and institutional context",
    "industrial": "industrial editorial wide shot, operational infrastructure and machinery, no close human portrait",
    "institutional": "institutional press coverage, podium or meeting room, medium or wide shot, formal documentary look",
    "documentary_wide": "documentary wide scene, environmental context first, no close-up faces, realistic reportage",
}

NEGATIVE_GROUPS = {
    "anatomy": "deformed anatomy, malformed body, broken proportions, extra limbs",
    "hands_face": "deformed hands, extra fingers, duplicate face, asymmetrical eyes, crossed eyes",
    "artificial_look": "plastic skin, uncanny valley, oversaturated colors, fake cinematic glow, artificial render",
    "composition_noise": "chaotic crowd, extreme close-up portrait, cropped face, warped perspective",
    "artifacts": "blurry, low quality, jpeg artifacts, unreadable text, watermark, logo",
}


@dataclass
class PromptPack:
    positive_prompts: list[str]
    negative_prompt: str
    strategy_profile: StrategyProfile


@dataclass
class ArticleSections:
    title: str
    summary: str
    tags: str
    first_paragraph: str
    body: str


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip(" ,.;:-")


def _is_tag_line(line: str) -> bool:
    lowered = line.strip().lower()
    return any(lowered.startswith(prefix) for prefix in TAG_PREFIXES)


def _clean_tag_prefix(line: str) -> str:
    cleaned = re.sub(r"^(?:tags?|etiquetas)\s*:\s*", "", line.strip(), flags=re.IGNORECASE)
    return _normalize_text(cleaned)


def _unique_nonempty(parts: list[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for part in parts:
        normalized = _normalize_text(part)
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(normalized)
    return ordered


def _split_article_sections(text: str) -> ArticleSections:
    lines = [_normalize_text(line) for line in text.splitlines() if _normalize_text(line)]
    if not lines:
        return ArticleSections("nota periodística", "", "", "", "")

    title = lines[0]
    body_lines = lines[1:]

    tag_values = [_clean_tag_prefix(line) for line in body_lines if _is_tag_line(line)]
    tags = ", ".join(_unique_nonempty(tag_values))

    useful_body = [line for line in body_lines if not _is_tag_line(line)]
    summary = useful_body[0] if useful_body else title
    first_paragraph = useful_body[1] if len(useful_body) > 1 else summary

    body_lines_context = useful_body[1:7] if useful_body else []
    body = " ".join(body_lines_context)

    return ArticleSections(title, summary, tags, first_paragraph, body)


def _extract_keywords(text: str, n: int = 8) -> list[str]:
    lowered = text.lower()
    tokens = re.findall(r"[a-zA-ZáéíóúÁÉÍÓÚñÑ-]{4,}", lowered)
    filtered = [
        token
        for token in tokens
        if token not in STOPWORDS and token not in JUNK_KEYWORDS and len(token) > 4
    ]
    freqs = Counter(filtered)
    return [token for token, _ in freqs.most_common(n)]


def _compose_negative_prompt(base_negative: str) -> str:
    grouped = ", ".join(NEGATIVE_GROUPS.values())
    return ", ".join(item for item in [grouped, base_negative.strip()] if item)


def _build_prompt_from_strategy(sections: ArticleSections, profile: StrategyProfile) -> str:
    source = " ".join(
        [sections.title, sections.summary, sections.tags, sections.first_paragraph, sections.body]
    )
    keywords = _extract_keywords(source, n=6)
    context = ", ".join(keywords[:3]) if keywords else "current affairs"

    strategy_directive = VISUAL_STRATEGY_DIRECTIVES[profile.visual_strategy]

    anti_deformation_clauses = [
        "avoid close-up portraits",
        "avoid visible distorted hands",
        "prioritize coherent proportions",
        "credible editorial realism",
    ]
    if profile.avoid_close_ups:
        anti_deformation_clauses.append("faces in medium or wide shot only")

    return ", ".join(
        [
            f"news scene about {sections.title}",
            f"domain focus: {profile.domain}",
            f"context terms: {context}",
            strategy_directive,
            ", ".join(anti_deformation_clauses),
            "natural lighting, documentary style, realistic composition",
        ]
    )


def build_prompts(
    text: str,
    negative_prompt: str,
    variants: int = 1,
    strategy_override: str = "auto",
) -> PromptPack:
    variants = max(1, min(variants, 2))
    sections = _split_article_sections(text)
    profile = analyze_visual_strategy(text, override=strategy_override)
    base = _build_prompt_from_strategy(sections, profile)

    prompts = [base]
    if variants == 2:
        prompts.append(
            f"{base}, alternate angle, maintain same strategy, keep composition stable and non-chaotic"
        )

    full_negative = _compose_negative_prompt(negative_prompt)
    return PromptPack(positive_prompts=prompts, negative_prompt=full_negative, strategy_profile=profile)
