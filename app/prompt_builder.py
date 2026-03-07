from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from app.strategy import StrategyProfile, analyze_visual_strategy
from app.types import PromptIntelligenceResult, PromptPlan

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

SPORTS_EDITORIAL_POSITIVE_GUARDRAILS = [
    "realistic sports editorial coverage",
    "medium or wide shot",
    "believable training ground, signing or stadium context",
    "human subject or clear sports context",
    "subtle club colors only when naturally present",
    "no readable text anywhere in frame",
]

SPORTS_EDITORIAL_NEGATIVE_TERMS = [
    "badge",
    "crest",
    "emblem",
    "shield logo",
    "poster",
    "poster layout",
    "infographic",
    "infographic board",
    "fake typography",
    "readable text",
    "visible typography",
    "jersey logo close-up",
    "soccer emblem collage",
    "collage of club symbols",
]

NEGATIVE_GROUPS = {
    "anatomy": "deformed anatomy, malformed body, broken proportions, extra limbs",
    "hands_face": "deformed hands, extra fingers, duplicate face, asymmetrical eyes, crossed eyes",
    "artificial_look": "plastic skin, uncanny valley, oversaturated colors, fake cinematic glow, artificial render",
    "composition_noise": "chaotic crowd, extreme close-up portrait, cropped face, warped perspective",
    "artifacts": "blurry, low quality, jpeg artifacts, unreadable text, watermark, logo",
    "sports_editorial": ", ".join(SPORTS_EDITORIAL_NEGATIVE_TERMS),
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


def split_article_sections(text: str) -> ArticleSections:
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


def build_document_context(text: str, max_chars: int = 8000) -> str:
    sections = split_article_sections(text)
    composed = "\n".join(
        [
            f"TITLE: {sections.title}",
            f"SUMMARY: {sections.summary}",
            f"TAGS: {sections.tags}",
            f"FIRST_PARAGRAPH: {sections.first_paragraph}",
            f"BODY: {sections.body}",
        ]
    )
    normalized = _normalize_text(composed)
    return normalized[:max_chars]


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


def _unique_negative_terms(*chunks: str) -> str:
    terms: list[str] = []
    seen: set[str] = set()
    for chunk in chunks:
        for raw in chunk.split(","):
            term = raw.strip()
            if not term:
                continue
            key = term.lower()
            if key in seen:
                continue
            seen.add(key)
            terms.append(term)
    return ", ".join(terms)


def compose_negative_prompt(base_negative: str, extra_negative: str = "") -> str:
    grouped = ", ".join(NEGATIVE_GROUPS.values())
    return _unique_negative_terms(grouped, extra_negative.strip(), base_negative.strip())


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


def build_local_fallback_result(
    text: str,
    strategy_override: str,
    base_negative_prompt: str,
    variants: int,
    fallback_reason: str,
) -> PromptIntelligenceResult:
    variants = max(1, min(variants, 2))
    sections = split_article_sections(text)
    profile = analyze_visual_strategy(text, override=strategy_override)
    base_prompt = _build_prompt_from_strategy(sections, profile)

    prompt_variants: list[str] = []
    if variants == 2:
        prompt_variants.append(
            f"{base_prompt}, alternate angle, maintain same strategy, keep composition stable and non-chaotic"
        )

    return PromptIntelligenceResult(
        source="local_fallback",
        domain=profile.domain,
        visual_strategy=profile.visual_strategy,
        human_closeup_risk=profile.human_closeup_risk,
        avoid_close_ups=profile.avoid_close_ups,
        prompt_main=base_prompt,
        prompt_variants=prompt_variants,
        negative_prompt=compose_negative_prompt(base_negative_prompt),
        composition_notes="local heuristic composition",
        style_notes="local heuristic editorial style",
        confidence=0.45,
        strategy_adjustment_reason=None,
        fallback_reason=fallback_reason,
        raw_schema_version="local.fallback.v1",
    )


def compose_prompt_plan(
    intelligence: PromptIntelligenceResult,
    base_negative_prompt: str,
    variants: int,
) -> PromptPlan:
    variants = max(1, min(variants, 2))

    main = intelligence.prompt_main.strip()
    if not main:
        main = "editorial documentary scene, realistic composition"

    positives = [main]
    for extra in intelligence.prompt_variants:
        if len(positives) >= variants:
            break
        normalized = extra.strip()
        if normalized and normalized.lower() != main.lower():
            positives.append(normalized)

    if len(positives) < variants:
        positives.append(f"{main}, alternate angle, composition remains stable")

    safe_addons = [
        intelligence.composition_notes.strip(),
        intelligence.style_notes.strip(),
        "avoid close-up portraits" if intelligence.avoid_close_ups else "",
        "credible editorial realism",
    ]
    positives = [", ".join(part for part in [prompt, *safe_addons] if part) for prompt in positives]

    if intelligence.domain == "sports_transfers":
        positives = [", ".join([prompt, *SPORTS_EDITORIAL_POSITIVE_GUARDRAILS]) for prompt in positives]

    combined_negative = compose_negative_prompt(
        base_negative=base_negative_prompt,
        extra_negative=intelligence.negative_prompt,
    )

    return PromptPlan(
        positive_prompts=positives,
        negative_prompt=combined_negative,
        strategy_effective=intelligence.visual_strategy,
        domain=intelligence.domain,
        source=intelligence.source,
        strategy_adjustment_reason=intelligence.strategy_adjustment_reason,
    )


def build_prompts(
    text: str,
    negative_prompt: str,
    variants: int = 1,
    strategy_override: str = "auto",
) -> PromptPack:
    fallback = build_local_fallback_result(
        text=text,
        strategy_override=strategy_override,
        base_negative_prompt=negative_prompt,
        variants=variants,
        fallback_reason="legacy_local_builder",
    )
    plan = compose_prompt_plan(
        intelligence=fallback,
        base_negative_prompt=negative_prompt,
        variants=variants,
    )
    profile = analyze_visual_strategy(text, override=strategy_override)
    return PromptPack(
        positive_prompts=plan.positive_prompts,
        negative_prompt=plan.negative_prompt,
        strategy_profile=profile,
    )
