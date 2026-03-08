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
    "editorial_photo": "photojournalistic photograph, professional news photography, medium-wide framing, visible action and contextual background, natural lighting",
    "conceptual": "editorial conceptual scene with concrete objects, clear foreground subject, clean composition, natural light behavior",
    "infographic_like": "newsroom-style data scene, screens or charts visible in context, clean composition, wide framing",
    "industrial": "industrial field report photography, operational machinery and workers in context, wide shot, natural light",
    "institutional": "institutional press photography, conference room or podium context, medium-wide framing, documentary tone",
    "documentary_wide": "on-location documentary news photograph, environmental context first, medium-wide to wide framing, press coverage look",
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
    "fake typography",
    "jersey logo close-up",
    "soccer emblem collage",
    "collage of club symbols",
]

UNIVERSAL_STRUCTURAL_NEGATIVE_TERMS = [
    "blurry",
    "blurry details",
    "low quality",
    "jpeg artifacts",
    "unreadable text",
    "watermark",
    "logo",
    "deformed anatomy",
    "malformed body",
    "broken proportions",
    "extra limbs",
    "deformed hands",
    "extra fingers",
    "duplicate face",
    "asymmetrical eyes",
    "crossed eyes",
    "plastic skin",
    "uncanny",
]

# Backwards-compatible alias used by tests and existing call sites.
BASE_NEGATIVE_TERMS = UNIVERSAL_STRUCTURAL_NEGATIVE_TERMS

DOMAIN_NEGATIVE_TERMS = {
    "sports_transfers": SPORTS_EDITORIAL_NEGATIVE_TERMS,
    "political_institutional": [
        "warped perspective",
        "poster layout",
        "duplicate person",
        "duplicated subject",
        "crowded symbolic background",
        "excessive flags",
        "chaotic composition",
        "poster-like layout",
        "over-staged political imagery",
    ],
    "economy_markets": ["poster layout"],
    "conflict_disaster_crisis": ["poster layout"],
}

STRATEGY_NEGATIVE_TERMS = {
    "editorial_photo": ["poster layout"],
    "documentary_wide": ["poster layout", "extreme close-up portrait"],
    "institutional": ["extreme close-up portrait"],
}

NEGATIVE_CANONICAL_ALIASES = {
    "jpeg artifact": "jpeg artifacts",
    "jpeg artifacts": "jpeg artifacts",
    "low-quality": "low quality",
    "extra limb": "extra limbs",
    "extra finger": "extra fingers",
    "large groups": "large group",
    "crowded scenes": "crowded scene",
}

GROUP_SCENE_TERMS = {
    "crowd",
    "multitud",
    "group",
    "team",
    "delegation",
    "delegates",
    "summit",
    "conference",
    "press room",
    "stadium",
}

CONTRADICTORY_GROUP_NEGATIVES = {
    "chaotic crowd",
    "crowded scene",
    "crowded scenes",
    "large group",
    "large groups",
}

NEGATIVE_CONFLICT_RULES = {
    "infographic_mode": {
        "triggers": {"infographic", "diagram", "chart", "data-driven"},
        "blocked": {"infographic", "infographic board"},
    },
    "crowd_scene": {
        "triggers": GROUP_SCENE_TERMS,
        "blocked": CONTRADICTORY_GROUP_NEGATIVES,
    },
    "institutional_group_scene": {
        "triggers": {"summit", "conference", "delegation", "delegates", "press briefing"},
        "blocked": {"extreme close-up portrait"},
    },
    "sports_scene": {
        "triggers": {"stadium", "training ground", "team", "match", "signing"},
        "blocked": CONTRADICTORY_GROUP_NEGATIVES,
    },
    "disaster_scene": {
        "triggers": {"evacuation", "rescue", "crisis", "disaster", "emergency response"},
        "blocked": CONTRADICTORY_GROUP_NEGATIVES,
    },
}

PROMPT_MAX_CHARS = 360

ABSTRACT_NOTE_PATTERNS = (
    "credible editorial realism",
    "high-stakes setting",
    "institutional significance",
    "policy environment",
    "local heuristic composition",
    "local heuristic editorial style",
    "safe diffusion composition",
    "balanced editorial framing",
)

WEAK_META_PATTERNS = (
    "plausible",
    "authentic",
    "realistic",
    "cinematic",
    "epic",
)

DOMAIN_VISUAL_CUES = {
    "political_institutional": [
        "photojournalistic photograph",
        "institutional press photography",
        "conference or summit setting",
        "medium-wide framing",
    ],
    "economy_markets": [
        "professional news photography",
        "financial newsroom or trading floor context",
        "visible data screens or market indicators",
    ],
    "conflict_disaster_crisis": [
        "on-location field reporting",
        "documentary news photograph",
        "wide environmental context",
    ],
    "sports_transfers": [
        "realistic sports editorial coverage",
        "professional sports news photography",
        "training ground or stadium context",
    ],
}

STRATEGY_VISUAL_CUES = {
    "editorial_photo": ["photojournalistic photograph", "press photography look"],
    "institutional": ["institutional press photography", "formal documentary framing"],
    "documentary_wide": ["documentary news photograph", "wide environmental framing"],
}

POLITICAL_EDITORIAL_STRATEGIES = {"editorial_photo", "institutional", "documentary_wide"}

POLITICAL_EDITORIAL_SCENE_CUES = [
    "official meeting room",
    "conference table with delegates seated",
    "documentary press-photo realism",
    "natural indoor lighting",
    "medium-wide documentary framing",
]
POLITICAL_DOMAIN_EQUIVALENT_HINTS = (
    "politic",
    "diplomat",
    "geopolit",
    "government",
    "state_affairs",
)

POLITICAL_OVERLOAD_CUE_HINTS = (
    "flag",
    "flags",
    "leader",
    "leaders",
    "summit",
    "nameplate",
    "nameplates",
)


def is_political_domain_equivalent(domain: str) -> bool:
    normalized = re.sub(r"[_\-]+", " ", (domain or "").strip().lower())
    if not normalized:
        return False
    if normalized in {"political institutional", "political news"}:
        return True
    return any(hint in normalized for hint in POLITICAL_DOMAIN_EQUIVALENT_HINTS)


def _count_person_name_segments(segments: list[str]) -> int:
    return sum(1 for segment in segments if _is_person_name_segment(segment))


def _is_political_overload_segment(segment: str) -> bool:
    lowered = segment.lower()
    return any(hint in lowered for hint in POLITICAL_OVERLOAD_CUE_HINTS)


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


def _trim_text(text: str, max_len: int = 90) -> str:
    normalized = _normalize_text(text)
    if len(normalized) <= max_len:
        return normalized
    cut = normalized[:max_len].rsplit(" ", 1)[0].strip()
    return cut or normalized[:max_len]


def _canonical_negative_term(term: str) -> str:
    normalized = re.sub(r"\s+", " ", term.lower().strip(" ,.;:-"))
    return NEGATIVE_CANONICAL_ALIASES.get(normalized, normalized)


def _flatten_negative_terms(*chunks: str, terms: list[str] | None = None) -> list[str]:
    flattened: list[str] = []
    if terms:
        flattened.extend(terms)
    for chunk in chunks:
        flattened.extend(chunk.split(","))
    return flattened


def _dedupe_negative_terms(*chunks: str, terms: list[str] | None = None) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for raw in _flatten_negative_terms(*chunks, terms=terms):
        term = _normalize_text(raw)
        if not term:
            continue
        key = _canonical_negative_term(term)
        if key in seen:
            continue
        seen.add(key)
        ordered.append(term)
    return ordered


def _negative_conflict_blocklist(domain: str, visual_strategy: str, positive_text: str) -> set[str]:
    lowered = positive_text.lower()
    blocked: set[str] = set()

    for rule in NEGATIVE_CONFLICT_RULES.values():
        if any(trigger in lowered for trigger in rule["triggers"]):
            blocked.update(rule["blocked"])

    if visual_strategy == "infographic_like":
        blocked.update(NEGATIVE_CONFLICT_RULES["infographic_mode"]["blocked"])
    if domain in {"political_institutional", "economy_markets", "conflict_disaster_crisis", "sports_transfers"}:
        if any(trigger in lowered for trigger in GROUP_SCENE_TERMS):
            blocked.update(CONTRADICTORY_GROUP_NEGATIVES)
    if domain == "political_institutional":
        blocked.update(NEGATIVE_CONFLICT_RULES["institutional_group_scene"]["blocked"])

    return {_canonical_negative_term(term) for term in blocked}


def _filter_conflicting_negative_terms(
    terms: list[str],
    domain: str,
    visual_strategy: str,
    positive_text: str,
) -> list[str]:
    blocked = _negative_conflict_blocklist(domain, visual_strategy, positive_text)
    if not blocked:
        return terms
    return [term for term in terms if _canonical_negative_term(term) not in blocked]


def _universal_negative_terms() -> list[str]:
    return list(UNIVERSAL_STRUCTURAL_NEGATIVE_TERMS)


def _strategy_negative_terms(visual_strategy: str) -> list[str]:
    return list(STRATEGY_NEGATIVE_TERMS.get(visual_strategy, []))


def _domain_negative_terms(domain: str) -> list[str]:
    return list(DOMAIN_NEGATIVE_TERMS.get(domain, []))


def compose_negative_prompt(
    base_negative: str,
    extra_negative: str = "",
    domain: str = "",
    visual_strategy: str = "",
    positive_context: str = "",
) -> str:
    domain_terms = _domain_negative_terms(domain)
    strategy_terms = _strategy_negative_terms(visual_strategy)
    universal_terms = _universal_negative_terms()

    terms = _dedupe_negative_terms(
        extra_negative.strip(),
        base_negative.strip(),
        terms=[*universal_terms, *strategy_terms, *domain_terms],
    )
    filtered = _filter_conflicting_negative_terms(
        terms,
        domain=domain,
        visual_strategy=visual_strategy,
        positive_text=positive_context,
    )
    return ", ".join(filtered)


def _split_segments(text: str) -> list[str]:
    parts = re.split(r"[,;]", text)
    return [_normalize_text(part) for part in parts if _normalize_text(part)]


def _is_abstract_segment(segment: str) -> bool:
    lowered = segment.lower()
    return any(pattern in lowered for pattern in ABSTRACT_NOTE_PATTERNS)


def _is_weak_meta_segment(segment: str) -> bool:
    lowered = _normalize_text(segment).lower()
    if not lowered:
        return True
    if lowered in WEAK_META_PATTERNS:
        return True
    words = lowered.split()
    return len(words) <= 3 and all(word in WEAK_META_PATTERNS for word in words)


def _sanitize_note_segments(text: str) -> list[str]:
    clean: list[str] = []
    seen: set[str] = set()
    for segment in _split_segments(text):
        key = segment.lower()
        if key in seen:
            continue
        if _is_abstract_segment(segment):
            continue
        if _is_weak_meta_segment(segment):
            continue
        seen.add(key)
        clean.append(segment)
    return clean


def _build_visual_cues(domain: str, visual_strategy: str) -> list[str]:
    cues = [*DOMAIN_VISUAL_CUES.get(domain, []), *STRATEGY_VISUAL_CUES.get(visual_strategy, [])]
    deduped: list[str] = []
    seen: set[str] = set()
    for cue in cues:
        key = cue.lower().strip()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(cue)
    return deduped[:4]


def _segment_concrete_score(segment: str) -> int:
    lowered = segment.lower()
    score = 0
    if any(token in lowered for token in (" in ", " at ", "inside", "outside", "street", "room", "stadium", "lab", "factory")):
        score += 2
    if any(token in lowered for token in ("with", "showing", "holding", "near", "background", "foreground")):
        score += 1
    if re.search(r"\d", segment):
        score += 1
    if len(segment.split()) >= 4:
        score += 1
    if _is_abstract_segment(segment):
        score -= 3
    return score


def _semantic_core_segments(segments: list[str], max_items: int = 4) -> list[str]:
    if not segments:
        return []

    indexed = list(enumerate(segments))
    ranked = sorted(
        indexed,
        key=lambda item: (_segment_concrete_score(item[1]), -item[0]),
        reverse=True,
    )
    selected = sorted(idx for idx, _ in ranked[:max_items])
    return [segments[idx] for idx in selected if _segment_concrete_score(segments[idx]) >= 0]


def _append_with_budget(result_parts: list[str], block: str, max_chars: int) -> bool:
    value = _normalize_text(block)
    if not value:
        return True
    candidate = ", ".join([*result_parts, value])
    if len(candidate) <= max_chars:
        result_parts.append(value)
        return True
    return False

def _is_person_name_segment(segment: str) -> bool:
    words = re.findall(r"\b[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+\b", segment)
    return len(words) >= 2


def _simplify_political_entity_segments(segments: list[str]) -> list[str]:
    if not segments:
        return []

    named_count = _count_person_name_segments(segments)
    simplified: list[str] = []
    generic_added = False

    for segment in segments:
        if _is_person_name_segment(segment):
            if named_count > 1 and not generic_added:
                simplified.append("senior officials and delegates")
                generic_added = True
            elif named_count <= 1:
                simplified.append(segment)
            continue
        if _is_political_overload_segment(segment):
            continue
        simplified.append(segment)

    return _unique_nonempty(simplified)


def _is_political_editorial_scene(domain: str, visual_strategy: str) -> bool:
    return is_political_domain_equivalent(domain) and visual_strategy in POLITICAL_EDITORIAL_STRATEGIES



def _compose_render_first_prompt(
    prompt_main: str,
    composition_notes: str,
    style_notes: str,
    avoid_close_ups: bool,
    domain: str,
    visual_strategy: str,
    max_chars: int = PROMPT_MAX_CHARS,
) -> str:
    main_segments_raw = _split_segments(prompt_main)
    main_segments = [
        segment
        for segment in main_segments_raw
        if not _is_abstract_segment(segment) and not _is_weak_meta_segment(segment)
    ]
    if not main_segments:
        main_segments = ["editorial documentary scene"]

    if _is_political_editorial_scene(domain, visual_strategy):
        main_segments = _simplify_political_entity_segments(main_segments)

    semantic_core = _semantic_core_segments(main_segments, max_items=4)
    remaining_main = [segment for segment in main_segments if segment not in semantic_core]

    composition_segments = _sanitize_note_segments(composition_notes)
    style_segments = _sanitize_note_segments(style_notes)

    style = ", ".join(style_segments[:2]) if style_segments else ""
    guardrails = ["coherent anatomy", "natural skin texture", "physically coherent scene"]
    if avoid_close_ups:
        guardrails.insert(0, "medium or wide framing")

    visual_cues = _build_visual_cues(domain, visual_strategy)
    if _is_political_editorial_scene(domain, visual_strategy):
        visual_cues = _unique_nonempty([*visual_cues, *POLITICAL_EDITORIAL_SCENE_CUES])
        guardrails.extend(
            [
                "avoid crowded symbolic background",
                "avoid excessive flags and political iconography",
                "avoid over-staged political hero shot",
            ]
        )

    sports_priority: list[str] = []
    if domain == "sports_transfers":
        sports_priority = [
            SPORTS_EDITORIAL_POSITIVE_GUARDRAILS[0],
            SPORTS_EDITORIAL_POSITIVE_GUARDRAILS[5],
        ]
        guardrails.extend(SPORTS_EDITORIAL_POSITIVE_GUARDRAILS[1:5])

    result_parts: list[str] = []

    for block in [*visual_cues, *semantic_core, *sports_priority, *remaining_main, *composition_segments]:
        if not _append_with_budget(result_parts, block, max_chars):
            remaining = max_chars - len(", ".join(result_parts)) - 2
            if remaining > 20:
                result_parts.append(_normalize_text(block)[:remaining].rstrip(" ,.;:-"))
            return ", ".join(result_parts)

    # Lower priority blocks: if budget overflows, skip silently.
    if style:
        _append_with_budget(result_parts, style, max_chars)
    for block in guardrails:
        _append_with_budget(result_parts, block, max_chars)

    return ", ".join(result_parts)


def _build_prompt_from_strategy(sections: ArticleSections, profile: StrategyProfile) -> str:
    source = " ".join(
        [sections.title, sections.summary, sections.tags, sections.first_paragraph, sections.body]
    )
    keywords = _extract_keywords(source, n=6)
    context = ", ".join(keywords[:3]) if keywords else "current affairs"

    strategy_directive = VISUAL_STRATEGY_DIRECTIVES[profile.visual_strategy]

    anti_deformation_clauses = [
        "avoid close-up portraits",
        "hands and body proportions coherent",
        "clear foreground subject and readable context",
        "natural light behavior",
    ]
    if profile.avoid_close_ups:
        anti_deformation_clauses.append("faces in medium or wide shot only")

    title_focus = _trim_text(sections.title, max_len=72)

    return ", ".join(
        [
            f"news scene about {title_focus}",
            f"domain focus: {profile.domain}",
            f"context terms: {context}",
            strategy_directive,
            ", ".join(anti_deformation_clauses),
            "documentary press coverage, visible action, medium-wide frame",
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

    pre_render_positives = list(positives)
    positives = [
        _compose_render_first_prompt(
            prompt_main=prompt,
            composition_notes=intelligence.composition_notes,
            style_notes=intelligence.style_notes,
            avoid_close_ups=intelligence.avoid_close_ups,
            domain=intelligence.domain,
            visual_strategy=intelligence.visual_strategy,
        )
        for prompt in positives
    ]

    combined_negative = compose_negative_prompt(
        base_negative=base_negative_prompt,
        extra_negative=intelligence.negative_prompt,
        domain=intelligence.domain,
        visual_strategy=intelligence.visual_strategy,
        positive_context=" ".join(positives),
    )

    political_guard_triggered = _is_political_editorial_scene(
        intelligence.domain,
        intelligence.visual_strategy,
    )
    before_names = sum(_count_person_name_segments(_split_segments(text)) for text in pre_render_positives)
    after_names = sum(_count_person_name_segments(_split_segments(text)) for text in positives)
    anti_text_negative_applied = "unreadable text" in combined_negative.lower()

    return PromptPlan(
        positive_prompts=positives,
        negative_prompt=combined_negative,
        strategy_effective=intelligence.visual_strategy,
        domain=intelligence.domain,
        source=intelligence.source,
        strategy_adjustment_reason=intelligence.strategy_adjustment_reason,
        semantic_adjustment_reason=intelligence.semantic_adjustment_reason,
        semantic_validation_status=intelligence.semantic_validation_status,
        sanitation_flags={
            "political_guard_triggered": political_guard_triggered,
            "multi_name_sanitized": political_guard_triggered and after_names < before_names,
            "political_domain_equivalent_detected": is_political_domain_equivalent(intelligence.domain),
            "anti_text_negative_applied": anti_text_negative_applied,
        },
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
