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

EVENT_KEYWORDS = {
    "cumbre": "international political summit",
    "summit": "international political summit",
    "foro": "international forum",
    "conferencia": "international conference",
    "reunión": "international diplomatic meeting",
    "reunion": "international diplomatic meeting",
    "encuentro": "international diplomatic meeting",
    "eleccion": "electoral event",
    "elecciones": "electoral event",
    "protesta": "public protest",
    "manifestacion": "public protest",
    "incendio": "fire emergency response",
    "terremoto": "earthquake response",
    "inundacion": "flood emergency response",
    "acuerdo": "diplomatic agreement",
    "jornada": "institutional event",
}

EVENT_KIND_MAP = {
    "cumbre": "Summit",
    "summit": "Summit",
    "foro": "Forum",
    "conferencia": "Conference",
    "reunión": "Meeting",
    "reunion": "Meeting",
    "encuentro": "Meeting",
}

TITLE_VERB_PREFIX = re.compile(
    r"^[A-ZÁÉÍÓÚÑ][^,;:.]{0,90}?\b(?:encabeza|participa en|asiste a|lidera|preside|impulsa|"
    r"abre|inaugura|attends|leads|heads)\b\s+",
    flags=re.IGNORECASE,
)


@dataclass
class PromptPack:
    positive_prompts: list[str]
    negative_prompt: str


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


def _split_article_sections(text: str) -> tuple[str, str, str, str]:
    lines = [_normalize_text(line) for line in text.splitlines() if _normalize_text(line)]
    if not lines:
        return "nota periodística", "", "", ""

    title = lines[0]
    body_lines = lines[1:]

    tag_values = [_clean_tag_prefix(line) for line in body_lines if _is_tag_line(line)]
    tags = ", ".join(_unique_nonempty(tag_values))

    useful_body = [line for line in body_lines if not _is_tag_line(line)]
    summary = useful_body[0] if useful_body else title

    first_paragraph = ""
    for line in useful_body[1:]:
        if line.lower() != summary.lower():
            first_paragraph = line
            break

    if not first_paragraph:
        first_paragraph = summary

    return title, summary, tags, first_paragraph


def _extract_keywords(text: str, n: int = 6) -> list[str]:
    lowered = text.lower()

    phrase_map = {
        "seguridad regional": "regional security",
        "america latina": "latin american relations",
        "américa latina": "latin american relations",
        "relaciones bilaterales": "bilateral relations",
        "politica internacional": "international politics",
        "política internacional": "international politics",
        "diplomacia": "diplomacy",
        "delegaciones": "official delegations",
        "comercio": "trade",
        "defensa": "defense",
        "economía": "economy",
        "economia": "economy",
        "energía": "energy",
        "energia": "energy",
        "migración": "migration",
        "migracion": "migration",
    }
    phrase_hits = [target for source, target in phrase_map.items() if source in lowered]

    tokens = re.findall(r"[a-zA-ZáéíóúÁÉÍÓÚñÑ]{4,}", lowered)
    filtered = [
        token
        for token in tokens
        if token not in STOPWORDS and token not in JUNK_KEYWORDS and len(token) > 4
    ]

    preferred_map = {
        "lideres": "leaders",
        "líderes": "leaders",
        "diplomacia": "diplomacy",
        "delegaciones": "official delegations",
        "cumbre": "summit",
        "conferencia": "conference",
        "comercio": "trade",
        "defensa": "defense",
        "economia": "economy",
        "economía": "economy",
        "energia": "energy",
        "energía": "energy",
        "migracion": "migration",
        "migración": "migration",
    }

    prioritized_tokens = [preferred_map[token] for token in filtered if token in preferred_map]
    freqs = Counter(filtered)
    fallback_tokens = [token for token, _ in freqs.most_common(max(n * 2, 8))]

    candidates = _unique_nonempty(phrase_hits + prioritized_tokens + fallback_tokens)
    if not candidates:
        return ["international diplomacy"]
    return candidates[:n]


def _extract_entities(title: str, summary: str, tags: str, first_paragraph: str) -> dict[str, list[str] | str]:
    source = ". ".join([title, summary, tags, first_paragraph]).strip()
    raw_people = re.findall(r"\b([A-ZÁÉÍÓÚÑ][a-záéíóúñ]+\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+)\b", source)
    people = [
        person
        for person in raw_people
        if person.split()[1].lower() not in {"la", "el", "los", "las", "de", "del"}
        and person.lower() not in {"américa latina", "estados unidos"}
    ]

    location_pattern = r"\b(?:en|desde|de|hacia)\s+([A-ZÁÉÍÓÚÑ][\wáéíóúñÁÉÍÓÚÑ-]*(?:\s+[A-ZÁÉÍÓÚÑ][\wáéíóúñÁÉÍÓÚÑ-]*)*)"
    locations = re.findall(location_pattern, source)

    org_pattern = (
        r"\b([A-ZÁÉÍÓÚÑ]{2,}(?:\s+[A-ZÁÉÍÓÚÑ]{2,})*|"
        r"(?:Ministerio|Gobierno|ONU|OTAN|UE|Casa Blanca|Congreso|Presidencia)"
        r"(?:\s+[A-ZÁÉÍÓÚÑa-záéíóúñ]+)*)"
    )
    organizations = re.findall(org_pattern, source)

    lowered = source.lower()
    event_type = "news event"
    for keyword, label in EVENT_KEYWORDS.items():
        if keyword in lowered:
            event_type = label
            break

    return {
        "people": _unique_nonempty(people)[:3],
        "locations": _unique_nonempty(locations)[:3],
        "organizations": _unique_nonempty(organizations)[:3],
        "event_type": event_type,
    }


def _translate_event_name(name_fragment: str) -> str:
    fragment = _normalize_text(name_fragment)
    replacements = {
        "escudo de las américas": "Shield of the Americas",
        "america latina": "Latin America",
        "américa latina": "Latin America",
    }
    lowered = fragment.lower()
    for src, target in replacements.items():
        lowered = lowered.replace(src, target)

    normalized = re.sub(r"\bdel\b", "of the", lowered, flags=re.IGNORECASE)
    normalized = re.sub(r"\bde\b", "of", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\bde las\b", "of the", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\bde los\b", "of the", normalized, flags=re.IGNORECASE)

    words = [w for w in normalized.split() if w]
    capitalized = " ".join(word if word in {"of", "the", "and"} else word.capitalize() for word in words)
    return _normalize_text(capitalized)


def _infer_event_label(title: str, event_type: str) -> str:
    cleaned_title = _normalize_text(title)
    lowered_title = cleaned_title.lower()

    stripped = TITLE_VERB_PREFIX.sub("", cleaned_title)
    stripped_lower = stripped.lower()

    marker_pattern = re.compile(
        r"\b(cumbre|summit|foro|conferencia|reunión|reunion|encuentro)\b"
        r"(?:\s+(?:del|de la|de los|de las|de|of|on))?\s*(.+)?",
        flags=re.IGNORECASE,
    )
    match = marker_pattern.search(stripped)
    if match:
        event_kind_raw = match.group(1).lower()
        event_kind = EVENT_KIND_MAP.get(event_kind_raw, "Summit")
        name_part = _normalize_text(match.group(2) or "")
        name_part = re.split(r"\b(?:en|in)\b\s+[A-ZÁÉÍÓÚÑ]", name_part, maxsplit=1)[0]
        name_part = _translate_event_name(name_part)
        if name_part and len(name_part) > 3:
            if event_kind.lower() in name_part.lower():
                return name_part
            return f"{name_part} {event_kind}"

    if any(token in lowered_title for token in {"cumbre", "summit", "foro", "conferencia"}):
        return "high-level political summit"
    if any(token in stripped_lower for token in {"reunión", "reunion", "encuentro"}):
        return "international diplomatic meeting"

    if "summit" in event_type:
        return "high-level political summit"
    if "diplomatic" in event_type:
        return "international diplomatic meeting"
    if "conference" in event_type:
        return "international conference"
    return "major current-affairs event"


def _infer_context_clauses(event_type: str, tags: str, summary: str, first_paragraph: str) -> list[str]:
    clauses: list[str] = []
    if "summit" in event_type or "diplomatic" in event_type:
        clauses.append("diplomatic conference setting")
    elif "electoral" in event_type:
        clauses.append("election coverage atmosphere")
    elif "protest" in event_type:
        clauses.append("public demonstration environment")
    elif "press conference" in event_type:
        clauses.append("official press conference setting")
    else:
        clauses.append("institutional news setting")

    source = f"{tags} {summary} {first_paragraph}".lower()
    if "líderes" in source or "lideres" in source:
        clauses.append("meeting with regional leaders")
    if "diplom" in source:
        clauses.append("high-level international diplomacy")
    if "seguridad" in source:
        clauses.append("regional security agenda")

    return _unique_nonempty(clauses)


def _build_editorial_prompt(
    title: str,
    summary: str,
    tags: str,
    first_paragraph: str,
    entities: dict[str, list[str] | str],
) -> str:
    people = entities["people"]
    locations = entities["locations"]
    organizations = entities["organizations"]
    event_type = entities["event_type"]

    event_label = _infer_event_label(title, event_type)
    subject = ", ".join(people[:2]) if people else "senior government officials"
    location = locations[0] if locations else "an official venue"

    lead_clause = f"{subject} at the {event_label} in {location}" if event_label else f"{subject} in {location}"

    context_clauses = _infer_context_clauses(event_type, tags, summary, first_paragraph)

    org_clause = ""
    if organizations:
        org_names = [org for org in organizations if len(org) > 2 and org.lower() not in {"eeuu"}]
        if org_names:
            org_clause = f"representatives from {', '.join(org_names[:2])}"

    topic_keywords = _extract_keywords(" ".join([summary, first_paragraph, tags]), n=4)
    scene_topics = [k for k in topic_keywords if " " in k or k in {"diplomacy", "defense", "trade", "energy", "migration"}]
    topic_clause = ""
    if scene_topics:
        topic_clause = f"focus on {', '.join(scene_topics[:2])}"

    source = f"{title} {summary} {first_paragraph}".lower()
    flag_clause = ""
    if ("eeuu" in source or "u.s." in source or "estados unidos" in source) and (
        "américa latina" in source or "america latina" in source
    ):
        flag_clause = "U.S. and Latin American flags"

    visual_parts = [
        lead_clause,
        *context_clauses,
        org_clause,
        topic_clause,
        "official delegations",
        "press photographers",
        "conference hall",
        flag_clause,
        "realistic photojournalism",
        "editorial photography",
        "natural lighting",
    ]
    final_parts = _unique_nonempty([part for part in visual_parts if part])
    return ", ".join(final_parts)


def build_prompts(
    text: str,
    negative_prompt: str,
    variants: int = 1,
) -> PromptPack:
    variants = max(1, min(variants, 2))
    title, summary, tags, first_paragraph = _split_article_sections(text)
    entities = _extract_entities(title, summary, tags, first_paragraph)
    base = _build_editorial_prompt(title, summary, tags, first_paragraph, entities)

    prompts = [base]
    if variants == 2:
        prompts.append(f"{base}, tighter framing on delegates and negotiations, documentary composition")

    return PromptPack(positive_prompts=prompts, negative_prompt=negative_prompt)
