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

DOMAIN_KEYWORDS = {
    "diplomatic": {
        "cumbre",
        "summit",
        "diplom",
        "delegación",
        "delegacion",
        "presidente",
        "canciller",
        "ministro",
        "foro",
        "bilateral",
        "embajada",
    },
    "mining": {
        "mina",
        "minero",
        "minera",
        "open pit",
        "yacimiento",
        "perforación",
        "perforacion",
        "drill",
        "ore",
        "core",
        "geología",
        "geologia",
        "haul truck",
        "planta de procesamiento",
    },
    "business": {
        "empresa",
        "empresarial",
        "ejecutivo",
        "board",
        "acuerdo comercial",
        "firmó",
        "firmo",
        "inversión",
        "inversion",
        "mercado",
        "bolsa",
        "acciones",
        "balance",
    },
    "judicial": {
        "justicia",
        "juez",
        "jueza",
        "fiscal",
        "tribunal",
        "corte",
        "policía",
        "policia",
        "detenido",
        "allanamiento",
        "investigación",
        "investigacion",
        "acusado",
    },
    "disaster": {
        "inundación",
        "inundacion",
        "incendio",
        "terremoto",
        "desastre",
        "evacuación",
        "evacuacion",
        "rescate",
    },
}

DOMAIN_DEFAULTS = {
    "diplomatic": {
        "scene": "summit conference table with delegations",
        "anchors": [
            "national flags",
            "conference nameplates",
            "table microphones",
            "press photographers",
        ],
        "shot": "medium-wide documentary shot capturing delegates and negotiation dynamics",
    },
    "mining": {
        "scene": "active open-pit mine operation",
        "anchors": [
            "haul trucks",
            "drill rig",
            "workers with helmets",
            "ore benches",
        ],
        "shot": "wide editorial field shot with operational depth and industrial context",
    },
    "business": {
        "scene": "corporate decision scene in boardroom or market context",
        "anchors": [
            "conference table",
            "financial monitors",
            "company logo backdrop",
            "executives speaking to press",
        ],
        "shot": "clean editorial medium shot with credible business atmosphere",
    },
    "judicial": {
        "scene": "judicial or police institutional coverage",
        "anchors": [
            "courthouse steps",
            "police briefing microphones",
            "official vehicles",
            "investigation documents",
        ],
        "shot": "documentary street-level shot with institutional framing",
    },
    "disaster": {
        "scene": "emergency response at damaged public infrastructure",
        "anchors": [
            "emergency responders",
            "damaged roads or buildings",
            "rescue equipment",
            "caution tape",
        ],
        "shot": "wide urgent photojournalistic shot focused on response actions",
    },
    "general": {
        "scene": "editorial press briefing environment",
        "anchors": ["podium", "reporters", "microphones", "institutional signage"],
        "shot": "neutral but specific medium editorial shot in real location",
    },
}

ACTION_PATTERNS = {
    "announcing": {"anunció", "anuncio", "anuncia", "presentó", "presento", "comunicó"},
    "negotiating": {"negocia", "negociación", "negociacion", "acuerdo", "dialogo", "reunión", "reunion"},
    "signing": {"firma", "firmó", "firmo", "suscribió", "suscribio"},
    "investigating": {"investiga", "allanó", "allano", "imputó", "detuvo", "acusó", "acuso"},
    "operating": {"producción", "produccion", "extracción", "extraccion", "perforación", "perforacion"},
    "responding": {"evacuó", "evacuo", "rescate", "asistencia", "emergencia"},
}

EVENT_LABELS = {
    "diplomatic": "high-level diplomatic development",
    "mining": "mining and resource operations update",
    "business": "business and economic development",
    "judicial": "judicial or law-enforcement development",
    "disaster": "public emergency development",
    "general": "current-affairs news event",
}


@dataclass
class PromptPack:
    positive_prompts: list[str]
    negative_prompt: str


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


def _clean_location(location: str) -> str:
    tokens = [tok for tok in _normalize_text(location).split() if tok]
    trailing_noise = {"La", "El", "Los", "Las", "Durante", "Autoridades", "Tags"}
    while len(tokens) > 1 and tokens[-1] in trailing_noise:
        tokens.pop()
    return " ".join(tokens)


def _extract_entities(full_text: str) -> dict[str, list[str]]:
    raw_people = re.findall(
        r"\b([A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+){1,2})\b",
        full_text,
    )
    people = []
    for person in raw_people:
        lowered = person.lower()
        if person.split()[1].lower() in {"la", "el", "los", "las", "de", "del"}:
            continue
        if "américa latina" in lowered or "america latina" in lowered:
            continue
        if any(noise in person.split() for noise in {"Durante", "Tags", "Autoridades"}):
            continue
        people.append(person)

    location_pattern = (
        r"\b(?:en|desde|hacia|frente a|near|in)\s+"
        r"([A-ZÁÉÍÓÚÑ][\wáéíóúñÁÉÍÓÚÑ-]*(?:\s+[A-ZÁÉÍÓÚÑ][\wáéíóúñÁÉÍÓÚÑ-]*){0,3})"
    )
    locations = [_clean_location(item) for item in re.findall(location_pattern, full_text)]

    org_pattern = (
        r"\b(?:Ministerio(?:\s+de\s+[A-ZÁÉÍÓÚÑa-záéíóúñ]+)?|Gobierno(?:\s+de\s+[A-ZÁÉÍÓÚÑa-záéíóúñ]+)?|"
        r"Presidencia|Congreso|Senado|Corte Suprema|Fiscalía|Policía|ONU|OTAN|UE|Banco Central|"
        r"[A-ZÁÉÍÓÚÑ]{2,}(?:\s+[A-ZÁÉÍÓÚÑ]{2,})*)\b"
    )
    organizations = re.findall(org_pattern, full_text)

    return {
        "people": _unique_nonempty(people)[:3],
        "locations": _unique_nonempty(locations)[:3],
        "organizations": _unique_nonempty(organizations)[:3],
    }


def _infer_domain(source_text: str) -> str:
    lowered = source_text.lower()
    scores: dict[str, int] = {key: 0 for key in DOMAIN_KEYWORDS}
    for domain, keywords in DOMAIN_KEYWORDS.items():
        for keyword in keywords:
            if keyword in lowered:
                scores[domain] += 1

    ordered = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
    best_domain, best_score = ordered[0]
    return best_domain if best_score > 0 else "general"


def _infer_action(source_text: str, domain: str) -> str:
    lowered = source_text.lower()
    for action, keywords in ACTION_PATTERNS.items():
        if any(keyword in lowered for keyword in keywords):
            return action

    default_by_domain = {
        "diplomatic": "negotiating",
        "mining": "operating",
        "business": "announcing",
        "judicial": "investigating",
        "disaster": "responding",
        "general": "announcing",
    }
    return default_by_domain.get(domain, "announcing")


def _choose_scene(source_text: str, domain: str, action: str) -> dict[str, str | list[str]]:
    lowered = source_text.lower()
    base = DOMAIN_DEFAULTS[domain].copy()

    if domain == "diplomatic":
        if "podio" in lowered or "declaración" in lowered or "statement" in lowered:
            base["scene"] = "official podium statement after diplomatic meeting"
        elif "apretón" in lowered or "handshake" in lowered:
            base["scene"] = "leaders handshake at summit venue"
    elif domain == "mining":
        if "planta" in lowered:
            base["scene"] = "mineral processing plant in operation"
        elif "geolog" in lowered or "mapeo" in lowered:
            base["scene"] = "geologists reviewing mapped drill cores at mine site"
    elif domain == "business":
        if "bolsa" in lowered or "mercado" in lowered:
            base["scene"] = "financial market scene with traders and live stock monitors"
        elif action == "signing":
            base["scene"] = "executives signing agreement at formal conference table"
    elif domain == "judicial":
        if "tribunal" in lowered or "corte" in lowered:
            base["scene"] = "courthouse exterior with legal teams and press"
        elif "polic" in lowered:
            base["scene"] = "police briefing with official spokesperson and reporters"

    return base


def _build_editorial_prompt(sections: ArticleSections) -> str:
    source = " ".join(
        [sections.title, sections.summary, sections.tags, sections.first_paragraph, sections.body]
    )
    entities = _extract_entities(source)
    domain = _infer_domain(source)
    action = _infer_action(source, domain)
    scene = _choose_scene(source, domain, action)

    people = entities["people"]
    locations = entities["locations"]
    organizations = entities["organizations"]

    subject = ", ".join(people[:2]) if people else ", ".join(organizations[:2])
    if not subject:
        subject = "senior officials and delegates"

    location = locations[0] if locations else "a formal institutional setting"
    event_label = EVENT_LABELS.get(domain, EVENT_LABELS["general"])

    action_map = {
        "announcing": "announcing key decisions",
        "negotiating": "engaged in high-level negotiations",
        "signing": "signing an official agreement",
        "investigating": "during an active investigation update",
        "operating": "during active operations",
        "responding": "coordinating emergency response",
    }
    action_phrase = action_map.get(action, "during a major current-affairs development")

    editorial_context = _extract_keywords(
        " ".join([sections.summary, sections.first_paragraph, sections.body, sections.tags]), n=5
    )
    context_terms = [term for term in editorial_context[:2] if len(term) > 4]
    if context_terms:
        context_clause = f"in a high-stakes current-affairs setting focused on {', '.join(context_terms)}"
    else:
        context_clause = "in a high-stakes current-affairs setting"

    anchors = ", ".join(scene["anchors"][:4])

    return ", ".join(
        [
            f"{subject} at a {event_label} in {location}, {action_phrase}",
            f"{scene['scene']}, featuring {anchors}",
            context_clause,
            scene["shot"],
            "realistic photojournalism, editorial photography, documentary realism, natural lighting, no fantasy, no advertising aesthetics",
        ]
    )


def build_prompts(
    text: str,
    negative_prompt: str,
    variants: int = 1,
) -> PromptPack:
    variants = max(1, min(variants, 2))
    sections = _split_article_sections(text)
    base = _build_editorial_prompt(sections)

    prompts = [base]
    if variants == 2:
        prompts.append(
            f"{base}, tighter framing around the central action and key visual anchors, maintain editorial realism"
        )

    return PromptPack(positive_prompts=prompts, negative_prompt=negative_prompt)
