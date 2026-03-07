from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

VisualStrategyName = Literal[
    "editorial_photo",
    "conceptual",
    "infographic_like",
    "industrial",
    "institutional",
    "documentary_wide",
]

DomainName = Literal[
    "political_institutional",
    "economy_markets",
    "corporate_industrial",
    "technology_science",
    "conflict_disaster_crisis",
    "people_news",
    "technical_generic",
]


@dataclass(frozen=True)
class StrategyProfile:
    domain: DomainName
    visual_strategy: VisualStrategyName
    human_closeup_risk: int
    avoid_close_ups: bool


DOMAIN_KEYWORDS: dict[DomainName, set[str]] = {
    "political_institutional": {
        "gobierno",
        "presidente",
        "ministerio",
        "congreso",
        "senado",
        "institucional",
        "diplom",
        "cumbre",
        "embajada",
        "canciller",
        "estado",
    },
    "economy_markets": {
        "econom",
        "mercado",
        "bolsa",
        "acciones",
        "inflación",
        "inflacion",
        "financ",
        "inversión",
        "inversion",
        "pib",
        "banco",
    },
    "corporate_industrial": {
        "empresa",
        "corporativo",
        "industria",
        "industrial",
        "mina",
        "minero",
        "minera",
        "open pit",
        "planta",
        "maquinaria",
        "operación",
        "operacion",
    },
    "technology_science": {
        "tecnolog",
        "ciencia",
        "cient",
        "investigación",
        "investigacion",
        "laboratorio",
        "software",
        "algoritmo",
        "satélite",
        "satelite",
        "innovación",
        "innovacion",
    },
    "conflict_disaster_crisis": {
        "crisis",
        "conflicto",
        "desastre",
        "incendio",
        "inundación",
        "inundacion",
        "terremoto",
        "emergencia",
        "rescate",
        "evacuación",
        "evacuacion",
        "guerra",
    },
    "people_news": {
        "persona",
        "personas",
        "familia",
        "ciudadanos",
        "trabajadores",
        "autoridades",
        "vocero",
        "entrevista",
        "declaró",
        "declaro",
    },
    "technical_generic": {
        "documento",
        "informe",
        "manual",
        "proceso",
        "análisis",
        "analisis",
        "esquema",
        "metodología",
        "metodologia",
    },
}

PEOPLE_RISK_KEYWORDS = {
    "retrato",
    "portrait",
    "rostro",
    "cara",
    "selfie",
    "primer plano",
    "close-up",
    "multitud",
    "crowd",
    "niños",
    "ninos",
}


def _tokenize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def classify_domain(text: str) -> DomainName:
    lowered = _tokenize(text)
    scores: dict[DomainName, int] = {key: 0 for key in DOMAIN_KEYWORDS}

    for domain, keywords in DOMAIN_KEYWORDS.items():
        for keyword in keywords:
            if keyword in lowered:
                scores[domain] += 1

    best_domain = max(scores, key=scores.get)
    if scores[best_domain] == 0:
        return "technical_generic"
    return best_domain


def estimate_human_closeup_risk(text: str, domain: DomainName) -> int:
    lowered = _tokenize(text)
    risk = 0

    for keyword in PEOPLE_RISK_KEYWORDS:
        if keyword in lowered:
            risk += 2

    if domain in {"people_news", "political_institutional"}:
        risk += 2
    if domain in {"economy_markets", "corporate_industrial", "technology_science"}:
        risk += 1

    names = re.findall(r"\b[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+", text)
    if len(names) >= 2:
        risk += 2

    return min(risk, 10)


def select_visual_strategy(domain: DomainName, human_closeup_risk: int) -> VisualStrategyName:
    if domain == "political_institutional":
        return "institutional"
    if domain == "economy_markets":
        return "infographic_like"
    if domain == "corporate_industrial":
        return "industrial"
    if domain == "technology_science":
        return "conceptual"
    if domain == "conflict_disaster_crisis":
        return "documentary_wide"
    if domain == "people_news":
        return "documentary_wide" if human_closeup_risk >= 5 else "editorial_photo"
    return "conceptual"


def analyze_visual_strategy(text: str, override: str = "auto") -> StrategyProfile:
    domain = classify_domain(text)
    risk = estimate_human_closeup_risk(text, domain)

    if override != "auto":
        strategy = override  # trusted from GUI constrained options
    else:
        strategy = select_visual_strategy(domain, risk)

    return StrategyProfile(
        domain=domain,
        visual_strategy=strategy,
        human_closeup_risk=risk,
        avoid_close_ups=risk >= 4,
    )
