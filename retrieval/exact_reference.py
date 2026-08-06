from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


LAW_ALIASES = {
    "civil_code_2005": (
        "bo luat dan su 2005",
    ),
    "civil_code_2015": (
        "bo luat dan su 2015",
    ),
    "civil_procedure_code_2015": (
        "bo luat to tung dan su 2015",
    ),
    "housing_law_2023": (
        "luat nha o 2023",
    ),
    "land_law_2024": (
        "luat dat dai 2024",
    ),
    "marriage_family_law_2014": (
        "luat hon nhan va gia dinh 2014",
        "luat hon nhan gia dinh 2014",
    ),
    "notary_law_2024": (
        "luat cong chung 2024",
    ),
    "real_estate_business_law_2023": (
        "luat kinh doanh bat dong san 2023",
    ),
}


@dataclass(
    frozen=True
)
class ExactLegalReference:
    law_id: str
    article_number: str


def normalize_text(
    value: str,
) -> str:
    clean_value = (
        str(value)
        .casefold()
        .replace(chr(273), "d")
    )

    normalized = unicodedata.normalize(
        "NFD",
        clean_value,
    )

    without_accents = "".join(
        character
        for character in normalized
        if unicodedata.category(character)
        != "Mn"
    )

    without_punctuation = re.sub(
        r"[^a-z0-9]+",
        " ",
        without_accents,
    )

    return re.sub(
        r"\s+",
        " ",
        without_punctuation,
    ).strip()


def resolve_exact_legal_reference(
    query: str,
) -> ExactLegalReference | None:
    normalized_query = normalize_text(
        query
    )

    article_match = re.search(
        r"\bdieu\s+([0-9]+[a-z]?)\b",
        normalized_query,
    )

    if article_match is None:
        return None

    article_number = (
        article_match.group(1)
    )

    for law_id, aliases in (
        LAW_ALIASES.items()
    ):
        if any(
            alias in normalized_query
            for alias in aliases
        ):
            return ExactLegalReference(
                law_id=law_id,
                article_number=(
                    article_number
                ),
            )

    return None
