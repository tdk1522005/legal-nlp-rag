from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


LAW_ALIASES = {
    # -------------------------------------------------
    # Civil codes
    # -------------------------------------------------
    "civil_code_2005": (
        "bo luat dan su 2005",
        "blds 2005",
    ),
    "civil_code_2015": (
        "bo luat dan su 2015",
        "blds 2015",
    ),
    "civil_procedure_code_2015": (
        "bo luat to tung dan su 2015",
        "blttds 2015",
    ),

    # -------------------------------------------------
    # -------------------------------------------------

    # -------------------------------------------------
    # Decree 21/2021/ND-CP
    # -------------------------------------------------
    "secured_obligations_decree_21_2021": (
        "nghi dinh 21 2021 nd cp",
        "nd 21 2021 nd cp",
        "nghi dinh 21 2021",
    ),

    # -------------------------------------------------
    # Decree 99/2022/ND-CP
    # Retrieval uses consolidated document 2161/VBHN-BTP.
    # -------------------------------------------------
    "security_registration_decree_99_2022": (
        "nghi dinh 99 2022 nd cp",
        "nd 99 2022 nd cp",
        "nghi dinh 99 2022",
        "2161 vbhn btp",
        "vbhn 2161 btp",
    ),

    # -------------------------------------------------
    # Resolution 01/2019/NQ-HDTP
    # -------------------------------------------------
    "interest_penalty_resolution_01_2019": (
        "nghi quyet 01 2019 nq hdtp",
        "nq 01 2019 nq hdtp",
        "nghi quyet 01 2019",
    ),

    # -------------------------------------------------
    # Resolution 02/2022/NQ-HDTP
    # -------------------------------------------------
    "tort_compensation_resolution_02_2022": (
        "nghi quyet 02 2022 nq hdtp",
        "nq 02 2022 nq hdtp",
        "nghi quyet 02 2022",
    ),

    # -------------------------------------------------
    # Decree 19/2019/ND-CP
    # -------------------------------------------------
    "hui_decree_19_2019": (
        "nghi dinh 19 2019 nd cp",
        "nd 19 2019 nd cp",
        "nghi dinh 19 2019",
    ),

    # -------------------------------------------------
    # Resolution 01/2020/NQ-HDTP
    # -------------------------------------------------
    "clan_property_resolution_01_2020": (
        "nghi quyet 01 2020 nq hdtp",
        "nq 01 2020 nq hdtp",
        "nghi quyet 01 2020",
    ),
}


LAW_FAMILY_ALIASES = {
    "bo luat dan su": (
        "civil_code_2005",
        "civil_code_2015",
    ),
    "blds": (
        "civil_code_2005",
        "civil_code_2015",
    ),
    "bo luat to tung dan su": (
        "civil_procedure_code_2015",
    ),
    "blttds": (
        "civil_procedure_code_2015",
    ),


    # New civil-law guidance documents
    "nghi dinh ve bao dam thuc hien nghia vu": (
        "secured_obligations_decree_21_2021",
    ),
    "nghi dinh ve dang ky bien phap bao dam": (
        "security_registration_decree_99_2022",
    ),
    "nghi quyet ve lai lai suat phat vi pham": (
        "interest_penalty_resolution_01_2019",
    ),
    "nghi quyet ve boi thuong thiet hai ngoai hop dong": (
        "tort_compensation_resolution_02_2022",
    ),
    "nghi dinh ve ho hui bieu phuong": (
        "hui_decree_19_2019",
    ),
    "nghi quyet ve tai san chung cua dong ho": (
        "clan_property_resolution_01_2020",
    ),
}


@dataclass(frozen=True)
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
    allowed_law_ids: (
        list[str]
        | tuple[str, ...]
        | set[str]
        | None
    ) = None,
) -> ExactLegalReference | None:
    normalized_query = normalize_text(query)

    article_match = re.search(
        r"\bdieu\s+([0-9]+[a-z]?)\b",
        normalized_query,
    )

    if article_match is None:
        return None

    article_number = article_match.group(1)

    # -------------------------------------------------
    # 1. Explicit document identifier.
    #
    # Example:
    # "Dieu 21 Nghi dinh 19/2019/ND-CP"
    #
    # Explicit document references are resolved directly,
    # including historical documents.
    # -------------------------------------------------
    for law_id, aliases in LAW_ALIASES.items():
        if any(
            alias in normalized_query
            for alias in aliases
        ):
            return ExactLegalReference(
                law_id=law_id,
                article_number=article_number,
            )

    # -------------------------------------------------
    # 2. Document family / descriptive title.
    #
    # Example:
    # "Dieu 122 Bo luat Dan su"
    #
    # Resolve only when the legal date leaves exactly
    # one matching document in allowed_law_ids.
    # -------------------------------------------------
    if allowed_law_ids is None:
        return None

    allowed = {
        str(law_id)
        for law_id in allowed_law_ids
    }

    for alias, candidate_law_ids in (
        LAW_FAMILY_ALIASES.items()
    ):
        if alias not in normalized_query:
            continue

        matching_laws = [
            law_id
            for law_id in candidate_law_ids
            if law_id in allowed
        ]

        if len(matching_laws) == 1:
            return ExactLegalReference(
                law_id=matching_laws[0],
                article_number=article_number,
            )

    return None
