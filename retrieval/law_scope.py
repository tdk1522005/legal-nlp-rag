from __future__ import annotations

from dataclasses import dataclass

from retrieval.exact_reference import normalize_text


# Only use high-confidence out-of-scope phrases here.
#
# Do not use "thue" alone:
# after accent normalization, both "thue" (tax) and
# "thue" (rent/lease) become the same string.
OUT_OF_SCOPE_PATTERNS = (
    "thue thu nhap",
    "thue gia tri gia tang",
    "nop thue",
    "ma so thue",
    "le phi",
    "phi cong chung",
    "bao hiem xa hoi",
    "vuot den do",
    "nguoi lao dong duoc nghi phep",
    "toi pham",
    "bi phat bao nhieu",
    "phat bao nhieu tien",
    "bao nhieu nam tu",
    "ket hon",
    "cam ket hon",
    "ly hon",
)


# More specific phrases must be placed before broader phrases.
LAW_SCOPE_PATTERNS = (
    # -------------------------------------------------
    # Security registration - Decree 99/2022
    # -------------------------------------------------
    (
        "dang ky bien phap bao dam",
        "security_registration_decree_99_2022",
    ),
    (
        "xoa dang ky bien phap bao dam",
        "security_registration_decree_99_2022",
    ),
    (
        "dang ky the chap",
        "security_registration_decree_99_2022",
    ),
    (
        "xoa dang ky the chap",
        "security_registration_decree_99_2022",
    ),

    # -------------------------------------------------
    # Secured obligations - Decree 21/2021
    # -------------------------------------------------
    (
        "bao dam thuc hien nghia vu",
        "secured_obligations_decree_21_2021",
    ),
    (
        "tai san bao dam",
        "secured_obligations_decree_21_2021",
    ),
    (
        "cam co tai san",
        "secured_obligations_decree_21_2021",
    ),
    (
        "cam giu tai san",
        "secured_obligations_decree_21_2021",
    ),
    (
        "the chap",
        "secured_obligations_decree_21_2021",
    ),
    (
        "bao lanh",
        "secured_obligations_decree_21_2021",
    ),
    (
        "dat coc",
        "secured_obligations_decree_21_2021",
    ),

    # -------------------------------------------------
    # Hui / ho / bieu / phuong - Decree 19/2019
    # These must appear before generic interest rules.
    # -------------------------------------------------
    (
        "ho hui bieu phuong",
        "hui_decree_19_2019",
    ),
    (
        "ho co lai",
        "hui_decree_19_2019",
    ),
    (
        "day ho",
        "hui_decree_19_2019",
    ),
    (
        "chu ho",
        "hui_decree_19_2019",
    ),
    (
        "hui",
        "hui_decree_19_2019",
    ),

    # -------------------------------------------------
    # Clan common property - Resolution 01/2020
    # -------------------------------------------------
    (
        "tai san chung cua dong ho",
        "clan_property_resolution_01_2020",
    ),
    (
        "tranh chap tai san chung dong ho",
        "clan_property_resolution_01_2020",
    ),
    (
        "thanh vien dong ho",
        "clan_property_resolution_01_2020",
    ),

    # -------------------------------------------------
    # Tort compensation - Resolution 02/2022
    # -------------------------------------------------
    (
        "boi thuong thiet hai ngoai hop dong",
        "tort_compensation_resolution_02_2022",
    ),
    (
        "nguon nguy hiem cao do",
        "tort_compensation_resolution_02_2022",
    ),
    (
        "suc khoe bi xam pham",
        "tort_compensation_resolution_02_2022",
    ),
    (
        "tinh mang bi xam pham",
        "tort_compensation_resolution_02_2022",
    ),
    (
        "danh du nhan pham uy tin bi xam pham",
        "tort_compensation_resolution_02_2022",
    ),

    # -------------------------------------------------
    # Interest / loan - Resolution 01/2019
    # -------------------------------------------------
    (
        "lai no qua han",
        "interest_penalty_resolution_01_2019",
    ),
    (
        "lai cham tra",
        "interest_penalty_resolution_01_2019",
    ),
    (
        "lai suat",
        "interest_penalty_resolution_01_2019",
    ),
    (
        "phat vi pham",
        "interest_penalty_resolution_01_2019",
    ),
    (
        "hop dong vay",
        "interest_penalty_resolution_01_2019",
    ),

    # -------------------------------------------------
    # Generic civil transactions
    # -------------------------------------------------
    (
        "giao dich dan su bi vo hieu",
        "civil_code_2015",
    ),
    (
        "hop dong thue tai san",
        "civil_code_2015",
    ),
    (
        "giao dich dan su",
        "civil_code_2015",
    ),
)


@dataclass(frozen=True)
class LawScopeResult:
    law_id: str | None
    matched_phrase: str | None
    is_out_of_scope: bool
    matched_out_of_scope_phrase: str | None


# A detected scope is allowed to retrieve from several related laws.
#
# effective_law_ids will remove documents that were not legally
# effective at the query date.
LAW_SCOPE_EQUIVALENTS = {
    "civil_code_2015": (
        "civil_code_2005",
        "civil_code_2015",
    ),

    "civil_procedure_code_2015": (
        "civil_procedure_code_2015",
    ),

    "secured_obligations_decree_21_2021": (
        "civil_code_2005",
        "civil_code_2015",
        "secured_obligations_decree_21_2021",
        "security_registration_decree_99_2022",
    ),

    "security_registration_decree_99_2022": (
        "civil_code_2005",
        "civil_code_2015",
        "secured_obligations_decree_21_2021",
        "security_registration_decree_99_2022",
    ),

    "interest_penalty_resolution_01_2019": (
        "civil_code_2005",
        "civil_code_2015",
        "interest_penalty_resolution_01_2019",
    ),

    "tort_compensation_resolution_02_2022": (
        "civil_code_2005",
        "civil_code_2015",
        "tort_compensation_resolution_02_2022",
    ),

    "hui_decree_19_2019": (
        "civil_code_2005",
        "civil_code_2015",
        "hui_decree_19_2019",
    ),

    "clan_property_resolution_01_2020": (
        "civil_code_2005",
        "civil_code_2015",
        "civil_procedure_code_2015",
        "clan_property_resolution_01_2020",
    ),
}


def filter_effective_law_ids_by_scope(
    scope: LawScopeResult,
    effective_law_ids: list[str],
) -> list[str]:
    """
    Restrict effective laws only when a high-confidence legal
    scope was detected.

    If no scope is detected, semantic retrieval keeps the full
    effective-law candidate set.
    """
    if scope.is_out_of_scope:
        return []

    if scope.law_id is None:
        return list(effective_law_ids)

    candidate_law_ids = LAW_SCOPE_EQUIVALENTS.get(
        scope.law_id,
        (scope.law_id,),
    )

    candidate_set = set(candidate_law_ids)

    return [
        law_id
        for law_id in effective_law_ids
        if law_id in candidate_set
    ]


def detect_law_scope(
    query: str,
) -> LawScopeResult:
    normalized_query = normalize_text(query)

    for pattern in OUT_OF_SCOPE_PATTERNS:
        if pattern in normalized_query:
            return LawScopeResult(
                law_id=None,
                matched_phrase=None,
                is_out_of_scope=True,
                matched_out_of_scope_phrase=pattern,
            )

    for phrase, law_id in LAW_SCOPE_PATTERNS:
        if phrase in normalized_query:
            return LawScopeResult(
                law_id=law_id,
                matched_phrase=phrase,
                is_out_of_scope=False,
                matched_out_of_scope_phrase=None,
            )

    return LawScopeResult(
        law_id=None,
        matched_phrase=None,
        is_out_of_scope=False,
        matched_out_of_scope_phrase=None,
    )
