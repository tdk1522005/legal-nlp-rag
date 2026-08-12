from __future__ import annotations

from dataclasses import dataclass

from retrieval.exact_reference import normalize_text


OUT_OF_SCOPE_PATTERNS = (
    "thue ",
    "thue",
    "le phi",
    "phi cong chung",
    "bao hiem xa hoi",
    "vuot den do",
    "nguoi lao dong duoc nghi phep",
    "toi pham",
    "bi phat bao nhieu",
    "phat bao nhieu tien",
    "bao nhieu nam tu",
)


LAW_SCOPE_PATTERNS = (
    (
        "giao dich dan su bi vo hieu",
        "civil_code_2015",
    ),
    (
        "giao dich dan su",
        "civil_code_2015",
    ),
    (
        "cam ket hon",
        "marriage_family_law_2014",
    ),
    (
        "ket hon",
        "marriage_family_law_2014",
    ),
    (
        "hon nhan",
        "marriage_family_law_2014",
    ),
    (
        "quyen su dung dat",
        "land_law_2024",
    ),
    (
        "kinh doanh bat dong san",
        "real_estate_business_law_2023",
    ),
    (
        "cong chung vien",
        "notary_law_2024",
    ),
    (
        "cong chung",
        "notary_law_2024",
    ),
    (
        "nha o",
        "housing_law_2023",
    ),
)


@dataclass(
    frozen=True
)
class LawScopeResult:
    law_id: str | None
    matched_phrase: str | None
    is_out_of_scope: bool
    matched_out_of_scope_phrase: str | None


LAW_SCOPE_EQUIVALENTS = {
    "civil_code_2015": (
        "civil_code_2005",
        "civil_code_2015",
    ),
    "civil_procedure_code_2015": (
        "civil_procedure_code_2015",
    ),
    "housing_law_2023": (
        "housing_law_2023",
    ),
    "land_law_2024": (
        "land_law_2024",
    ),
    "marriage_family_law_2014": (
        "marriage_family_law_2014",
    ),
    "notary_law_2024": (
        "notary_law_2024",
    ),
    "real_estate_business_law_2023": (
        "real_estate_business_law_2023",
    ),
}


def filter_effective_law_ids_by_scope(
    scope: LawScopeResult,
    effective_law_ids: list[str],
) -> list[str]:
    """
    Gi?i h?n c?c lu?t c? hi?u l?c theo ph?m vi c?u h?i.

    V? d?:
    - c?u h?i v? giao d?ch d?n s? ? hi?n t?i
      -> civil_code_2015
    - c?u h?i v? giao d?ch d?n s? n?m 2010
      -> civil_code_2005

    N?u kh?ng x?c ??nh ???c scope th? gi? nguy?n
    danh s?ch lu?t c? hi?u l?c.
    """
    if scope.is_out_of_scope:
        return []

    if scope.law_id is None:
        return list(
            effective_law_ids
        )

    candidate_law_ids = (
        LAW_SCOPE_EQUIVALENTS.get(
            scope.law_id,
            (scope.law_id,),
        )
    )

    candidate_set = set(
        candidate_law_ids
    )

    return [
        law_id
        for law_id in effective_law_ids
        if law_id in candidate_set
    ]


def detect_law_scope(
    query: str,
) -> LawScopeResult:
    normalized_query = normalize_text(
        query
    )

    # -------------------------------------------------
    # 1. C?c intent m? corpus hi?n t?i kh?ng ??
    #    c?n c? ?? tr? l?i.
    # -------------------------------------------------
    for pattern in OUT_OF_SCOPE_PATTERNS:
        if pattern in normalized_query:
            return LawScopeResult(
                law_id=None,
                matched_phrase=None,
                is_out_of_scope=True,
                matched_out_of_scope_phrase=pattern,
            )

    # -------------------------------------------------
    # 2. Nh?n di?n ph?m vi lu?t t? thu?t ng? ph?p l?.
    # -------------------------------------------------
    for phrase, law_id in LAW_SCOPE_PATTERNS:
        if phrase in normalized_query:
            return LawScopeResult(
                law_id=law_id,
                matched_phrase=phrase,
                is_out_of_scope=False,
                matched_out_of_scope_phrase=None,
            )

    # -------------------------------------------------
    # 3. Kh?ng x?c ??nh ???c scope.
    #
    # Kh?ng ??ng ngh?a ch?c ch?n l? ngo?i corpus.
    # Semantic retrieval v?n c? th? x? l? sau.
    # -------------------------------------------------
    return LawScopeResult(
        law_id=None,
        matched_phrase=None,
        is_out_of_scope=False,
        matched_out_of_scope_phrase=None,
    )
