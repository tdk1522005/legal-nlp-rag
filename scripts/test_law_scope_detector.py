from __future__ import annotations

from retrieval.exact_reference import normalize_text


# =========================================================
# C?c intent n?m ngo?i corpus hi?n t?i
# =========================================================

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


# =========================================================
# Mapping c?m t? ph?p l? -> law_id
#
# C?m d?i h?n ph?i ??ng tr??c c?m ng?n h?n.
# =========================================================

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


def detect_law_scope(
    query: str,
) -> tuple[str | None, str | None]:
    normalized_query = normalize_text(
        query
    )

    # ---------------------------------------------
    # Ch?n tr??c c?c intent ngo?i corpus.
    # V? d?:
    # - thu? nh? ??t
    # - l? ph?
    # - h?nh s?
    # - giao th?ng
    # ---------------------------------------------

    for pattern in OUT_OF_SCOPE_PATTERNS:
        if pattern in normalized_query:
            return None, None

    # ---------------------------------------------
    # X?c ??nh ph?m vi lu?t
    # ---------------------------------------------

    for phrase, law_id in LAW_SCOPE_PATTERNS:
        if phrase in normalized_query:
            return law_id, phrase

    return None, None


def main() -> None:
    queries = [
        # =================================================
        # Near-domain OOD
        # Mong mu?n: NONE
        # =================================================
        "thue khi chuyen nhuong quyen su dung dat la bao nhieu",
        "phi cong chung hop dong mua ban nha dat la bao nhieu",
        "nguoi ket hon phai dong le phi bao nhieu",
        "kinh doanh bat dong san phai dong thue bao nhieu",
        "nha o phai dong thue bao nhieu",
        "giao dich dan su phai dong thue bao nhieu",
        "toi pham chiem doat quyen su dung dat bi phat bao nhieu nam",

        # =================================================
        # In-domain
        # Mong mu?n: x?c ??nh ??ng lu?t
        # =================================================
        "cong chung vien co quyen gi",
        "dieu kien de tro thanh cong chung vien la gi",
        "quyen su dung dat cua ca nhan duoc quy dinh nhu the nao",
        "nhung hanh vi nao bi cam trong hon nhan",
        "doanh nghiep kinh doanh bat dong san phai dap ung dieu kien gi",
        "nha o tham gia giao dich phai dap ung dieu kien nao",
        "giao dich dan su co hieu luc khi nao",

        # =================================================
        # Stress test
        # =================================================
        "nguoi lao dong co quyen su dung dat khong",
        "nguoi pham toi co duoc so huu nha khong",
        "cong chung vien mua dat co phai dong thue khong",
    ]

    for query in queries:
        law_id, matched_phrase = (
            detect_law_scope(
                query
            )
        )

        print()
        print("=" * 80)
        print(
            "QUERY:",
            query,
        )

        if law_id is None:
            print("SCOPE: NONE")
        else:
            print(
                f"SCOPE: {law_id} | "
                f"MATCH='{matched_phrase}'"
            )


if __name__ == "__main__":
    main()
