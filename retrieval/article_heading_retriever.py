from __future__ import annotations

import re
from typing import Any

from retrieval.exact_reference import normalize_text


# Used only for lexical matching of legal article headings.
# Stopword removal is NOT applied before Qwen3 dense embedding.
# Keep legally meaningful words such as: co, duoc, phai, khong, khi.
HEADING_STOP_WORDS = {
    "la",
    "gi",
    "va",
    "cua",
    "nhu",
    "the",
    "nao",
    "nhung",
    "cac",
    "mot",
    "cho",
    "ve",
    "trong",
    "tai",
}


def _token_list(
    text: str,
) -> list[str]:
    normalized = normalize_text(
        text
    )

    return [
        token
        for token in normalized.split()
        if token not in HEADING_STOP_WORDS
        and len(token) > 1
    ]


def _bigrams(
    words: list[str],
) -> set[tuple[str, str]]:
    return {
        (
            words[index],
            words[index + 1],
        )
        for index in range(
            len(words) - 1
        )
    }


def extract_article_heading(
    text: str,
    article_number: str,
) -> str:
    """
    Extract the article heading from a legal chunk.

    Example:
        Dieu 18. Quyen va nghia vu...
        ->
        quyen va nghia vu...
    """
    prefix = (
        "dieu "
        + str(article_number).strip()
    )

    for line in str(text).splitlines():
        line = line.strip()

        if not line:
            continue

        normalized_line = normalize_text(
            line
        )

        if normalized_line.startswith(
            prefix + " "
        ):
            return normalized_line[
                len(prefix):
            ].strip()

    return ""


def score_article_heading(
    query: str,
    heading: str,
    scope_phrase: str | None = None,
) -> float:
    """
    Score how well an article heading matches the query.

    Main signals:
    - query token recall
    - intent token recall
    - heading precision
    - intent bigram preservation
    """
    query_words = _token_list(
        query
    )

    heading_words = _token_list(
        heading
    )

    query_set = set(
        query_words
    )

    heading_set = set(
        heading_words
    )

    if (
        not query_set
        or not heading_set
    ):
        return 0.0

    common = (
        query_set
        & heading_set
    )

    full_recall = (
        len(common)
        / len(query_set)
    )

    heading_precision = (
        len(common)
        / len(heading_set)
    )

    scope_words = (
        set(
            _token_list(
                scope_phrase
            )
        )
        if scope_phrase
        else set()
    )

    intent_words = [
        word
        for word in query_words
        if word not in scope_words
    ]

    intent_set = set(
        intent_words
    )

    if intent_set:
        intent_recall = (
            len(
                intent_set
                & heading_set
            )
            / len(intent_set)
        )
    else:
        intent_recall = (
            full_recall
        )

    intent_bigrams = _bigrams(
        intent_words
    )

    heading_bigrams = _bigrams(
        heading_words
    )

    if intent_bigrams:
        bigram_score = (
            len(
                intent_bigrams
                & heading_bigrams
            )
            / len(intent_bigrams)
        )
    elif intent_words:
        bigram_score = (
            1.0
            if intent_words[0]
            in heading_set
            else 0.0
        )
    else:
        bigram_score = 0.0

    return (
        0.35 * full_recall
        + 0.40 * intent_recall
        + 0.15 * heading_precision
        + 0.10 * bigram_score
    )


def is_generic_condition_query(
    query: str,
    scope_phrase: str | None,
) -> bool:
    """
    Detect queries equivalent to:
        dieu kien + scope

    Example:
        dieu kien ket hon la gi
        dieu kien kinh doanh bat dong san la gi
    """
    query_words = set(
        _token_list(
            query
        )
    )

    scope_words = (
        set(
            _token_list(
                scope_phrase
            )
        )
        if scope_phrase
        else set()
    )

    intent_words = (
        query_words
        - scope_words
    )

    return bool(
        intent_words
    ) and intent_words.issubset(
        {
            "dieu",
            "kien",
        }
    )


def _article_order(
    article_number: str,
) -> int:
    match = re.match(
        r"(\d+)",
        str(article_number).strip(),
    )

    if match:
        return int(
            match.group(1)
        )

    return 999999


def rank_article_headings(
    query: str,
    documents: list[dict[str, Any]],
    *,
    allowed_law_ids: list[str] | None = None,
    scope_phrase: str | None = None,
    top_k: int = 5,
    near_tie_margin: float = 0.03,
) -> list[dict[str, Any]]:
    """
    Rank article headings inside the allowed legal documents.

    One result is returned per article, not per clause.
    """
    if top_k < 1:
        raise ValueError(
            "top_k must be >= 1."
        )

    allowed_set = (
        set(allowed_law_ids)
        if allowed_law_ids
        else None
    )

    article_map: dict[
        tuple[str, str],
        dict[str, Any],
    ] = {}

    for document in documents:
        law_id = str(
            document.get(
                "law_id",
                "",
            )
        ).strip()

        if (
            allowed_set is not None
            and law_id not in allowed_set
        ):
            continue

        article_number = str(
            document.get(
                "article_number",
                "",
            )
        ).strip()

        if not article_number:
            continue

        key = (
            law_id,
            article_number,
        )

        if key in article_map:
            continue

        heading = extract_article_heading(
            str(
                document.get(
                    "text",
                    "",
                )
            ),
            article_number,
        )

        if not heading:
            continue

        score = score_article_heading(
            query=query,
            heading=heading,
            scope_phrase=scope_phrase,
        )

        article_map[key] = {
            "law_id": law_id,
            "article_number": (
                article_number
            ),
            "heading": heading,
            "heading_score": float(
                score
            ),
            "article_order": (
                _article_order(
                    article_number
                )
            ),
        }

    ranked = list(
        article_map.values()
    )

    ranked.sort(
        key=lambda item: (
            item["heading_score"]
        ),
        reverse=True,
    )

    generic = (
        is_generic_condition_query(
            query=query,
            scope_phrase=scope_phrase,
        )
    )

    if generic and ranked:
        best_score = float(
            ranked[0][
                "heading_score"
            ]
        )

        near_ties = [
            item
            for item in ranked
            if (
                best_score
                - float(
                    item[
                        "heading_score"
                    ]
                )
                <= near_tie_margin
            )
        ]

        near_ties.sort(
            key=lambda item: (
                item["article_order"]
            )
        )

        near_keys = {
            (
                item["law_id"],
                item[
                    "article_number"
                ],
            )
            for item in near_ties
        }

        remaining = [
            item
            for item in ranked
            if (
                item["law_id"],
                item[
                    "article_number"
                ],
            )
            not in near_keys
        ]

        ranked = (
            near_ties
            + remaining
        )

    return ranked[:top_k]
