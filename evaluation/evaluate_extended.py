from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from chat import is_semantic_result_relevant
from evaluation.evaluate import (
    load_retrieval_pipeline,
    find_correct_position,
)


BASE_DIR = Path(__file__).resolve().parent
TEST_FILE = BASE_DIR / "test_questions_extended.json"

MAX_RESULTS = 8
METRIC_KS = (1, 3, 5, 8)


def load_questions() -> list[dict[str, Any]]:
    with TEST_FILE.open(
        "r",
        encoding="utf-8-sig",
    ) as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError(
            "Extended test file must contain a list."
        )

    return data


def format_score(
    value: Any,
) -> str:
    if value is None:
        return "-"

    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def main() -> None:
    questions = load_questions()

    print()
    print("=" * 90)
    print("EXTENDED RAG EVALUATION")
    print("=" * 90)
    print(
        "Total questions:",
        len(questions),
    )

    router = load_retrieval_pipeline()

    # -----------------------------------------------------
    # Ranking metrics for relevant questions only
    # -----------------------------------------------------

    relevant_count = 0

    hit_counts = {
        k: 0
        for k in METRIC_KS
    }

    reciprocal_rank_sum = 0.0

    # -----------------------------------------------------
    # Relevance gate confusion matrix
    # -----------------------------------------------------

    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0

    # -----------------------------------------------------
    # Metrics per category
    # -----------------------------------------------------

    category_stats = defaultdict(
        lambda: {
            "count": 0,
            "relevant": 0,
            "gate_correct": 0,
            "top1": 0,
            "hit8": 0,
            "rr_sum": 0.0,
        }
    )

    failed_cases = []

    for index, item in enumerate(
        questions,
        start=1,
    ):
        question_id = str(
            item.get("id", index)
        )

        category = str(
            item.get(
                "category",
                "unknown",
            )
        )

        question = str(
            item["question"]
        ).strip()

        expected_relevant = bool(
            item.get(
                "expected_relevant",
                True,
            )
        )

        stats = category_stats[
            category
        ]

        stats["count"] += 1

        print()
        print("-" * 90)
        print(
            f"{index}/{len(questions)} "
            f"| {question_id} "
            f"| {category}"
        )
        print(
            "QUERY:",
            question,
        )

        try:
            output = router.retrieve(
                query=question,
                top_k=MAX_RESULTS,
            )

            results = output.get(
                "results",
                [],
            )

            accepted = (
                is_semantic_result_relevant(
                    output
                )
            )

            gate_correct = (
                accepted
                == expected_relevant
            )

            if gate_correct:
                stats["gate_correct"] += 1

            # -------------------------------------------------
            # Gate confusion matrix
            # -------------------------------------------------

            if expected_relevant:
                if accepted:
                    true_positive += 1
                else:
                    false_negative += 1
            else:
                if accepted:
                    false_positive += 1
                else:
                    true_negative += 1

            print(
                "EXPECTED GATE:",
                (
                    "ACCEPT"
                    if expected_relevant
                    else "REJECT"
                ),
            )

            print(
                "ACTUAL GATE:",
                (
                    "ACCEPT"
                    if accepted
                    else "REJECT"
                ),
            )

            print(
                "MODE:",
                output.get(
                    "retrieval_mode",
                    "-",
                ),
            )

            print(
                "SCOPE:",
                output.get(
                    "law_scope",
                    "-",
                ),
            )

            heading_match = output.get(
                "article_heading_match"
            )

            if heading_match:
                print(
                    "HEADING MATCH:",
                    "Article",
                    heading_match.get(
                        "article_number"
                    ),
                    "| score =",
                    format_score(
                        heading_match.get(
                            "heading_score"
                        )
                    ),
                )

            # -------------------------------------------------
            # OOD / non-relevant case
            # -------------------------------------------------

            if not expected_relevant:
                if gate_correct:
                    print(
                        "RESULT: PASS "
                        "(correctly rejected)"
                    )
                else:
                    print(
                        "RESULT: FAIL "
                        "(false accept)"
                    )

                    failed_cases.append(
                        (
                            question_id,
                            category,
                            "FALSE_ACCEPT",
                            question,
                        )
                    )

                continue

            # -------------------------------------------------
            # Relevant retrieval case
            # -------------------------------------------------

            relevant_count += 1
            stats["relevant"] += 1

            expected_law_id = str(
                item["expected_law_id"]
            ).strip()

            expected_article = str(
                item["expected_article"]
            ).strip()

            position = find_correct_position(
                results=results,
                expected_law_id=(
                    expected_law_id
                ),
                expected_article=(
                    expected_article
                ),
            )

            print(
                "EXPECTED:",
                expected_law_id,
                "- Article",
                expected_article,
            )

            if results:
                top = results[0]

                metadata = top.get(
                    "metadata",
                    {},
                )

                print(
                    "TOP RESULT:",
                    metadata.get(
                        "law_id",
                        "-",
                    ),
                    "- Article",
                    metadata.get(
                        "article_number",
                        "-",
                    ),
                )

                print(
                    "SEM SCORE:",
                    format_score(
                        top.get("score")
                    ),
                )

                print(
                    "HEAD SCORE:",
                    format_score(
                        top.get(
                            "heading_score"
                        )
                    ),
                )

                print(
                    "SOURCE:",
                    top.get(
                        "retrieval_source",
                        "semantic",
                    ),
                )
            else:
                print(
                    "TOP RESULT: NONE"
                )

            if position is not None:
                reciprocal_rank_sum += (
                    1.0 / position
                )

                stats["rr_sum"] += (
                    1.0 / position
                )

                for k in METRIC_KS:
                    if position <= k:
                        hit_counts[k] += 1

                if position == 1:
                    stats["top1"] += 1

                if position <= 8:
                    stats["hit8"] += 1

                print(
                    "CORRECT ARTICLE RANK:",
                    position,
                )
            else:
                print(
                    "CORRECT ARTICLE RANK:",
                    "NOT FOUND",
                )

            case_pass = (
                accepted
                and position == 1
            )

            if case_pass:
                print(
                    "RESULT: PASS"
                )
            else:
                print(
                    "RESULT: FAIL"
                )

                reason = []

                if not accepted:
                    reason.append(
                        "FALSE_REJECT"
                    )

                if position != 1:
                    reason.append(
                        "NOT_TOP1"
                    )

                failed_cases.append(
                    (
                        question_id,
                        category,
                        "+".join(reason),
                        question,
                    )
                )

        except Exception as error:
            print(
                "ERROR:",
                repr(error),
            )

            failed_cases.append(
                (
                    question_id,
                    category,
                    "ERROR",
                    question,
                )
            )

    # =====================================================
    # Ket qua danh gia
    # =====================================================

    total_gate = (
        true_positive
        + true_negative
        + false_positive
        + false_negative
    )

    gate_accuracy = (
        (true_positive + true_negative) / total_gate
        if total_gate
        else 0.0
    )

    gate_precision = (
        true_positive
        / (true_positive + false_positive)
        if (true_positive + false_positive)
        else 0.0
    )

    gate_recall = (
        true_positive
        / (true_positive + false_negative)
        if (true_positive + false_negative)
        else 0.0
    )

    mrr = (
        reciprocal_rank_sum / relevant_count
        if relevant_count
        else 0.0
    )

    total_questions = total_gate
    out_of_domain_count = (
        true_negative + false_positive
    )

    print()
    print("=" * 72)
    print(
        "\u0110\u00c1NH GI\u00c1 H\u1ec6 TH\u1ed0NG RAG PH\u00c1P LU\u1eacT"
    )
    print("=" * 72)

    print()
    print(
        f"T\u1ed5ng s\u1ed1 c\u00e2u ki\u1ec3m th\u1eed : "
        f"{total_questions}"
    )
    print(
        f"C\u00e2u c\u00f3 \u0111\u00e1p \u00e1n      : "
        f"{relevant_count}"
    )
    print(
        f"C\u00e2u ngo\u00e0i ph\u1ea1m vi   : "
        f"{out_of_domain_count}"
    )

    print()
    print("-" * 72)
    print(
        "K\u1ebeT QU\u1ea2 TRUY XU\u1ea4T"
    )
    print("-" * 72)

    for k in METRIC_KS:
        value = (
            hit_counts[k] / relevant_count
            if relevant_count
            else 0.0
        )

        print(
            f"Hit@{k:<2} : "
            f"{value * 100:6.2f}% "
            f"({hit_counts[k]}/{relevant_count})"
        )

    print(
        f"MRR@8  : {mrr:.4f}"
    )

    print()
    print("-" * 72)
    print(
        "K\u1ebeT QU\u1ea2 KI\u1ec2M SO\u00c1T C\u00c2U H\u1eceI NGO\u00c0I PH\u1ea0M VI"
    )
    print("-" * 72)

    print(
        f"Accuracy  : {gate_accuracy * 100:6.2f}%"
    )
    print(
        f"Precision : {gate_precision * 100:6.2f}%"
    )
    print(
        f"Recall    : {gate_recall * 100:6.2f}%"
    )

    print()
    print("-" * 72)
    print(
        "MA TR\u1eacN K\u1ebeT QU\u1ea2"
    )
    print("-" * 72)

    print(
        f"True Positive  : {true_positive}"
    )
    print(
        f"True Negative  : {true_negative}"
    )
    print(
        f"False Positive : {false_positive}"
    )
    print(
        f"False Negative : {false_negative}"
    )

    print()
    print("-" * 72)
    print(
        "T\u1ed4NG K\u1ebeT"
    )
    print("-" * 72)

    print(
        f"Truy xu\u1ea5t \u0111\u00fang ngay Top-1 : "
        f"{hit_counts[1]}/{relevant_count}"
    )

    print(
        f"T\u1eeb ch\u1ed1i \u0111\u00fang c\u00e2u ngo\u00e0i ph\u1ea1m vi : "
        f"{true_negative}/{out_of_domain_count}"
    )

    print(
        f"S\u1ed1 tr\u01b0\u1eddng h\u1ee3p l\u1ed7i : "
        f"{len(failed_cases)}"
    )

    print("=" * 72)

    if failed_cases:
        print()
        print(
            "C\u00c1C TR\u01af\u1edcNG H\u1ee2P CH\u01afA \u0110\u1ea0T"
        )
        print("-" * 72)

        for item in failed_cases:
            print(item)


if __name__ == "__main__":
    main()
