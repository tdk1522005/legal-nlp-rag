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
    # Global ranking metrics
    # =====================================================

    print()
    print("=" * 90)
    print("GLOBAL RANKING METRICS")
    print("=" * 90)

    print(
        "Relevant questions:",
        relevant_count,
    )

    for k in METRIC_KS:
        value = (
            hit_counts[k]
            / relevant_count
            if relevant_count
            else 0.0
        )

        print(
            f"Hit@{k:<2}: "
            f"{value * 100:.2f}% "
            f"({hit_counts[k]}/"
            f"{relevant_count})"
        )

    mrr = (
        reciprocal_rank_sum
        / relevant_count
        if relevant_count
        else 0.0
    )

    print(
        f"MRR@8 : {mrr:.4f}"
    )

    # =====================================================
    # Relevance gate metrics
    # =====================================================

    print()
    print("=" * 90)
    print("RELEVANCE GATE")
    print("=" * 90)

    total_gate = (
        true_positive
        + true_negative
        + false_positive
        + false_negative
    )

    gate_accuracy = (
        (
            true_positive
            + true_negative
        )
        / total_gate
        if total_gate
        else 0.0
    )

    precision = (
        true_positive
        / (
            true_positive
            + false_positive
        )
        if (
            true_positive
            + false_positive
        )
        else 0.0
    )

    recall = (
        true_positive
        / (
            true_positive
            + false_negative
        )
        if (
            true_positive
            + false_negative
        )
        else 0.0
    )

    print(
        "True Positive :",
        true_positive,
    )
    print(
        "True Negative :",
        true_negative,
    )
    print(
        "False Positive:",
        false_positive,
    )
    print(
        "False Negative:",
        false_negative,
    )

    print(
        f"Gate Accuracy : "
        f"{gate_accuracy * 100:.2f}%"
    )

    print(
        f"Gate Precision: "
        f"{precision * 100:.2f}%"
    )

    print(
        f"Gate Recall   : "
        f"{recall * 100:.2f}%"
    )

    # =====================================================
    # Category metrics
    # =====================================================

    print()
    print("=" * 90)
    print("METRICS BY CATEGORY")
    print("=" * 90)

    for category, stats in (
        category_stats.items()
    ):
        count = stats["count"]

        gate_acc = (
            stats["gate_correct"]
            / count
            if count
            else 0.0
        )

        print()
        print(
            "CATEGORY:",
            category,
        )

        print(
            "Questions:",
            count,
        )

        print(
            f"Gate accuracy: "
            f"{gate_acc * 100:.2f}% "
            f"({stats['gate_correct']}/"
            f"{count})"
        )

        relevant = stats[
            "relevant"
        ]

        if relevant:
            top1 = (
                stats["top1"]
                / relevant
            )

            hit8 = (
                stats["hit8"]
                / relevant
            )

            category_mrr = (
                stats["rr_sum"]
                / relevant
            )

            print(
                f"Hit@1: "
                f"{top1 * 100:.2f}% "
                f"({stats['top1']}/"
                f"{relevant})"
            )

            print(
                f"Hit@8: "
                f"{hit8 * 100:.2f}% "
                f"({stats['hit8']}/"
                f"{relevant})"
            )

            print(
                f"MRR@8: "
                f"{category_mrr:.4f}"
            )

    # =====================================================
    # Failed cases
    # =====================================================

    print()
    print("=" * 90)
    print("FAILED CASES")
    print("=" * 90)

    if not failed_cases:
        print(
            "NONE"
        )
    else:
        for (
            question_id,
            category,
            reason,
            question,
        ) in failed_cases:
            print(
                f"{question_id} | "
                f"{category} | "
                f"{reason}"
            )
            print(
                "  ",
                question,
            )


if __name__ == "__main__":
    main()
