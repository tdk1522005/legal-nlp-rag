from __future__ import annotations

import sys
from pathlib import Path

EVALUATION_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EVALUATION_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chat import is_semantic_result_relevant
from evaluation.evaluate import find_correct_position, load_retrieval_pipeline
from evaluation.evaluate_extended import load_questions
from retrieval.reranker import VietnameseReranker
from retrieval.reranked_router import RerankedTemporalRetrievalRouter

TOP_K = 8
RERANK_CANDIDATE_K = 20
METRIC_KS = (1, 3, 5, 8)


def main() -> None:
    questions = load_questions()

    print()
    print("=" * 90)
    print("BRANCH 9 - RERANKER EVALUATION")
    print("=" * 90)
    print(f"Total questions: {len(questions)}")
    print(f"Reranker candidates: {RERANK_CANDIDATE_K}")

    base_router = load_retrieval_pipeline()
    reranker = VietnameseReranker(
        device="cpu",
        batch_size=8,
        max_length=768,
    )
    router = RerankedTemporalRetrievalRouter(
        base_router,
        reranker,
        candidate_k=RERANK_CANDIDATE_K,
    )

    relevant_count = 0
    hit_counts = {k: 0 for k in METRIC_KS}
    reciprocal_rank_sum = 0.0

    tp = tn = fp = fn = 0

    for index, item in enumerate(questions, start=1):
        question_id = str(item.get("id", index))
        category = str(item.get("category", "unknown"))
        question = str(item["question"]).strip()
        expected_relevant = bool(item.get("expected_relevant", True))

        print("-" * 90)
        print(f"{index}/{len(questions)} | {question_id} | {category}")
        print("QUERY:", question)

        try:
            output = router.retrieve(query=question, top_k=TOP_K)
            results = output.get("results", [])
            accepted = is_semantic_result_relevant(output)

            if expected_relevant:
                if accepted:
                    tp += 1
                else:
                    fn += 1
            else:
                if accepted:
                    fp += 1
                else:
                    tn += 1

            if expected_relevant:
                relevant_count += 1
                expected_law_id = str(item["expected_law_id"])
                expected_article = str(item["expected_article"])
                position = find_correct_position(
                    results,
                    expected_law_id,
                    expected_article,
                )

                if position is not None:
                    for k in METRIC_KS:
                        if position <= k:
                            hit_counts[k] += 1
                    reciprocal_rank_sum += 1.0 / position

                print("CORRECT POSITION:", position or "NOT FOUND")

            if results:
                top = results[0]
                print(
                    "TOP-1:",
                    top.get("metadata", {}).get("citation", "-"),
                    "| rerank_score=",
                    f"{float(top.get('rerank_score', 0.0)):.4f}",
                )
            print("GATE:", "ACCEPT" if accepted else "REJECT")

        except Exception as error:
            print("ERROR:", error)
            if expected_relevant:
                fn += 1
            else:
                tn += 1

    accuracy = (tp + tn) / len(questions) if questions else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    mrr = reciprocal_rank_sum / relevant_count if relevant_count else 0.0

    print()
    print("=" * 90)
    print("RESULTS")
    print("=" * 90)
    print(f"Total questions : {len(questions)}")
    print(f"Relevant        : {relevant_count}")
    print(f"OOD             : {len(questions) - relevant_count}")
    for k in METRIC_KS:
        value = hit_counts[k] / relevant_count if relevant_count else 0.0
        print(f"Hit@{k:<2}          : {value * 100:6.2f}% ({hit_counts[k]}/{relevant_count})")
    print(f"MRR@8           : {mrr:.4f}")
    print()
    print(f"Accuracy        : {accuracy * 100:.2f}%")
    print(f"Precision       : {precision * 100:.2f}%")
    print(f"Recall          : {recall * 100:.2f}%")
    print()
    print(f"TP={tp}  TN={tn}  FP={fp}  FN={fn}")
    print(f"OOD rejection   : {tn / (tn + fp) * 100:.2f}%" if (tn + fp) else "OOD rejection   : -")


if __name__ == "__main__":
    main()
