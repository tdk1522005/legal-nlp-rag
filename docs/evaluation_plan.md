# Evaluation Plan

The current repository does **not** yet contain an automated evaluation benchmark. This document defines the next evaluation steps without presenting unverified scores.

## 1. Build a legal QA evaluation set

Create a small curated set of Vietnamese Civil Code questions. Each example should contain:

```json
{
  "question": "...",
  "relevant_passage_ids": ["..."],
  "reference_answer": "..."
}
```

The initial goal is not dataset size but reliable labels that make retrieval errors easy to inspect.

## 2. Retrieval evaluation

For each evaluation question, retrieve the top-k chunks and compare them with the labeled relevant passages.

Recommended metrics:

- **Hit@K**: whether at least one relevant passage appears in the top-k results.
- **Recall@K**: fraction of known relevant passages retrieved in the top-k results.
- **MRR (Mean Reciprocal Rank)**: rewards systems that rank the first relevant result near the top.

Suggested values of `k`:

```text
K = 1, 3, 5, 10
```

## 3. Error analysis

For retrieval failures, inspect:

- chunk boundaries
- ambiguous legal terminology
- overly broad or overly specific questions
- missing source passages
- semantic similarity between confusing articles

The purpose is to identify whether errors come from document preprocessing, chunking, embeddings or ranking.

## 4. Answer-level evaluation

After retrieval is measured independently, evaluate generated answers on:

- **Groundedness**: whether claims are supported by retrieved context.
- **Relevance**: whether the answer addresses the question directly.
- **Citation correctness**: whether cited passages actually support the answer.
- **Completeness**: whether important retrieved legal information is omitted.

A small human-reviewed rubric is preferable to reporting an automated score without validation.

## 5. Future comparison experiments

Once a baseline is established, compare the current dense retriever against:

- alternative chunk sizes / overlaps
- different embedding models
- hybrid lexical + dense retrieval
- reranking after initial FAISS retrieval

All future README metrics should be copied from reproducible evaluation outputs rather than entered manually.
