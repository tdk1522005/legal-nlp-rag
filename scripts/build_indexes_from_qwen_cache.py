from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vectorstore.faiss_store import FaissStore


CACHE_DIR = (
    PROJECT_ROOT
    / "data"
    / "embeddings"
    / "qwen3_embedding_0_6b_1024"
)

CURRENT_CORPUS = (
    PROJECT_ROOT
    / "data"
    / "chunks"
    / "default_retrieval_corpus.jsonl"
)

TEMPORAL_CORPUS = (
    PROJECT_ROOT
    / "data"
    / "chunks"
    / "legal_corpus.jsonl"
)

CURRENT_OUTPUT = (
    PROJECT_ROOT
    / "index"
    / "legal_dense_qwen"
)

TEMPORAL_OUTPUT = (
    PROJECT_ROOT
    / "index"
    / "legal_temporal_qwen"
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            block = f.read(1024 * 1024)

            if not block:
                break

            h.update(block)

    return h.hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    rows = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            rows.append(
                json.loads(line)
            )

    return rows


def get_chunk_id(row: dict) -> str:
    chunk_id = (
        row.get("chunk_id")
        or row.get("id")
    )

    if not chunk_id:
        raise ValueError(
            "Chunk does not contain chunk_id/id."
        )

    return str(chunk_id)


def write_json(
    path: Path,
    data: dict,
) -> None:
    path.write_text(
        json.dumps(
            data,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def build_index(
    *,
    corpus_path: Path,
    output_dir: Path,
    cache_embeddings: np.ndarray,
    cache_id_to_position: dict[str, int],
    cache_manifest: dict,
) -> None:

    rows = load_jsonl(
        corpus_path
    )

    if not rows:
        raise ValueError(
            f"Corpus is empty: {corpus_path}"
        )

    chunk_ids = [
        get_chunk_id(row)
        for row in rows
    ]

    if len(set(chunk_ids)) != len(chunk_ids):
        raise ValueError(
            f"Duplicate chunk IDs in {corpus_path}"
        )

    missing = [
        chunk_id
        for chunk_id in chunk_ids
        if chunk_id not in cache_id_to_position
    ]

    if missing:
        raise RuntimeError(
            "Some corpus chunks do not exist "
            f"in embedding cache. Missing={len(missing)}"
        )

    positions = np.asarray(
        [
            cache_id_to_position[chunk_id]
            for chunk_id in chunk_ids
        ],
        dtype=np.int64,
    )

    embeddings = np.asarray(
        cache_embeddings[positions],
        dtype=np.float32,
    )

    dimension = int(
        embeddings.shape[1]
    )

    expected_dimension = int(
        cache_manifest["dimension"]
    )

    if dimension != expected_dimension:
        raise RuntimeError(
            "Dimension mismatch: "
            f"{dimension} != {expected_dimension}"
        )

    norms = np.linalg.norm(
        embeddings,
        axis=1,
    )

    if not np.allclose(
        norms,
        1.0,
        atol=1e-4,
    ):
        raise RuntimeError(
            "Embeddings are not L2 normalized."
        )

    law_counts = Counter(
        str(row.get("law_id", "UNKNOWN"))
        for row in rows
    )

    print()
    print("=" * 80)
    print("BUILD INDEX FROM QWEN CACHE")
    print("=" * 80)
    print(f"Corpus       : {corpus_path}")
    print(f"Output       : {output_dir}")
    print(f"Chunks       : {len(rows)}")
    print(f"Dimension    : {dimension}")
    print()

    for law_id, count in sorted(
        law_counts.items()
    ):
        print(
            f"- {law_id}: {count}"
        )

    print()
    print("Creating FAISS IndexFlatIP...")

    store = FaissStore(
        dimension=dimension
    )

    store.add(
        embeddings=embeddings,
        documents=rows,
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    index_path = (
        output_dir
        / "legal_dense.index"
    )

    metadata_path = (
        output_dir
        / "chunk_metadata.jsonl"
    )

    manifest_path = (
        output_dir
        / "index_manifest.json"
    )

    store.save(
        index_path=index_path,
        metadata_path=metadata_path,
    )

    manifest = {
        "schema_version": "1.0",
        "created_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "corpus_file": str(
            corpus_path
        ),
        "corpus_sha256": sha256_file(
            corpus_path
        ),
        "model_name": cache_manifest[
            "model_name"
        ],
        "embedding_provider": "qwen",
        "embedding_mode": "dense",
        "dimension": dimension,
        "vector_count": int(
            store.index.ntotal
        ),
        "metadata_count": len(
            store.documents
        ),
        "normalized_l2": True,
        "faiss_index_type": "IndexFlatIP",
        "similarity": "cosine",
        "limited_build": False,
        "limit": None,
        "law_counts": dict(
            sorted(
                law_counts.items()
            )
        ),
        "embedding_cache": str(
            CACHE_DIR
        ),
        "embedding_cache_model": (
            cache_manifest["model_name"]
        ),
        "embedding_cache_count": (
            cache_manifest["vector_count"]
        ),
        "files": {
            "index": index_path.name,
            "metadata": metadata_path.name,
            "manifest": manifest_path.name,
        },
    }

    write_json(
        manifest_path,
        manifest,
    )

    print()
    print("BUILD COMPLETE")
    print(
        f"Vector count : {store.index.ntotal}"
    )
    print(
        f"Dimension    : {dimension}"
    )
    print(
        f"FAISS index  : {index_path}"
    )
    print(
        f"Metadata     : {metadata_path}"
    )
    print(
        f"Manifest     : {manifest_path}"
    )


def main() -> None:

    embeddings_path = (
        CACHE_DIR
        / "embeddings.npy"
    )

    ids_path = (
        CACHE_DIR
        / "chunk_ids.json"
    )

    manifest_path = (
        CACHE_DIR
        / "cache_manifest.json"
    )

    for path in [
        embeddings_path,
        ids_path,
        manifest_path,
        CURRENT_CORPUS,
        TEMPORAL_CORPUS,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                path
            )

    cache_manifest = json.loads(
        manifest_path.read_text(
            encoding="utf-8"
        )
    )

    if (
        cache_manifest.get("complete")
        is not True
    ):
        raise RuntimeError(
            "Embedding cache is incomplete."
        )

    cache_ids = json.loads(
        ids_path.read_text(
            encoding="utf-8"
        )
    )

    cache_embeddings = np.load(
        embeddings_path,
        mmap_mode="r",
    )

    expected_shape = (
        len(cache_ids),
        int(
            cache_manifest["dimension"]
        ),
    )

    if (
        cache_embeddings.shape
        != expected_shape
    ):
        raise RuntimeError(
            "Cache shape mismatch: "
            f"{cache_embeddings.shape} "
            f"!= {expected_shape}"
        )

    if (
        len(set(cache_ids))
        != len(cache_ids)
    ):
        raise RuntimeError(
            "Duplicate chunk IDs in cache."
        )

    cache_id_to_position = {
        chunk_id: position
        for position, chunk_id
        in enumerate(cache_ids)
    }

    print("=" * 80)
    print("QWEN CACHE")
    print("=" * 80)
    print(
        f"Vectors      : {cache_embeddings.shape[0]}"
    )
    print(
        f"Dimension    : {cache_embeddings.shape[1]}"
    )
    print(
        f"Model        : {cache_manifest['model_name']}"
    )

    build_index(
        corpus_path=CURRENT_CORPUS,
        output_dir=CURRENT_OUTPUT,
        cache_embeddings=cache_embeddings,
        cache_id_to_position=cache_id_to_position,
        cache_manifest=cache_manifest,
    )

    build_index(
        corpus_path=TEMPORAL_CORPUS,
        output_dir=TEMPORAL_OUTPUT,
        cache_embeddings=cache_embeddings,
        cache_id_to_position=cache_id_to_position,
        cache_manifest=cache_manifest,
    )

    print()
    print("=" * 80)
    print("ALL QWEN INDEXES BUILT")
    print("=" * 80)
    print(
        f"Current  : {CURRENT_OUTPUT}"
    )
    print(
        f"Temporal : {TEMPORAL_OUTPUT}"
    )


if __name__ == "__main__":
    main()
