from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.qwen_embedding import QwenEmbedding

DEFAULT_CORPUS = (
    PROJECT_ROOT
    / "data"
    / "chunks"
    / "legal_corpus.jsonl"
)

DEFAULT_CACHE_DIR = (
    PROJECT_ROOT
    / "data"
    / "embeddings"
    / "qwen3_embedding_0_6b_1024"
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


def load_corpus(path: Path) -> list[dict]:
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
            "Chunk is missing chunk_id/id."
        )

    return str(chunk_id)


def get_embedding_text(row: dict) -> str:
    text = (
        row.get("text")
        or row.get("content")
    )

    if not text:
        raise ValueError(
            f"Chunk {get_chunk_id(row)} has no text."
        )

    return str(text)


def atomic_write_json(
    path: Path,
    data: dict,
) -> None:
    temp_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    temp_path.write_text(
        json.dumps(
            data,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    temp_path.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--corpus",
        type=Path,
        default=DEFAULT_CORPUS,
    )

    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
    )

    parser.add_argument(
        "--model-name",
        default="Qwen/Qwen3-Embedding-0.6B",
    )

    parser.add_argument(
        "--dimension",
        type=int,
        default=1024,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--force",
        action="store_true",
    )

    args = parser.parse_args()

    corpus_path = args.corpus.resolve()
    cache_dir = args.cache_dir.resolve()

    if not corpus_path.exists():
        raise FileNotFoundError(
            corpus_path
        )

    rows = load_corpus(
        corpus_path
    )

    if args.limit is not None:
        rows = rows[:args.limit]

    if not rows:
        raise ValueError(
            "Corpus is empty."
        )

    cache_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    embeddings_path = (
        cache_dir
        / "embeddings.npy"
    )

    chunk_ids_path = (
        cache_dir
        / "chunk_ids.json"
    )

    progress_path = (
        cache_dir
        / "progress.json"
    )

    manifest_path = (
        cache_dir
        / "cache_manifest.json"
    )

    corpus_hash = sha256_file(
        corpus_path
    )

    chunk_ids = [
        get_chunk_id(row)
        for row in rows
    ]

    total = len(rows)

    if len(set(chunk_ids)) != total:
        raise ValueError(
            "Duplicate chunk IDs found."
        )

    if args.force:
        for path in [
            embeddings_path,
            chunk_ids_path,
            progress_path,
            manifest_path,
        ]:
            if path.exists():
                path.unlink()

    start_index = 0

    if progress_path.exists():
        progress = json.loads(
            progress_path.read_text(
                encoding="utf-8"
            )
        )

        if (
            progress.get("model_name")
            != args.model_name
        ):
            raise RuntimeError(
                "Model mismatch in checkpoint."
            )

        if (
            int(progress.get("dimension", -1))
            != args.dimension
        ):
            raise RuntimeError(
                "Dimension mismatch in checkpoint."
            )

        if (
            progress.get("corpus_sha256")
            != corpus_hash
        ):
            raise RuntimeError(
                "Corpus changed since checkpoint."
            )

        if (
            int(progress.get("total", -1))
            != total
        ):
            raise RuntimeError(
                "Corpus size mismatch in checkpoint."
            )

        start_index = int(
            progress.get(
                "completed",
                0,
            )
        )

    if embeddings_path.exists():
        embeddings = np.load(
            embeddings_path,
            mmap_mode="r+",
        )

        expected_shape = (
            total,
            args.dimension,
        )

        if embeddings.shape != expected_shape:
            raise RuntimeError(
                "Embedding cache shape mismatch: "
                f"{embeddings.shape} != "
                f"{expected_shape}"
            )

    else:
        embeddings = (
            np.lib.format.open_memmap(
                embeddings_path,
                mode="w+",
                dtype=np.float32,
                shape=(
                    total,
                    args.dimension,
                ),
            )
        )

    if start_index > total:
        raise RuntimeError(
            "Invalid checkpoint position."
        )

    chunk_ids_path.write_text(
        json.dumps(
            chunk_ids,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("=" * 80)
    print("QWEN EMBEDDING CACHE")
    print("=" * 80)
    print(f"Corpus       : {corpus_path}")
    print(f"Cache        : {cache_dir}")
    print(f"Chunks       : {total}")
    print(f"Dimension    : {args.dimension}")
    print(f"Batch size   : {args.batch_size}")
    print(f"Resume from  : {start_index}")
    print()

    if start_index == total:
        print("CACHE_ALREADY_COMPLETE")
        return

    model = QwenEmbedding(
        model_name=args.model_name,
        device="cpu",
        dimension=args.dimension,
        batch_size=args.batch_size,
    )

    for start in range(
        start_index,
        total,
        args.batch_size,
    ):
        end = min(
            start + args.batch_size,
            total,
        )

        texts = [
            get_embedding_text(row)
            for row in rows[start:end]
        ]

        vectors = model.encode_documents(
            texts
        )

        if vectors.shape != (
            end - start,
            args.dimension,
        ):
            raise RuntimeError(
                "Unexpected batch shape: "
                f"{vectors.shape}"
            )

        embeddings[start:end] = vectors
        embeddings.flush()

        progress = {
            "model_name": args.model_name,
            "dimension": args.dimension,
            "batch_size": args.batch_size,
            "corpus": str(corpus_path),
            "corpus_sha256": corpus_hash,
            "total": total,
            "completed": end,
        }

        atomic_write_json(
            progress_path,
            progress,
        )

        percent = (
            end / total
        ) * 100

        print(
            f"[{end}/{total}] "
            f"{percent:.2f}%"
        )

    manifest = {
        "model_name": args.model_name,
        "embedding_provider": "qwen",
        "embedding_mode": "dense",
        "dimension": args.dimension,
        "normalized_l2": True,
        "dtype": "float32",
        "vector_count": total,
        "corpus": str(corpus_path),
        "corpus_sha256": corpus_hash,
        "embeddings_file": str(
            embeddings_path
        ),
        "chunk_ids_file": str(
            chunk_ids_path
        ),
        "complete": True,
    }

    atomic_write_json(
        manifest_path,
        manifest,
    )

    print()
    print("=" * 80)
    print("CACHE BUILD COMPLETE")
    print("=" * 80)
    print(f"Vector count : {total}")
    print(f"Dimension    : {args.dimension}")
    print(f"Embeddings   : {embeddings_path}")
    print(f"Manifest     : {manifest_path}")


if __name__ == "__main__":
    main()
