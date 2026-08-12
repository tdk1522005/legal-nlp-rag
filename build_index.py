from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


from models.qwen_embedding import QwenEmbedding
from vectorstore.faiss_store import FaissStore


PROJECT_ROOT = Path(__file__).resolve().parent

DEFAULT_CORPUS = (
    PROJECT_ROOT
    / "data"
    / "chunks"
    / "default_retrieval_corpus.jsonl"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "index"
    / "legal_dense"
)


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def load_jsonl(
    path: Path,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy corpus: {path}"
        )

    rows: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8-sig") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"JSONL lỗi tại dòng {line_number}: {error}"
                ) from error

            if not isinstance(row, dict):
                raise ValueError(
                    f"Dòng {line_number} không phải object."
                )

            rows.append(row)

            if limit is not None and len(rows) >= limit:
                break

    if not rows:
        raise ValueError("Corpus không có chunk nào.")

    return rows


def validate_rows(rows: list[dict[str, Any]]) -> None:
    chunk_ids: set[str] = set()

    for index, row in enumerate(rows, start=1):
        chunk_id = str(row.get("chunk_id", "")).strip()
        text = str(row.get("text", "")).strip()
        law_id = str(row.get("law_id", "")).strip()

        if not chunk_id:
            raise ValueError(
                f"Chunk thứ {index} thiếu chunk_id."
            )

        if chunk_id in chunk_ids:
            raise ValueError(
                f"chunk_id bị trùng: {chunk_id}"
            )

        if not text:
            raise ValueError(
                f"{chunk_id}: trường text đang rỗng."
            )

        if not law_id:
            raise ValueError(
                f"{chunk_id}: thiếu law_id."
            )

        chunk_ids.add(chunk_id)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
        )
        file.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Tạo dense FAISS index từ legal corpus JSONL."
        )
    )

    parser.add_argument(
        "--corpus",
        default=str(DEFAULT_CORPUS),
        help="Đường dẫn corpus JSONL.",
    )

    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Thư mục lưu index và metadata.",
    )

    parser.add_argument(
        "--model-name",
        default="Qwen/Qwen3-Embedding-0.6B",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--dimension",
        type=int,
        default=1024,
        help="Qwen embedding dimension.",
    )



    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Chỉ build N chunk để kiểm thử.",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Cho phép ghi đè index đã tồn tại.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    corpus_path = resolve_path(args.corpus)
    output_dir = resolve_path(args.output_dir)

    index_path = output_dir / "legal_dense.index"
    metadata_path = output_dir / "chunk_metadata.jsonl"
    manifest_path = output_dir / "index_manifest.json"

    existing_files = [
        path
        for path in (
            index_path,
            metadata_path,
            manifest_path,
        )
        if path.exists()
    ]

    if existing_files and not args.force:
        joined = "\n".join(
            f"  - {path}"
            for path in existing_files
        )
        raise FileExistsError(
            "Index đã tồn tại. Dùng --force để ghi đè:\n"
            f"{joined}"
        )

    print("=" * 80)
    print("BUILD LEGAL DENSE INDEX")
    print("=" * 80)
    print(f"Corpus       : {corpus_path}")
    print(f"Output       : {output_dir}")
    print(f"Model        : {args.model_name}")
    print(f"Batch size   : {args.batch_size}")

    rows = load_jsonl(corpus_path, limit=args.limit)
    validate_rows(rows)

    law_counts = Counter(
        str(row["law_id"])
        for row in rows
    )

    texts = [str(row["text"]) for row in rows]

    print("-" * 80)
    print(f"Số chunk     : {len(rows)}")
    print(f"Số law_id    : {len(law_counts)}")

    for law_id, count in sorted(law_counts.items()):
        print(f"  - {law_id}: {count}")

    print("-" * 80)
    print("Initializing Qwen Embedding...")

    embedding_model = QwenEmbedding(
        model_name=args.model_name,
        device="cpu",
        dimension=args.dimension,
        batch_size=args.batch_size,
    )

    print("Đang tạo dense embedding...")
    embeddings = embedding_model.encode_documents(texts)

    if embeddings.shape[0] != len(rows):
        raise RuntimeError(
            "Số embedding không khớp số chunk."
        )

    vector_norms = np.linalg.norm(embeddings, axis=1)

    if not np.allclose(vector_norms, 1.0, atol=1e-4):
        raise RuntimeError(
            "Embedding chưa được chuẩn hóa L2."
        )

    dimension = int(embeddings.shape[1])

    print(f"Embedding     : {embeddings.shape}")
    print("Đang tạo FAISS IndexFlatIP...")

    store = FaissStore(dimension=dimension)
    store.add(
        embeddings=embeddings,
        documents=rows,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    store.save(
        index_path=index_path,
        metadata_path=metadata_path,
    )

    manifest = {
        "schema_version": "1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "corpus_file": str(corpus_path),
        "corpus_sha256": sha256_file(corpus_path),
        "model_name": args.model_name,
        "embedding_provider": "qwen",
        "embedding_mode": "dense",
        "dimension": dimension,
        "vector_count": int(store.index.ntotal),
        "metadata_count": len(store.documents),
        "normalized_l2": True,
        "faiss_index_type": "IndexFlatIP",
        "similarity": "cosine",
        "batch_size": args.batch_size,
        "limited_build": args.limit is not None,
        "limit": args.limit,
        "law_counts": dict(sorted(law_counts.items())),
        "files": {
            "index": index_path.name,
            "metadata": metadata_path.name,
            "manifest": manifest_path.name,
        },
    }

    write_json(manifest_path, manifest)

    print("=" * 80)
    print("BUILD THÀNH CÔNG")
    print("=" * 80)
    print(f"Vector count : {store.index.ntotal}")
    print(f"Dimension    : {dimension}")
    print(f"FAISS index  : {index_path}")
    print(f"Metadata     : {metadata_path}")
    print(f"Manifest     : {manifest_path}")


if __name__ == "__main__":
    main()
