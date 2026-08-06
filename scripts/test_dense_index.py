from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from models.bge_embedding import BGEEmbedding
from vectorstore.faiss_store import FaissStore
DEFAULT_INDEX_DIR = PROJECT_ROOT / "index" / "legal_dense"


def load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy manifest: {path}"
        )

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Kiểm tra legal dense FAISS index."
    )

    parser.add_argument(
        "--query",
        default="Điều kiện để giao dịch dân sự có hiệu lực là gì?",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--index-dir",
        default=str(DEFAULT_INDEX_DIR),
    )

    parser.add_argument(
        "--law-id",
        default=None,
        help="Lọc kết quả theo một law_id.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--use-fp16",
        action="store_true",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    index_dir = Path(args.index_dir)

    if not index_dir.is_absolute():
        index_dir = PROJECT_ROOT / index_dir

    manifest_path = index_dir / "index_manifest.json"
    index_path = index_dir / "legal_dense.index"
    metadata_path = index_dir / "chunk_metadata.jsonl"

    manifest = load_manifest(manifest_path)

    embedding_model = BGEEmbedding(
        model_name=manifest["model_name"],
        use_fp16=args.use_fp16,
        batch_size=args.batch_size,
        max_length=int(manifest["max_length"]),
    )

    store = FaissStore(
        dimension=int(manifest["dimension"])
    )
    store.load(
        index_path=index_path,
        metadata_path=metadata_path,
    )

    query_embedding = embedding_model.encode_query(args.query)

    filters = (
        {"law_id": args.law_id}
        if args.law_id
        else None
    )

    results = store.search(
        query_embedding=query_embedding,
        top_k=args.top_k,
        filters=filters,
    )

    print("=" * 80)
    print(f"QUERY: {args.query}")
    print("=" * 80)

    if not results:
        print("Không tìm thấy kết quả.")
        return

    for rank, result in enumerate(results, start=1):
        metadata = result["metadata"]

        print()
        print("-" * 80)
        print(
            f"TOP {rank} | score={result['score']:.6f}"
        )
        print(f"law_id   : {metadata.get('law_id')}")
        print(f"chunk_id : {metadata.get('chunk_id')}")
        print(f"citation : {metadata.get('citation')}")
        print("-" * 80)
        print(result["text"])


if __name__ == "__main__":
    main()
