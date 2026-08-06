import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# =========================================================
# 1. THƯ MỤC PROJECT
# =========================================================

RAG_MODEL_DIR = Path(__file__).resolve().parents[1]

if str(RAG_MODEL_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(RAG_MODEL_DIR),
    )


from preprocess.legal_chunker import LegalChunker
from preprocess.legal_parser import LegalDocxParser


DATA_DIR = RAG_MODEL_DIR / "data"

METADATA_DIR = DATA_DIR / "metadata"
PARSED_DIR = DATA_DIR / "parsed"
CHUNKS_DIR = DATA_DIR / "chunks"

LAWS_PATH = METADATA_DIR / "laws.json"

MASTER_CORPUS_PATH = (
    CHUNKS_DIR
    / "legal_corpus.jsonl"
)

DEFAULT_CORPUS_PATH = (
    CHUNKS_DIR
    / "default_retrieval_corpus.jsonl"
)

MANIFEST_PATH = (
    CHUNKS_DIR
    / "manifest.json"
)


# =========================================================
# 2. ĐỌC VÀ GHI FILE
# =========================================================

def load_json(
    file_path: Path,
) -> dict[str, Any]:
    if not file_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file: {file_path}"
        )

    with file_path.open(
        mode="r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def save_json(
    data: dict[str, Any],
    file_path: Path,
) -> None:
    file_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with file_path.open(
        mode="w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
        )


def save_jsonl(
    records: list[dict[str, Any]],
    file_path: Path,
) -> None:
    file_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with file_path.open(
        mode="w",
        encoding="utf-8",
    ) as file:
        for record in records:
            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )


# =========================================================
# 3. XỬ LÝ ĐƯỜNG DẪN
# =========================================================

def normalize_relative_path(
    value: str,
) -> str:
    return value.replace("\\", "/")


def resolve_data_path(
    relative_path: str,
) -> Path:
    normalized = normalize_relative_path(
        relative_path
    )

    return DATA_DIR / normalized


def build_parsed_output_path(
    law_id: str,
    source_path: Path,
    source_index: int,
    source_count: int,
) -> Path:
    """
    Nếu một luật chỉ có một retrieval file:

        civil_code_2015.json

    Nếu có nhiều retrieval file:

        housing_law_2023__01__part_1.json
        housing_law_2023__02__part_2.json
    """
    if source_count == 1:
        return (
            PARSED_DIR
            / f"{law_id}.json"
        )

    source_name = source_path.stem

    return (
        PARSED_DIR
        / (
            f"{law_id}"
            f"__{source_index:02d}"
            f"__{source_name}.json"
        )
    )


# =========================================================
# 4. THỐNG KÊ CÂY
# =========================================================

def count_tree_nodes(
    parser: LegalDocxParser,
    tree: dict[str, Any],
) -> dict[str, int]:
    counts: Counter[str] = Counter()

    for node in parser.iter_nodes(tree):
        node_type = node.get(
            "node_type",
            "UNKNOWN",
        )

        counts[node_type] += 1

    return dict(counts)


def merge_counts(
    target: Counter[str],
    source: dict[str, int],
) -> None:
    for key, value in source.items():
        target[key] += value


# =========================================================
# 5. KIỂM TRA CHUNK ID
# =========================================================

def validate_combined_chunk_ids(
    law_id: str,
    chunks: list[dict[str, Any]],
) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()

    for chunk in chunks:
        chunk_id = chunk.get("chunk_id")

        if not isinstance(chunk_id, str):
            raise ValueError(
                f"{law_id}: phát hiện chunk_id "
                f"không hợp lệ."
            )

        if chunk_id in seen:
            duplicates.add(chunk_id)

        seen.add(chunk_id)

    if duplicates:
        duplicate_preview = sorted(
            duplicates
        )[:10]

        raise ValueError(
            f"{law_id}: phát hiện chunk_id trùng "
            f"giữa các retrieval file:\n"
            + "\n".join(
                f"  - {chunk_id}"
                for chunk_id in duplicate_preview
            )
        )


# =========================================================
# 6. BUILD MỘT VĂN BẢN PHÁP LUẬT
# =========================================================

def build_one_law(
    law_metadata: dict[str, Any],
    parser: LegalDocxParser,
    chunker: LegalChunker,
) -> tuple[
    list[dict[str, Any]],
    dict[str, Any],
]:
    law_id = law_metadata["law_id"]
    law_title = law_metadata["title"]

    retrieval_files = law_metadata.get(
        "retrieval_files",
        [],
    )

    if not isinstance(retrieval_files, list):
        raise ValueError(
            f"{law_id}: retrieval_files "
            f"phải là danh sách."
        )

    if not retrieval_files:
        raise ValueError(
            f"{law_id}: retrieval_files "
            f"đang rỗng."
        )

    combined_chunks: list[
        dict[str, Any]
    ] = []

    parsed_file_paths: list[str] = []
    tree_counts: Counter[str] = Counter()

    source_reports: list[
        dict[str, Any]
    ] = []

    source_count = len(retrieval_files)

    for source_index, relative_path in enumerate(
        retrieval_files,
        start=1,
    ):
        relative_path = normalize_relative_path(
            relative_path
        )

        source_path = resolve_data_path(
            relative_path
        )

        if not source_path.exists():
            raise FileNotFoundError(
                f"{law_id}: không tìm thấy "
                f"retrieval file:\n"
                f"  {source_path}"
            )

        # -----------------------------------------------
        # Parse DOCX
        # -----------------------------------------------

        parsed_document = parser.parse(
            file_path=source_path,
            law_id=law_id,
            document_title=law_title,
        )

        # Không lưu đường dẫn tuyệt đối để project
        # có thể di chuyển sang máy khác.
        parsed_document["source_file"] = (
            relative_path
        )

        parsed_output_path = (
            build_parsed_output_path(
                law_id=law_id,
                source_path=source_path,
                source_index=source_index,
                source_count=source_count,
            )
        )

        save_json(
            data=parsed_document,
            file_path=parsed_output_path,
        )

        parsed_relative_path = (
            parsed_output_path
            .relative_to(DATA_DIR)
            .as_posix()
        )

        parsed_file_paths.append(
            parsed_relative_path
        )

        # -----------------------------------------------
        # Thống kê cây
        # -----------------------------------------------

        current_tree_counts = count_tree_nodes(
            parser=parser,
            tree=parsed_document["tree"],
        )

        merge_counts(
            target=tree_counts,
            source=current_tree_counts,
        )

        # -----------------------------------------------
        # Tạo legal chunk
        # -----------------------------------------------

        current_chunks = (
            chunker.build_chunks(
                parsed_document=parsed_document,
                law_metadata=law_metadata,
            )
        )

        for chunk in current_chunks:
            chunk["source_file"] = (
                relative_path
            )

            chunk["source_document_index"] = (
                source_index
            )

            chunk["source_document_count"] = (
                source_count
            )

            chunk["is_default_retrieval"] = (
                law_metadata.get(
                    "default_retrieval",
                    False,
                )
            )

        combined_chunks.extend(
            current_chunks
        )

        current_summary = chunker.summary(
            current_chunks
        )

        source_reports.append({
            "source_file": relative_path,
            "parsed_file": parsed_relative_path,
            "paragraph_count": (
                parsed_document
                .get("statistics", {})
                .get("paragraphs_total", 0)
            ),
            "table_count": (
                parsed_document
                .get("statistics", {})
                .get("tables_total", 0)
            ),
            "tree_counts": current_tree_counts,
            "chunk_summary": current_summary,
        })

    # Kiểm tra trùng chunk giữa nhiều file
    validate_combined_chunk_ids(
        law_id=law_id,
        chunks=combined_chunks,
    )

    # Lưu chunk riêng của luật
    law_chunk_path = (
        CHUNKS_DIR
        / f"{law_id}.jsonl"
    )

    save_jsonl(
        records=combined_chunks,
        file_path=law_chunk_path,
    )

    law_summary = chunker.summary(
        combined_chunks
    )

    report = {
        "law_id": law_id,
        "law_title": law_title,
        "law_number": law_metadata.get(
            "law_number"
        ),
        "status": law_metadata.get(
            "status"
        ),
        "default_retrieval": (
            law_metadata.get(
                "default_retrieval",
                False,
            )
        ),
        "retrieval_files": [
            normalize_relative_path(path)
            for path in retrieval_files
        ],
        "parsed_files": parsed_file_paths,
        "chunk_file": (
            law_chunk_path
            .relative_to(DATA_DIR)
            .as_posix()
        ),
        "tree_counts": dict(tree_counts),
        "chunk_summary": law_summary,
        "sources": source_reports,
    }

    return combined_chunks, report


# =========================================================
# 7. IN KẾT QUẢ MỘT LUẬT
# =========================================================

def print_law_report(
    report: dict[str, Any],
) -> None:
    law_id = report["law_id"]
    title = report["law_title"]

    tree_counts = report.get(
        "tree_counts",
        {},
    )

    chunk_summary = report.get(
        "chunk_summary",
        {},
    )

    print("\n" + "-" * 80)
    print(
        f"[OK] {title}"
    )
    print(
        f"     law_id: {law_id}"
    )
    print(
        f"     Số retrieval file: "
        f"{len(report['retrieval_files'])}"
    )
    print(
        f"     Số Điều: "
        f"{tree_counts.get('ARTICLE', 0)}"
    )
    print(
        f"     Số Khoản: "
        f"{tree_counts.get('CLAUSE', 0)}"
    )
    print(
        f"     Số Điểm: "
        f"{tree_counts.get('POINT', 0)}"
    )
    print(
        f"     Số chunk: "
        f"{chunk_summary.get('chunk_count', 0)}"
    )
    print(
        f"     Loại chunk: "
        f"{chunk_summary.get('type_counts', {})}"
    )
    print(
        f"     Độ dài: "
        f"min={chunk_summary.get('min_chars', 0)}, "
        f"max={chunk_summary.get('max_chars', 0)}, "
        f"avg={chunk_summary.get('average_chars', 0)}"
    )
    print(
        f"     Default retrieval: "
        f"{report['default_retrieval']}"
    )


# =========================================================
# 8. XỬ LÝ THAM SỐ
# =========================================================

def parse_arguments() -> argparse.Namespace:
    argument_parser = argparse.ArgumentParser(
        description=(
            "Parse và chunk toàn bộ văn bản pháp luật "
            "được khai báo trong laws.json."
        )
    )

    argument_parser.add_argument(
        "--law-id",
        action="append",
        default=None,
        help=(
            "Chỉ build một law_id. "
            "Có thể truyền nhiều lần."
        ),
    )

    argument_parser.add_argument(
        "--only-default",
        action="store_true",
        help=(
            "Chỉ build các văn bản có "
            "default_retrieval=true."
        ),
    )

    argument_parser.add_argument(
        "--max-chars",
        type=int,
        default=1800,
        help="Độ dài tối đa của chunk.",
    )

    argument_parser.add_argument(
        "--min-chars",
        type=int,
        default=40,
        help="Độ dài tối thiểu của chunk.",
    )

    return argument_parser.parse_args()


# =========================================================
# 9. CHƯƠNG TRÌNH CHÍNH
# =========================================================

def main() -> None:
    arguments = parse_arguments()

    print("=" * 80)
    print("BUILD LEGAL CORPUS")
    print("=" * 80)

    laws_data = load_json(
        LAWS_PATH
    )

    laws = laws_data.get(
        "laws",
        [],
    )

    if not isinstance(laws, list):
        print(
            "[ERROR] laws.json không có "
            "danh sách laws hợp lệ."
        )
        sys.exit(1)

    selected_law_ids = (
        set(arguments.law_id)
        if arguments.law_id
        else None
    )

    selected_laws: list[
        dict[str, Any]
    ] = []

    for law in laws:
        law_id = law.get("law_id")

        if (
            selected_law_ids is not None
            and law_id not in selected_law_ids
        ):
            continue

        if (
            arguments.only_default
            and not law.get(
                "default_retrieval",
                False,
            )
        ):
            continue

        selected_laws.append(law)

    if not selected_laws:
        print(
            "[ERROR] Không có văn bản nào "
            "được chọn để build."
        )
        sys.exit(1)

    if selected_law_ids is not None:
        existing_ids = {
            law.get("law_id")
            for law in selected_laws
        }

        missing_ids = (
            selected_law_ids
            - existing_ids
        )

        if missing_ids:
            print(
                "[ERROR] Không tìm thấy law_id:"
            )

            for law_id in sorted(missing_ids):
                print(f"  - {law_id}")

            sys.exit(1)

    PARSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    CHUNKS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    parser = LegalDocxParser()

    chunker = LegalChunker(
        max_chars=arguments.max_chars,
        min_chars=arguments.min_chars,
    )

    all_chunks: list[
        dict[str, Any]
    ] = []

    default_chunks: list[
        dict[str, Any]
    ] = []

    law_reports: list[
        dict[str, Any]
    ] = []

    errors: list[str] = []

    for law in selected_laws:
        law_id = law.get(
            "law_id",
            "UNKNOWN",
        )

        try:
            chunks, report = build_one_law(
                law_metadata=law,
                parser=parser,
                chunker=chunker,
            )

            all_chunks.extend(chunks)

            if law.get(
                "default_retrieval",
                False,
            ):
                default_chunks.extend(
                    chunks
                )

            law_reports.append(report)

            print_law_report(report)

        except Exception as error:
            error_message = (
                f"{law_id}: {error}"
            )

            errors.append(
                error_message
            )

            print("\n" + "-" * 80)
            print(
                f"[ERROR] {error_message}"
            )

    # Không ghi master corpus nếu có luật bị lỗi.
    if errors:
        print("\n" + "=" * 80)
        print(
            f"BUILD THẤT BẠI: "
            f"{len(errors)} lỗi"
        )
        print("=" * 80)

        for index, error in enumerate(
            errors,
            start=1,
        ):
            print(
                f"{index}. {error}"
            )

        sys.exit(1)

    # Kiểm tra chunk_id toàn corpus
    validate_combined_chunk_ids(
        law_id="MASTER_CORPUS",
        chunks=all_chunks,
    )

    validate_combined_chunk_ids(
        law_id="DEFAULT_CORPUS",
        chunks=default_chunks,
    )

    # Ghi hai corpus
    save_jsonl(
        records=all_chunks,
        file_path=MASTER_CORPUS_PATH,
    )

    save_jsonl(
        records=default_chunks,
        file_path=DEFAULT_CORPUS_PATH,
    )

    all_summary = chunker.summary(
        all_chunks
    )

    default_summary = chunker.summary(
        default_chunks
    )

    default_law_ids = sorted({
        chunk["law_id"]
        for chunk in default_chunks
    })

    manifest = {
        "schema_version": "1.0",
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "settings": {
            "max_chars": arguments.max_chars,
            "min_chars": arguments.min_chars,
            "only_default": (
                arguments.only_default
            ),
            "selected_law_ids": (
                sorted(selected_law_ids)
                if selected_law_ids
                else None
            ),
        },
        "law_count": len(
            law_reports
        ),
        "retrieval_file_count": sum(
            len(
                report.get(
                    "retrieval_files",
                    [],
                )
            )
            for report in law_reports
        ),
        "all_corpus": {
            "file": (
                MASTER_CORPUS_PATH
                .relative_to(DATA_DIR)
                .as_posix()
            ),
            **all_summary,
        },
        "default_corpus": {
            "file": (
                DEFAULT_CORPUS_PATH
                .relative_to(DATA_DIR)
                .as_posix()
            ),
            "law_ids": default_law_ids,
            **default_summary,
        },
        "laws": law_reports,
    }

    save_json(
        data=manifest,
        file_path=MANIFEST_PATH,
    )

    print("\n" + "=" * 80)
    print("BUILD THÀNH CÔNG")
    print("=" * 80)

    print(
        f"Số văn bản: "
        f"{len(law_reports)}"
    )

    print(
        f"Tổng số chunk: "
        f"{len(all_chunks)}"
    )

    print(
        f"Số chunk mặc định cho retrieval: "
        f"{len(default_chunks)}"
    )

    print(
        "Các law_id mặc định:"
    )

    for law_id in default_law_ids:
        print(f"  - {law_id}")

    print(
        f"\nCorpus đầy đủ:\n"
        f"  {MASTER_CORPUS_PATH}"
    )

    print(
        f"\nCorpus dùng cho FAISS:\n"
        f"  {DEFAULT_CORPUS_PATH}"
    )

    print(
        f"\nManifest:\n"
        f"  {MANIFEST_PATH}"
    )


if __name__ == "__main__":
    main()