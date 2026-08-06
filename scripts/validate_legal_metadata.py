import json
import sys
from datetime import date
from pathlib import Path
from typing import Any


# =========================================================
# 1. ĐƯỜNG DẪN TRONG PROJECT
# =========================================================

# File hiện tại:
# rag_model/scripts/validate_legal_metadata.py
#
# parents[0] = scripts
# parents[1] = rag_model
RAG_MODEL_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = RAG_MODEL_DIR / "data"
METADATA_DIR = DATA_DIR / "metadata"

LAWS_PATH = METADATA_DIR / "laws.json"
RELATIONS_PATH = METADATA_DIR / "relations.json"
RELATION_TYPES_PATH = METADATA_DIR / "relation_types.json"


# =========================================================
# 2. CÁC GIÁ TRỊ HỢP LỆ
# =========================================================

ALLOWED_STATUSES = {
    "active",
    "partially_expired",
    "expired",
    "not_yet_effective",
    "unknown",
}

ALLOWED_STATUS_SCOPES = {
    "whole_document",
    "partial",
    "article",
    "clause",
}

ALLOWED_DOCUMENT_ROLES = {
    "PRIMARY",
    "SECONDARY",
    "HISTORICAL",
    "AMENDMENT",
    "CONSOLIDATED",
    "GUIDANCE",
}

ALLOWED_RELATION_SCOPES = {
    "whole_document",
    "partial",
    "legal_domain",
    "article",
    "clause",
    "unknown",
}


# =========================================================
# 3. HÀM ĐỌC JSON
# =========================================================

def load_json(path: Path) -> dict[str, Any]:
    """
    Đọc file JSON và trả về dictionary.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file: {path}"
        )

    with path.open(
        mode="r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


# =========================================================
# 4. KIỂM TRA NGÀY
# =========================================================

def is_valid_date(value: str | None) -> bool:
    """
    Kiểm tra ngày có đúng định dạng YYYY-MM-DD hay không.

    None được chấp nhận cho các ngày chưa xác định.
    """
    if value is None:
        return True

    if not isinstance(value, str):
        return False

    try:
        date.fromisoformat(value)
        return True
    except ValueError:
        return False


# =========================================================
# 5. KIỂM TRA ĐƯỜNG DẪN FILE
# =========================================================

def validate_document_path(
    relative_path: str,
    owner_id: str,
    field_name: str,
) -> list[str]:
    """
    Kiểm tra một đường dẫn tài liệu được khai báo
    trong metadata.
    """
    errors: list[str] = []

    if not isinstance(relative_path, str):
        return [
            f"{owner_id}: {field_name} phải là chuỗi."
        ]

    if not relative_path.strip():
        return [
            f"{owner_id}: {field_name} không được rỗng."
        ]

    document_path = DATA_DIR / relative_path

    if not document_path.exists():
        errors.append(
            f"{owner_id}: chưa tìm thấy file trong "
            f"{field_name}:\n"
            f"  {document_path}"
        )
        return errors

    if not document_path.is_file():
        errors.append(
            f"{owner_id}: đường dẫn trong {field_name} "
            f"không phải file:\n"
            f"  {document_path}"
        )

    if document_path.suffix.lower() != ".docx":
        errors.append(
            f"{owner_id}: file trong {field_name} "
            f"phải có đuôi .docx, nhận được:\n"
            f"  {document_path.name}"
        )

    return errors


# =========================================================
# 6. KIỂM TRA MỘT DANH SÁCH FILE
# =========================================================

def validate_document_list(
    paths: Any,
    owner_id: str,
    field_name: str,
    require_non_empty: bool = True,
) -> list[str]:
    """
    Kiểm tra source_files hoặc retrieval_files.
    """
    errors: list[str] = []

    if not isinstance(paths, list):
        return [
            f"{owner_id}: {field_name} phải là danh sách."
        ]

    if require_non_empty and not paths:
        errors.append(
            f"{owner_id}: {field_name} không được rỗng."
        )
        return errors

    seen_paths: set[str] = set()

    for index, relative_path in enumerate(
        paths,
        start=1,
    ):
        if relative_path in seen_paths:
            errors.append(
                f"{owner_id}: đường dẫn bị trùng trong "
                f"{field_name}: {relative_path}"
            )
            continue

        if isinstance(relative_path, str):
            seen_paths.add(relative_path)

        errors.extend(
            validate_document_path(
                relative_path=relative_path,
                owner_id=owner_id,
                field_name=f"{field_name}[{index}]",
            )
        )

    return errors


# =========================================================
# 7. KIỂM TRA VĂN BẢN HỢP NHẤT
# =========================================================

def validate_consolidated_document(
    law_id: str,
    consolidated_document: Any,
    retrieval_files: list[str],
) -> list[str]:
    """
    Kiểm tra object consolidated_document.
    """
    errors: list[str] = []

    if consolidated_document is None:
        return errors

    if not isinstance(consolidated_document, dict):
        return [
            f"{law_id}: consolidated_document phải là "
            f"object hoặc null."
        ]

    required_fields = {
        "document_number",
        "verified_date",
        "source_file",
        "source_url",
    }

    missing_fields = (
        required_fields
        - set(consolidated_document.keys())
    )

    if missing_fields:
        errors.append(
            f"{law_id}: consolidated_document thiếu trường: "
            f"{sorted(missing_fields)}"
        )
        return errors

    document_number = consolidated_document.get(
        "document_number"
    )

    if (
        not isinstance(document_number, str)
        or not document_number.strip()
    ):
        errors.append(
            f"{law_id}: document_number của văn bản "
            f"hợp nhất không hợp lệ."
        )

    verified_date = consolidated_document.get(
        "verified_date"
    )

    if not is_valid_date(verified_date):
        errors.append(
            f"{law_id}: verified_date của văn bản hợp nhất "
            f"phải có định dạng YYYY-MM-DD."
        )

    source_file = consolidated_document.get(
        "source_file"
    )

    errors.extend(
        validate_document_path(
            relative_path=source_file,
            owner_id=law_id,
            field_name="consolidated_document.source_file",
        )
    )

    if (
        isinstance(source_file, str)
        and source_file not in retrieval_files
    ):
        errors.append(
            f"{law_id}: file văn bản hợp nhất phải xuất hiện "
            f"trong retrieval_files:\n"
            f"  {source_file}"
        )

    return errors


# =========================================================
# 8. KIỂM TRA NGÀY HIỆU LỰC ĐẶC BIỆT
# =========================================================

def validate_special_effective_dates(
    law_id: str,
    special_effective_dates: Any,
) -> list[str]:
    """
    Kiểm tra các điều khoản có ngày hiệu lực riêng.
    """
    errors: list[str] = []

    if not isinstance(special_effective_dates, list):
        return [
            f"{law_id}: special_effective_dates phải là "
            f"danh sách."
        ]

    for index, item in enumerate(
        special_effective_dates,
        start=1,
    ):
        label = (
            f"{law_id}.special_effective_dates[{index}]"
        )

        if not isinstance(item, dict):
            errors.append(
                f"{label}: phải là object."
            )
            continue

        required_fields = {
            "scope",
            "effective_from",
        }

        missing_fields = (
            required_fields
            - set(item.keys())
        )

        if missing_fields:
            errors.append(
                f"{label}: thiếu trường "
                f"{sorted(missing_fields)}"
            )
            continue

        scope = item.get("scope")

        if (
            not isinstance(scope, str)
            or not scope.strip()
        ):
            errors.append(
                f"{label}: scope không hợp lệ."
            )

        effective_from = item.get(
            "effective_from"
        )

        if not is_valid_date(effective_from):
            errors.append(
                f"{label}: effective_from phải có "
                f"định dạng YYYY-MM-DD."
            )

    return errors


# =========================================================
# 9. KIỂM TRA DANH SÁCH VĂN BẢN PHÁP LUẬT
# =========================================================

def validate_laws(
    laws: list[dict[str, Any]],
) -> tuple[
    dict[str, dict[str, Any]],
    list[str],
    list[str],
]:
    """
    Kiểm tra toàn bộ laws.json.

    Returns:
        laws_by_id:
            Tra cứu luật theo law_id.

        errors:
            Các lỗi bắt buộc phải sửa.

        warnings:
            Các cảnh báo cần xem xét.
    """
    errors: list[str] = []
    warnings: list[str] = []

    laws_by_id: dict[str, dict[str, Any]] = {}
    law_numbers: dict[str, str] = {}

    required_fields = {
        "law_id",
        "title",
        "short_title",
        "law_number",
        "document_type",
        "document_role",
        "issuer",
        "issued_date",
        "effective_from",
        "effective_to",
        "status",
        "status_scope",
        "legal_domains",
        "topics",
        "is_primary",
        "default_retrieval",
        "source_files",
        "retrieval_files",
        "consolidated_document",
        "special_effective_dates",
        "source_url",
        "status_checked_at",
    }

    for index, law in enumerate(
        laws,
        start=1,
    ):
        label = f"Văn bản thứ {index}"

        if not isinstance(law, dict):
            errors.append(
                f"{label}: phải là object."
            )
            continue

        missing_fields = (
            required_fields
            - set(law.keys())
        )

        if missing_fields:
            errors.append(
                f"{label} thiếu trường: "
                f"{sorted(missing_fields)}"
            )
            continue

        law_id = law.get("law_id")

        if (
            not isinstance(law_id, str)
            or not law_id.strip()
        ):
            errors.append(
                f"{label}: law_id không hợp lệ."
            )
            continue

        if law_id in laws_by_id:
            errors.append(
                f"law_id bị trùng: {law_id}"
            )
            continue

        laws_by_id[law_id] = law

        # -------------------------------------------------
        # Kiểm tra số hiệu văn bản
        # -------------------------------------------------

        law_number = law.get("law_number")

        if (
            not isinstance(law_number, str)
            or not law_number.strip()
        ):
            errors.append(
                f"{law_id}: law_number không hợp lệ."
            )
        elif law_number in law_numbers:
            errors.append(
                f"{law_id}: law_number bị trùng với "
                f"{law_numbers[law_number]}: {law_number}"
            )
        else:
            law_numbers[law_number] = law_id

        # -------------------------------------------------
        # Kiểm tra vai trò văn bản
        # -------------------------------------------------

        document_role = law.get(
            "document_role"
        )

        if document_role not in ALLOWED_DOCUMENT_ROLES:
            errors.append(
                f"{law_id}: document_role không hợp lệ: "
                f"{document_role}"
            )

        # -------------------------------------------------
        # Kiểm tra trạng thái hiệu lực
        # -------------------------------------------------

        status = law.get("status")

        if status not in ALLOWED_STATUSES:
            errors.append(
                f"{law_id}: status không hợp lệ: {status}"
            )

        status_scope = law.get(
            "status_scope"
        )

        if status_scope not in ALLOWED_STATUS_SCOPES:
            errors.append(
                f"{law_id}: status_scope không hợp lệ: "
                f"{status_scope}"
            )

        # -------------------------------------------------
        # Kiểm tra ngày
        # -------------------------------------------------

        date_fields = {
            "issued_date": law.get("issued_date"),
            "effective_from": law.get(
                "effective_from"
            ),
            "effective_to": law.get(
                "effective_to"
            ),
            "status_checked_at": law.get(
                "status_checked_at"
            ),
        }

        for field_name, value in date_fields.items():
            if not is_valid_date(value):
                errors.append(
                    f"{law_id}: {field_name} phải có "
                    f"định dạng YYYY-MM-DD, nhận được: "
                    f"{value}"
                )

        if (
            status == "active"
            and law.get("effective_to") is not None
        ):
            errors.append(
                f"{law_id}: luật active nhưng "
                f"effective_to không phải null."
            )

        if (
            status == "expired"
            and not law.get("effective_to")
        ):
            errors.append(
                f"{law_id}: luật expired nhưng "
                f"thiếu effective_to."
            )

        if (
            status == "partially_expired"
            and status_scope == "whole_document"
        ):
            errors.append(
                f"{law_id}: partially_expired không thể có "
                f"status_scope='whole_document'."
            )

        # -------------------------------------------------
        # Kiểm tra legal_domains và topics
        # -------------------------------------------------

        legal_domains = law.get(
            "legal_domains"
        )

        if (
            not isinstance(legal_domains, list)
            or not legal_domains
        ):
            errors.append(
                f"{law_id}: legal_domains phải là "
                f"danh sách không rỗng."
            )

        topics = law.get("topics")

        if (
            not isinstance(topics, list)
            or not topics
        ):
            warnings.append(
                f"{law_id}: topics đang rỗng hoặc "
                f"không phải danh sách."
            )

        # -------------------------------------------------
        # Kiểm tra giá trị boolean
        # -------------------------------------------------

        if not isinstance(
            law.get("is_primary"),
            bool,
        ):
            errors.append(
                f"{law_id}: is_primary phải là true/false."
            )

        if not isinstance(
            law.get("default_retrieval"),
            bool,
        ):
            errors.append(
                f"{law_id}: default_retrieval phải là "
                f"true/false."
            )

        if (
            status == "expired"
            and law.get("default_retrieval") is True
        ):
            warnings.append(
                f"{law_id}: văn bản expired nhưng "
                f"default_retrieval=true."
            )

        # -------------------------------------------------
        # Kiểm tra file nguồn
        # -------------------------------------------------

        source_files = law.get(
            "source_files"
        )

        errors.extend(
            validate_document_list(
                paths=source_files,
                owner_id=law_id,
                field_name="source_files",
                require_non_empty=True,
            )
        )

        retrieval_files = law.get(
            "retrieval_files"
        )

        errors.extend(
            validate_document_list(
                paths=retrieval_files,
                owner_id=law_id,
                field_name="retrieval_files",
                require_non_empty=True,
            )
        )

        # -------------------------------------------------
        # Kiểm tra văn bản hợp nhất
        # -------------------------------------------------

        safe_retrieval_files = (
            retrieval_files
            if isinstance(retrieval_files, list)
            else []
        )

        errors.extend(
            validate_consolidated_document(
                law_id=law_id,
                consolidated_document=law.get(
                    "consolidated_document"
                ),
                retrieval_files=safe_retrieval_files,
            )
        )

        # -------------------------------------------------
        # Kiểm tra ngày hiệu lực riêng
        # -------------------------------------------------

        errors.extend(
            validate_special_effective_dates(
                law_id=law_id,
                special_effective_dates=law.get(
                    "special_effective_dates"
                ),
            )
        )

    return laws_by_id, errors, warnings


# =========================================================
# 10. KIỂM TRA DANH SÁCH LOẠI QUAN HỆ
# =========================================================

def validate_relation_types(
    relation_types_data: dict[str, Any],
) -> tuple[set[str], list[str]]:
    """
    Đọc các loại quan hệ từ relation_types.json.
    """
    errors: list[str] = []

    relation_types = relation_types_data.get(
        "relation_types"
    )

    if not isinstance(relation_types, dict):
        return (
            set(),
            [
                "relation_types.json phải có object "
                "'relation_types'."
            ],
        )

    allowed_relations = set(
        relation_types.keys()
    )

    for relation_name, config in (
        relation_types.items()
    ):
        if not isinstance(config, dict):
            errors.append(
                f"{relation_name}: cấu hình phải là object."
            )
            continue

        if "description" not in config:
            errors.append(
                f"{relation_name}: thiếu description."
            )

        directed = config.get("directed")

        if not isinstance(directed, bool):
            errors.append(
                f"{relation_name}: directed phải là "
                f"true hoặc false."
            )

    return allowed_relations, errors


# =========================================================
# 11. KIỂM TRA QUAN HỆ GIỮA CÁC LUẬT
# =========================================================

def validate_relations(
    relations: list[dict[str, Any]],
    laws_by_id: dict[str, dict[str, Any]],
    allowed_relations: set[str],
) -> tuple[list[str], list[str]]:
    """
    Kiểm tra relations.json.
    """
    errors: list[str] = []
    warnings: list[str] = []

    relation_ids: set[str] = set()
    relation_triples: set[
        tuple[str, str, str]
    ] = set()

    required_fields = {
        "relation_id",
        "source",
        "relation",
        "target",
        "effective_date",
        "scope",
        "affected_articles",
        "topics",
        "description",
    }

    for index, relation in enumerate(
        relations,
        start=1,
    ):
        label = f"Quan hệ thứ {index}"

        if not isinstance(relation, dict):
            errors.append(
                f"{label}: phải là object."
            )
            continue

        missing_fields = (
            required_fields
            - set(relation.keys())
        )

        if missing_fields:
            errors.append(
                f"{label} thiếu trường: "
                f"{sorted(missing_fields)}"
            )
            continue

        relation_id = relation.get(
            "relation_id"
        )
        source = relation.get("source")
        target = relation.get("target")
        relation_type = relation.get(
            "relation"
        )

        if (
            not isinstance(relation_id, str)
            or not relation_id.strip()
        ):
            errors.append(
                f"{label}: relation_id không hợp lệ."
            )
            continue

        if relation_id in relation_ids:
            errors.append(
                f"relation_id bị trùng: {relation_id}"
            )

        relation_ids.add(relation_id)

        if source not in laws_by_id:
            errors.append(
                f"{relation_id}: source không tồn tại "
                f"trong laws.json: {source}"
            )

        if target not in laws_by_id:
            errors.append(
                f"{relation_id}: target không tồn tại "
                f"trong laws.json: {target}"
            )

        if source == target:
            errors.append(
                f"{relation_id}: source và target "
                f"không được giống nhau."
            )

        if relation_type not in allowed_relations:
            errors.append(
                f"{relation_id}: loại quan hệ "
                f"không hợp lệ: {relation_type}"
            )

        relation_triple = (
            str(source),
            str(relation_type),
            str(target),
        )

        if relation_triple in relation_triples:
            errors.append(
                f"{relation_id}: quan hệ bị trùng: "
                f"{relation_triple}"
            )

        relation_triples.add(
            relation_triple
        )

        effective_date = relation.get(
            "effective_date"
        )

        if not is_valid_date(effective_date):
            errors.append(
                f"{relation_id}: effective_date phải có "
                f"định dạng YYYY-MM-DD hoặc null."
            )

        scope = relation.get("scope")

        if scope not in ALLOWED_RELATION_SCOPES:
            errors.append(
                f"{relation_id}: scope không hợp lệ: "
                f"{scope}"
            )

        if not isinstance(
            relation.get("affected_articles"),
            list,
        ):
            errors.append(
                f"{relation_id}: affected_articles "
                f"phải là danh sách."
            )

        if not isinstance(
            relation.get("topics"),
            list,
        ):
            errors.append(
                f"{relation_id}: topics phải là danh sách."
            )

        # -------------------------------------------------
        # Kiểm tra riêng REPLACES
        # -------------------------------------------------

        if relation_type == "REPLACES":
            source_law = laws_by_id.get(source)
            target_law = laws_by_id.get(target)

            if (
                target_law is not None
                and target_law.get("status")
                != "expired"
            ):
                errors.append(
                    f"{relation_id}: văn bản bị thay thế "
                    f"phải có status='expired'."
                )

            if (
                source_law is not None
                and source_law.get("status")
                == "expired"
            ):
                warnings.append(
                    f"{relation_id}: văn bản thay thế đang "
                    f"có status='expired'."
                )

            if source_law is not None:
                source_effective_from = (
                    source_law.get("effective_from")
                )

                if (
                    effective_date
                    and source_effective_from
                    and effective_date
                    != source_effective_from
                ):
                    warnings.append(
                        f"{relation_id}: effective_date "
                        f"khác effective_from của văn bản "
                        f"thay thế."
                    )

        # -------------------------------------------------
        # Kiểm tra riêng RELATED_TO
        # -------------------------------------------------

        if (
            relation_type == "RELATED_TO"
            and not relation.get("topics")
        ):
            warnings.append(
                f"{relation_id}: RELATED_TO nên có topics "
                f"để phục vụ graph expansion."
            )

        # -------------------------------------------------
        # Kiểm tra riêng AMENDS
        # -------------------------------------------------

        if (
            relation_type == "AMENDS"
            and effective_date is None
        ):
            warnings.append(
                f"{relation_id}: AMENDS chưa có "
                f"effective_date."
            )

    return errors, warnings


# =========================================================
# 12. CHƯƠNG TRÌNH CHÍNH
# =========================================================

def main() -> None:
    print("=" * 70)
    print("KIỂM TRA DỮ LIỆU PHÁP LUẬT - SCHEMA 2.0")
    print("=" * 70)

    try:
        laws_data = load_json(LAWS_PATH)
        relations_data = load_json(
            RELATIONS_PATH
        )
        relation_types_data = load_json(
            RELATION_TYPES_PATH
        )

    except FileNotFoundError as error:
        print(f"\n[FILE ERROR] {error}")
        sys.exit(1)

    except json.JSONDecodeError as error:
        print("\n[JSON ERROR]")
        print(f"Dòng: {error.lineno}")
        print(f"Cột: {error.colno}")
        print(f"Chi tiết: {error.msg}")
        sys.exit(1)

    # -----------------------------------------------------
    # Kiểm tra schema version
    # -----------------------------------------------------

    root_errors: list[str] = []

    if laws_data.get("schema_version") != "2.0":
        root_errors.append(
            "laws.json phải có schema_version='2.0'."
        )

    if relations_data.get("schema_version") != "2.0":
        root_errors.append(
            "relations.json phải có schema_version='2.0'."
        )

    laws = laws_data.get("laws")
    relations = relations_data.get(
        "relations"
    )

    if not isinstance(laws, list):
        print(
            "\n[ERROR] laws.json phải có danh sách 'laws'."
        )
        sys.exit(1)

    if not isinstance(relations, list):
        print(
            "\n[ERROR] relations.json phải có "
            "danh sách 'relations'."
        )
        sys.exit(1)

    # -----------------------------------------------------
    # Chạy các validator
    # -----------------------------------------------------

    (
        allowed_relations,
        relation_type_errors,
    ) = validate_relation_types(
        relation_types_data
    )

    (
        laws_by_id,
        law_errors,
        law_warnings,
    ) = validate_laws(laws)

    (
        relation_errors,
        relation_warnings,
    ) = validate_relations(
        relations=relations,
        laws_by_id=laws_by_id,
        allowed_relations=allowed_relations,
    )

    errors = (
        root_errors
        + relation_type_errors
        + law_errors
        + relation_errors
    )

    warnings = (
        law_warnings
        + relation_warnings
    )

    # -----------------------------------------------------
    # Thống kê
    # -----------------------------------------------------

    source_file_count = sum(
        len(law.get("source_files", []))
        for law in laws
        if isinstance(law, dict)
    )

    retrieval_file_count = sum(
        len(law.get("retrieval_files", []))
        for law in laws
        if isinstance(law, dict)
    )

    consolidated_count = sum(
        1
        for law in laws
        if isinstance(law, dict)
        and law.get("consolidated_document")
        is not None
    )

    print(f"\nSố văn bản pháp luật: {len(laws)}")
    print(f"Số quan hệ graph: {len(relations)}")
    print(f"Số file nguồn: {source_file_count}")
    print(
        f"Số file dùng retrieval: "
        f"{retrieval_file_count}"
    )
    print(
        f"Số văn bản hợp nhất: "
        f"{consolidated_count}"
    )
    print(
        "Loại quan hệ hợp lệ: "
        f"{sorted(allowed_relations)}"
    )

    # -----------------------------------------------------
    # In cảnh báo
    # -----------------------------------------------------

    if warnings:
        print(
            f"\nPhát hiện {len(warnings)} cảnh báo:\n"
        )

        for number, warning in enumerate(
            warnings,
            start=1,
        ):
            print(f"[WARNING {number}] {warning}")

    # -----------------------------------------------------
    # In lỗi
    # -----------------------------------------------------

    if errors:
        print(
            f"\nPhát hiện {len(errors)} lỗi:\n"
        )

        for number, error in enumerate(
            errors,
            start=1,
        ):
            print(f"[ERROR {number}] {error}")

        print(
            "\nMetadata chưa hợp lệ. "
            "Hãy sửa các lỗi trên."
        )

        sys.exit(1)

    print("\nKhông phát hiện lỗi.")
    print("Metadata schema 2.0 hợp lệ.")
    print("Tất cả file nguồn và retrieval đều tồn tại.")
    print(
        "Các quan hệ graph đều tham chiếu "
        "đến law_id hợp lệ."
    )


if __name__ == "__main__":
    main()