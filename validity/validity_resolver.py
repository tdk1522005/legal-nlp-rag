from __future__ import annotations

from datetime import date, datetime
from typing import Any

from graph.law_graph import LawGraph


class ValidityResolver:
    """
    Kiểm tra hiệu lực pháp lý ở cấp văn bản.

    Resolver hiện xử lý:
    - Ngày bắt đầu hiệu lực.
    - Ngày hết hiệu lực.
    - Văn bản bị thay thế.
    - Văn bản sửa đổi, bổ sung.
    - Trạng thái hết hiệu lực một phần.
    - Văn bản hợp nhất được ưu tiên.

    Chưa xử lý chính xác hiệu lực ở cấp:
    - Điều.
    - Khoản.
    - Điểm.

    Muốn xử lý cấp điều khoản, metadata quan hệ phải
    khai báo đầy đủ affected_articles hoặc provision_id.
    """

    PARTIAL_STATUSES = {
        "partially_expired",
        "partially_effective",
    }

    INACTIVE_STATUSES = {
        "expired",
        "repealed",
        "replaced",
        "cancelled",
    }

    def __init__(
        self,
        law_graph: LawGraph | None = None,
    ) -> None:
        self.law_graph = (
            law_graph
            if law_graph is not None
            else LawGraph()
        )

        if (
            self.law_graph.graph.number_of_nodes()
            == 0
        ):
            self.law_graph.load()

    @staticmethod
    def parse_date(
        value: str | date | datetime | None,
        *,
        default_today: bool = False,
        field_name: str = "date",
    ) -> date | None:
        if value is None:
            return (
                date.today()
                if default_today
                else None
            )

        if isinstance(value, datetime):
            return value.date()

        if isinstance(value, date):
            return value

        clean_value = str(value).strip()

        if not clean_value:
            return (
                date.today()
                if default_today
                else None
            )

        try:
            return date.fromisoformat(
                clean_value
            )

        except ValueError as error:
            raise ValueError(
                f"{field_name} phải có định dạng "
                "YYYY-MM-DD."
            ) from error

    @staticmethod
    def _relation_is_effective(
        edge: dict[str, Any],
        as_of: date,
    ) -> bool:
        effective_date = (
            ValidityResolver.parse_date(
                edge.get("effective_date"),
                field_name=(
                    "relation.effective_date"
                ),
            )
        )

        if effective_date is None:
            return True

        return as_of >= effective_date

    def _evaluate_metadata(
        self,
        law: dict[str, Any],
        as_of: date,
    ) -> dict[str, Any]:
        effective_from = self.parse_date(
            law.get("effective_from"),
            field_name="effective_from",
        )

        effective_to = self.parse_date(
            law.get("effective_to"),
            field_name="effective_to",
        )

        status = str(
            law.get("status", "unknown")
        ).strip().lower()

        status_scope = str(
            law.get(
                "status_scope",
                "whole_document",
            )
        ).strip().lower()

        if (
            effective_from is not None
            and as_of < effective_from
        ):
            return {
                "is_effective": False,
                "validity_state": (
                    "not_yet_effective"
                ),
                "reason": (
                    "Văn bản chưa có hiệu lực "
                    f"tại ngày {as_of.isoformat()}."
                ),
                "effective_from": (
                    effective_from.isoformat()
                ),
                "effective_to": (
                    effective_to.isoformat()
                    if effective_to
                    else None
                ),
                "status": status,
                "status_scope": status_scope,
            }

        # effective_to được hiểu là mốc văn bản
        # không còn được áp dụng kể từ ngày đó.
        if (
            effective_to is not None
            and as_of >= effective_to
        ):
            return {
                "is_effective": False,
                "validity_state": "expired",
                "reason": (
                    "Văn bản đã hết hiệu lực "
                    f"tại ngày {as_of.isoformat()}."
                ),
                "effective_from": (
                    effective_from.isoformat()
                    if effective_from
                    else None
                ),
                "effective_to": (
                    effective_to.isoformat()
                ),
                "status": status,
                "status_scope": status_scope,
            }

        if (
            status in self.INACTIVE_STATUSES
            and effective_to is None
        ):
            return {
                "is_effective": False,
                "validity_state": status,
                "reason": (
                    "Metadata đánh dấu văn bản "
                    f"ở trạng thái {status}."
                ),
                "effective_from": (
                    effective_from.isoformat()
                    if effective_from
                    else None
                ),
                "effective_to": None,
                "status": status,
                "status_scope": status_scope,
            }

        if (
            status in self.PARTIAL_STATUSES
            or status_scope == "partial"
        ):
            validity_state = (
                "partially_effective"
            )
            reason = (
                "Văn bản còn được sử dụng nhưng "
                "có nội dung đã hết hiệu lực hoặc "
                "đã được sửa đổi một phần."
            )

        else:
            validity_state = "effective"
            reason = (
                "Văn bản có hiệu lực tại thời "
                "điểm được kiểm tra."
            )

        return {
            "is_effective": True,
            "validity_state": validity_state,
            "reason": reason,
            "effective_from": (
                effective_from.isoformat()
                if effective_from
                else None
            ),
            "effective_to": (
                effective_to.isoformat()
                if effective_to
                else None
            ),
            "status": status,
            "status_scope": status_scope,
        }

    def _get_replacements(
        self,
        law_id: str,
        as_of: date,
    ) -> list[dict[str, Any]]:
        replacements: list[
            dict[str, Any]
        ] = []

        relations = (
            self.law_graph.get_replacement_for(
                law_id
            )
        )

        for relation in relations:
            edge = relation.get(
                "edge",
                {},
            )

            if not self._relation_is_effective(
                edge,
                as_of,
            ):
                continue

            replacement_law = dict(
                relation.get(
                    "source_law",
                    {},
                )
            )

            replacement_law_id = str(
                relation.get("source", "")
            )

            evaluation = (
                self._evaluate_metadata(
                    replacement_law,
                    as_of,
                )
            )

            replacements.append({
                "law_id": replacement_law_id,
                "title": replacement_law.get(
                    "title"
                ),
                "law_number": (
                    replacement_law.get(
                        "law_number"
                    )
                ),
                "relation_id": relation.get(
                    "relation_id"
                ),
                "effective_date": edge.get(
                    "effective_date"
                ),
                "is_effective": evaluation[
                    "is_effective"
                ],
                "validity_state": evaluation[
                    "validity_state"
                ],
            })

        return replacements

    def _get_amending_laws(
        self,
        law_id: str,
        as_of: date,
    ) -> list[dict[str, Any]]:
        amending_laws: list[
            dict[str, Any]
        ] = []

        relations = (
            self.law_graph.get_amending_laws(
                law_id
            )
        )

        for relation in relations:
            edge = relation.get(
                "edge",
                {},
            )

            if not self._relation_is_effective(
                edge,
                as_of,
            ):
                continue

            amendment_law = dict(
                relation.get(
                    "source_law",
                    {},
                )
            )

            amendment_law_id = str(
                relation.get("source", "")
            )

            evaluation = (
                self._evaluate_metadata(
                    amendment_law,
                    as_of,
                )
            )

            if not evaluation[
                "is_effective"
            ]:
                continue

            amending_laws.append({
                "law_id": amendment_law_id,
                "title": amendment_law.get(
                    "title"
                ),
                "law_number": (
                    amendment_law.get(
                        "law_number"
                    )
                ),
                "relation_id": relation.get(
                    "relation_id"
                ),
                "effective_date": edge.get(
                    "effective_date"
                ),
                "scope": edge.get("scope"),
                "affected_articles": (
                    edge.get(
                        "affected_articles",
                        [],
                    )
                ),
                "description": edge.get(
                    "description"
                ),
            })

        return amending_laws

    def evaluate_law(
        self,
        identifier: str,
        *,
        as_of: str | date | datetime | None = None,
    ) -> dict[str, Any]:
        query_date = self.parse_date(
            as_of,
            default_today=True,
            field_name="as_of",
        )

        if query_date is None:
            raise RuntimeError(
                "Không xác định được ngày tra cứu."
            )

        law = self.law_graph.get_law(
            identifier
        )

        if law is None:
            raise ValueError(
                "Không tìm thấy văn bản: "
                f"{identifier}"
            )

        law_id = str(law["law_id"])

        evaluation = self._evaluate_metadata(
            law,
            query_date,
        )

        replacements = self._get_replacements(
            law_id,
            query_date,
        )

        amending_laws = (
            self._get_amending_laws(
                law_id,
                query_date,
            )
        )

        warnings: list[str] = []

        if (
            evaluation["validity_state"]
            == "partially_effective"
        ):
            warnings.append(
                "Văn bản có nội dung hết hiệu lực "
                "hoặc bị sửa đổi một phần. Không "
                "được mặc định mọi Điều, Khoản "
                "đều còn nguyên hiệu lực."
            )

        if amending_laws:
            missing_affected_articles = any(
                not item.get(
                    "affected_articles"
                )
                for item in amending_laws
            )

            if missing_affected_articles:
                warnings.append(
                    "Có văn bản sửa đổi nhưng "
                    "metadata chưa chỉ rõ các Điều, "
                    "Khoản bị tác động."
                )

        special_effective_dates = list(
            law.get(
                "special_effective_dates",
                [],
            )
            or []
        )

        if special_effective_dates:
            warnings.append(
                "Văn bản có ngày hiệu lực đặc "
                "biệt đối với một số điều khoản."
            )

        checked_at = self.parse_date(
            law.get("status_checked_at"),
            field_name="status_checked_at",
        )

        if (
            checked_at is not None
            and query_date > checked_at
        ):
            warnings.append(
                "Trạng thái metadata mới được "
                "kiểm tra đến ngày "
                f"{checked_at.isoformat()}."
            )

        effective_replacements = [
            item
            for item in replacements
            if item["is_effective"]
        ]

        if (
            not evaluation["is_effective"]
            and effective_replacements
        ):
            replacement_names = ", ".join(
                str(
                    item.get("title")
                    or item["law_id"]
                )
                for item in (
                    effective_replacements
                )
            )

            evaluation["reason"] += (
                " Văn bản thay thế hiện hành: "
                f"{replacement_names}."
            )

        return {
            "law_id": law_id,
            "title": law.get("title"),
            "law_number": law.get(
                "law_number"
            ),
            "as_of": query_date.isoformat(),
            **evaluation,
            "default_retrieval": law.get(
                "default_retrieval",
                False,
            ),
            "document_role": law.get(
                "document_role"
            ),
            "consolidated_document": (
                law.get(
                    "consolidated_document"
                )
            ),
            "special_effective_dates": (
                special_effective_dates
            ),
            "replacements": replacements,
            "amending_laws": amending_laws,
            "warnings": warnings,
        }

    def resolve_results(
        self,
        results: list[dict[str, Any]],
        *,
        as_of: str | date | datetime | None = None,
    ) -> dict[str, Any]:
        """
        Kiểm tra danh sách kết quả retrieval.

        Kết quả không còn hiệu lực ở thời điểm tra
        cứu sẽ được chuyển sang excluded_results.
        """
        query_date = self.parse_date(
            as_of,
            default_today=True,
            field_name="as_of",
        )

        if query_date is None:
            raise RuntimeError(
                "Không xác định được ngày tra cứu."
            )

        evaluations: dict[
            str,
            dict[str, Any],
        ] = {}

        valid_results: list[
            dict[str, Any]
        ] = []

        excluded_results: list[
            dict[str, Any]
        ] = []

        for result in results:
            metadata = dict(
                result.get(
                    "metadata",
                    {},
                )
            )

            law_id = str(
                metadata.get(
                    "law_id",
                    "",
                )
            ).strip()

            if not law_id:
                excluded_item = dict(result)
                excluded_item[
                    "validity_error"
                ] = "Kết quả thiếu law_id."

                excluded_results.append(
                    excluded_item
                )
                continue

            if law_id not in evaluations:
                evaluations[law_id] = (
                    self.evaluate_law(
                        law_id,
                        as_of=query_date,
                    )
                )

            evaluation = evaluations[
                law_id
            ]

            annotated_result = dict(
                result
            )

            metadata["validity"] = {
                "as_of": evaluation[
                    "as_of"
                ],
                "is_effective": evaluation[
                    "is_effective"
                ],
                "validity_state": evaluation[
                    "validity_state"
                ],
                "warnings": evaluation[
                    "warnings"
                ],
            }

            annotated_result[
                "metadata"
            ] = metadata

            if evaluation["is_effective"]:
                valid_results.append(
                    annotated_result
                )
            else:
                annotated_result[
                    "validity_reason"
                ] = evaluation["reason"]

                excluded_results.append(
                    annotated_result
                )

        return {
            "as_of": query_date.isoformat(),
            "valid_results": valid_results,
            "excluded_results": (
                excluded_results
            ),
            "law_evaluations": evaluations,
        }