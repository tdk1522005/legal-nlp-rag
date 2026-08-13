import json
import re
from pathlib import Path
from typing import Any, Iterable

import networkx as nx


class LawGraph:
    """
    Graph biểu diễn quan hệ giữa các văn bản pháp luật.

    Node:
        Một văn bản pháp luật trong laws.json.

    Edge:
        Một quan hệ pháp luật trong relations.json.

    Ví dụ:
        civil_code_2015
            --REPLACES-->
        civil_code_2005
    """

    def __init__(
        self,
        metadata_dir: str | Path | None = None,
    ) -> None:
        # File hiện tại:
        # rag_model/graph/law_graph.py
        #
        # parents[0] = graph
        # parents[1] = rag_model
        rag_model_dir = Path(__file__).resolve().parents[1]

        if metadata_dir is None:
            metadata_dir = (
                rag_model_dir
                / "data"
                / "metadata"
            )

        self.metadata_dir = Path(metadata_dir)

        self.laws_path = (
            self.metadata_dir
            / "laws.json"
        )

        self.relations_path = (
            self.metadata_dir
            / "relations.json"
        )

        self.relation_types_path = (
            self.metadata_dir
            / "relation_types.json"
        )

        # MultiDiGraph:
        # - Có hướng.
        # - Cho phép nhiều loại quan hệ giữa cùng hai luật.
        self.graph = nx.MultiDiGraph()

        self.relation_types: dict[str, dict[str, Any]] = {}

        # Dùng để tìm luật theo:
        # - law_id
        # - số hiệu
        # - tên đầy đủ
        # - tên viết tắt
        self.alias_to_law_id: dict[str, str] = {}

    # =====================================================
    # 1. ĐỌC DỮ LIỆU
    # =====================================================

    @staticmethod
    def _load_json(path: Path) -> dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy file: {path}"
            )

        with path.open(
            mode="r",
            encoding="utf-8",
        ) as file:
            return json.load(file)

    def load(self) -> None:
        """
        Đọc metadata và xây graph trong bộ nhớ.
        """
        laws_data = self._load_json(
            self.laws_path
        )

        relations_data = self._load_json(
            self.relations_path
        )

        relation_types_data = self._load_json(
            self.relation_types_path
        )

        laws = laws_data.get("laws", [])
        relations = relations_data.get(
            "relations",
            [],
        )

        self.relation_types = (
            relation_types_data.get(
                "relation_types",
                {},
            )
        )

        # Xóa graph cũ nếu load lại.
        self.graph.clear()
        self.alias_to_law_id.clear()

        self._add_law_nodes(laws)
        self._add_relation_edges(relations)
        self._build_alias_index()

    # =====================================================
    # 2. THÊM NODE VĂN BẢN
    # =====================================================

    def _add_law_nodes(
        self,
        laws: list[dict[str, Any]],
    ) -> None:
        for law in laws:
            law_id = law["law_id"]

            # Toàn bộ metadata của luật được lưu
            # dưới dạng thuộc tính node.
            self.graph.add_node(
                law_id,
                **law,
            )

    # =====================================================
    # 3. THÊM EDGE QUAN HỆ
    # =====================================================

    def _add_relation_edges(
        self,
        relations: list[dict[str, Any]],
    ) -> None:
        for relation in relations:
            source = relation["source"]
            target = relation["target"]
            relation_id = relation[
                "relation_id"
            ]

            if source not in self.graph:
                raise ValueError(
                    f"Source không tồn tại: {source}"
                )

            if target not in self.graph:
                raise ValueError(
                    f"Target không tồn tại: {target}"
                )

            self.graph.add_edge(
                source,
                target,
                key=relation_id,
                **relation,
            )

    # =====================================================
    # 4. CHUẨN HÓA CHUỖI TÌM KIẾM
    # =====================================================

    @staticmethod
    def _normalize_identifier(
        value: str,
    ) -> str:
        """
        Chuẩn hóa tên luật hoặc số hiệu để tìm kiếm.
        """
        value = value.strip().lower()

        # Xóa khoảng trắng liên tiếp.
        value = re.sub(
            r"\s+",
            " ",
            value,
        )

        return value

    def _build_alias_index(self) -> None:
        """
        Tạo bảng ánh xạ tên/số hiệu sang law_id.
        """
        for law_id, metadata in (
            self.graph.nodes(data=True)
        ):
            aliases = {
                law_id,
                metadata.get("law_number", ""),
                metadata.get("title", ""),
                metadata.get("short_title", ""),
            }

            for alias in aliases:
                if not isinstance(alias, str):
                    continue

                if not alias.strip():
                    continue

                normalized_alias = (
                    self._normalize_identifier(alias)
                )

                self.alias_to_law_id[
                    normalized_alias
                ] = law_id

    # =====================================================
    # 5. TÌM MỘT LUẬT
    # =====================================================

    def resolve_law_id(
        self,
        identifier: str,
    ) -> str | None:
        """
        Tìm law_id từ law_id, số hiệu hoặc tên luật.

        Ví dụ:
            civil_code_2015
            91/2015/QH13
            Bộ luật Dân sự 2015
            BLDS 2015
        """
        normalized_identifier = (
            self._normalize_identifier(
                identifier
            )
        )

        return self.alias_to_law_id.get(
            normalized_identifier
        )

    def get_law(
        self,
        identifier: str,
    ) -> dict[str, Any] | None:
        """
        Lấy metadata của một luật.
        """
        law_id = self.resolve_law_id(
            identifier
        )

        if law_id is None:
            return None

        metadata = dict(
            self.graph.nodes[law_id]
        )

        metadata["law_id"] = law_id

        return metadata

    # =====================================================
    # 6. LỌC LOẠI QUAN HỆ
    # =====================================================

    @staticmethod
    def _relation_is_allowed(
        relation_type: str,
        relation_types: set[str] | None,
    ) -> bool:
        if relation_types is None:
            return True

        return relation_type in relation_types

    # =====================================================
    # 7. QUAN HỆ ĐI RA
    # =====================================================

    def get_outgoing_relations(
        self,
        identifier: str,
        relation_types: Iterable[str] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Lấy các quan hệ đi từ luật đang xét.

        Ví dụ:
            Luật 43/2024/QH15
                --AMENDS-->
            Luật Đất đai 2024
        """
        law_id = self.resolve_law_id(
            identifier
        )

        if law_id is None:
            return []

        allowed_types = (
            set(relation_types)
            if relation_types is not None
            else None
        )

        results: list[dict[str, Any]] = []

        for (
            source,
            target,
            relation_id,
            edge_data,
        ) in self.graph.out_edges(
            law_id,
            keys=True,
            data=True,
        ):
            relation_type = edge_data.get(
                "relation",
                "",
            )

            if not self._relation_is_allowed(
                relation_type,
                allowed_types,
            ):
                continue

            target_metadata = dict(
                self.graph.nodes[target]
            )

            results.append({
                "relation_id": relation_id,
                "source": source,
                "relation": relation_type,
                "target": target,
                "target_law": target_metadata,
                "edge": dict(edge_data),
            })

        return results

    # =====================================================
    # 8. QUAN HỆ ĐI VÀO
    # =====================================================

    def get_incoming_relations(
        self,
        identifier: str,
        relation_types: Iterable[str] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Lấy các quan hệ đi vào luật đang xét.

        Ví dụ:
            Bộ luật Dân sự 2015
                --REPLACES-->
            Bộ luật Dân sự 2005

        Khi xét Bộ luật Dân sự 2005,
        đây là một incoming relation.
        """
        law_id = self.resolve_law_id(
            identifier
        )

        if law_id is None:
            return []

        allowed_types = (
            set(relation_types)
            if relation_types is not None
            else None
        )

        results: list[dict[str, Any]] = []

        for (
            source,
            target,
            relation_id,
            edge_data,
        ) in self.graph.in_edges(
            law_id,
            keys=True,
            data=True,
        ):
            relation_type = edge_data.get(
                "relation",
                "",
            )

            if not self._relation_is_allowed(
                relation_type,
                allowed_types,
            ):
                continue

            source_metadata = dict(
                self.graph.nodes[source]
            )

            results.append({
                "relation_id": relation_id,
                "source": source,
                "source_law": source_metadata,
                "relation": relation_type,
                "target": target,
                "edge": dict(edge_data),
            })

        return results

    # =====================================================
    # 9. LUẬT THAY THẾ
    # =====================================================

    def get_replacement_for(
        self,
        old_law_identifier: str,
    ) -> list[dict[str, Any]]:
        """
        Tìm văn bản thay thế một văn bản cũ.

        Ví dụ:
            civil_code_2005
                <- REPLACES -
            civil_code_2015
        """
        return self.get_incoming_relations(
            identifier=old_law_identifier,
            relation_types={"REPLACES"},
        )

    def get_replaced_laws(
        self,
        new_law_identifier: str,
    ) -> list[dict[str, Any]]:
        """
        Tìm các văn bản bị văn bản mới thay thế.
        """
        return self.get_outgoing_relations(
            identifier=new_law_identifier,
            relation_types={"REPLACES"},
        )

    # =====================================================
    # 10. QUAN HỆ SỬA ĐỔI
    # =====================================================

    def get_amended_laws(
        self,
        amendment_identifier: str,
    ) -> list[dict[str, Any]]:
        """
        Tìm những luật bị văn bản sửa đổi tác động.
        """
        return self.get_outgoing_relations(
            identifier=amendment_identifier,
            relation_types={"AMENDS"},
        )

    def get_amending_laws(
        self,
        target_identifier: str,
    ) -> list[dict[str, Any]]:
        """
        Tìm các luật sửa đổi một luật đang xét.
        """
        return self.get_incoming_relations(
            identifier=target_identifier,
            relation_types={"AMENDS"},
        )

    # =====================================================
    # 11. CÁC LUẬT LIÊN QUAN
    # =====================================================

    def get_related_laws(
        self,
        identifier: str,
    ) -> list[dict[str, Any]]:
        """
        RELATED_TO được coi là quan hệ hai chiều khi truy vấn.

        Trong JSON chỉ cần lưu một cạnh:
            civil_code_2015
                --RELATED_TO-->
            land_law_2024

        Nhưng khi hỏi từ land_law_2024, hệ thống vẫn
        tìm thấy civil_code_2015.
        """
        law_id = self.resolve_law_id(
            identifier
        )

        if law_id is None:
            return []

        related: dict[str, dict[str, Any]] = {}

        outgoing = self.get_outgoing_relations(
            law_id,
            relation_types={"RELATED_TO"},
        )

        for item in outgoing:
            target = item["target"]

            related[target] = {
                "law_id": target,
                "relation": "RELATED_TO",
                "law": item["target_law"],
                "edge": item["edge"],
            }

        incoming = self.get_incoming_relations(
            law_id,
            relation_types={"RELATED_TO"},
        )

        for item in incoming:
            source = item["source"]

            related[source] = {
                "law_id": source,
                "relation": "RELATED_TO",
                "law": item["source_law"],
                "edge": item["edge"],
            }

        return list(related.values())

    # =====================================================
    # 12. THỐNG KÊ GRAPH
    # =====================================================

    def summary(self) -> dict[str, Any]:
        """
        Trả về thống kê cơ bản của graph.
        """
        relation_counts: dict[str, int] = {}

        for _, _, edge_data in self.graph.edges(
            data=True
        ):
            relation_type = edge_data.get(
                "relation",
                "UNKNOWN",
            )

            relation_counts[relation_type] = (
                relation_counts.get(
                    relation_type,
                    0,
                )
                + 1
            )

        return {
            "node_count": self.graph.number_of_nodes(),
            "edge_count": self.graph.number_of_edges(),
            "relation_counts": relation_counts,
        }