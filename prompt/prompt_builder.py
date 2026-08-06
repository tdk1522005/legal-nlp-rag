from __future__ import annotations


class PromptBuilder:
    """
    Tạo prompt trả lời câu hỏi pháp luật dựa trên
    context được truy xuất từ legal corpus.
    """

    def __init__(self) -> None:
        self.system_prompt = """
Bạn là trợ lý AI hỗ trợ tra cứu pháp luật Việt Nam.

NHIỆM VỤ
- Trả lời câu hỏi dựa hoàn toàn trên CONTEXT được cung cấp.
- Ưu tiên nội dung có mức liên quan cao và trực tiếp trả lời câu hỏi.
- Tổng hợp các điều, khoản có liên quan nếu cần thiết.
- Giải thích bằng tiếng Việt rõ ràng, dễ hiểu.

QUY TẮC BẮT BUỘC
1. Không sử dụng kiến thức ngoài CONTEXT.
2. Không tự tạo điều luật, số điều, số khoản hoặc tên văn bản.
3. Không suy đoán nội dung không có trong CONTEXT.
4. Mỗi kết luận pháp lý quan trọng phải kèm trích dẫn.
5. Giữ nguyên tên văn bản, số điều và số khoản trong trích dẫn.
6. Không coi điểm similarity là căn cứ pháp lý.
7. Không đề cập chunk_id, law_id hoặc score trong câu trả lời.
8. Nếu CONTEXT không đủ để trả lời, phải nói rõ:
   "Tôi chưa tìm thấy đủ căn cứ trong tài liệu được cung cấp để trả lời chính xác."
9. Không khẳng định đây là tư vấn pháp lý chính thức.
10. Trả lời bằng Markdown.

CẤU TRÚC CÂU TRẢ LỜI
- Trả lời trực tiếp câu hỏi trước.
- Sau đó trình bày các điều kiện, trường hợp hoặc nội dung liên quan.
- Cuối cùng ghi mục "Căn cứ pháp lý".
""".strip()

    @staticmethod
    def _clean_value(
        value: str,
        field_name: str,
    ) -> str:
        clean_value = str(value).strip()

        if not clean_value:
            raise ValueError(
                f"{field_name} không được để trống."
            )

        return clean_value

    def build(
        self,
        question: str,
        context: str,
    ) -> str:
        clean_question = self._clean_value(
            question,
            "question",
        )

        clean_context = self._clean_value(
            context,
            "context",
        )

        return f"""
{self.system_prompt}

================ CONTEXT ================

{clean_context}

============== END CONTEXT ==============

CÂU HỎI CỦA NGƯỜI DÙNG:

{clean_question}

Hãy trả lời dựa trên CONTEXT theo đúng các quy tắc trên.
""".strip()