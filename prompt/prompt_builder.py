
class PromptBuilder:
    def __init__(self):
        self.system_prompt="""
        Bạn là trợ lý AI tư vấn Luật Dân sự Việt Nam.

        QUY TẮC
        
        1. Chỉ sử dụng thông tin trong CONTEXT.
        2. Không tự suy diễn.
        3. Không dùng kiến thức ngoài CONTEXT.
        4. Nếu không có thông tin thì trả lời:
        "Tôi không tìm thấy thông tin trong tài liệu."
        5. Nếu có nhiều điều luật liên quan hãy tổng hợp.
        6. Trích dẫn điều luật nếu có.
        7. Trả lời bằng Markdown.
        """

    def build(self, question, context):
        return f"""
        {self.system_prompt}
        
        # CONTEXT
        
        {context}
        
        # USER QUESTION
        
        {question}
        
        # ASSISTANT
        """