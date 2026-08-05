import re
import unicodedata

class TextCleaner:
    def clean(self, text: str) -> str:
        # Unicode
        text = unicodedata.normalize("NFC", text)

        # Chuẩn hóa xuống dòng
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Chuẩn hóa khoảng trắng
        text = re.sub(r"[ \t]+", " ", text)

        # Xóa khoảng trắng cuối dòng
        text = re.sub(r" *\n *", "\n", text)

        # Loại ký tự đặc biệt không cần thiết
        text = re.sub(r"[■◆►•]", "", text)
        return text.strip()