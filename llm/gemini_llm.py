from __future__ import annotations

from typing import Any

from google import genai
from google.genai import types

from llm.base_llm import BaseLLM


class GeminiLLM(BaseLLM):
    """
    Gemini wrapper sử dụng SDK google-genai.

    Chỉ trả kết quả khi model kết thúc bình thường
    với finish_reason=STOP.
    """

    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-2.5-flash",
        *,
        temperature: float = 0.1,
        top_p: float = 0.8,
        max_output_tokens: int = 4096,
        thinking_budget: int = 0,
    ) -> None:
        clean_api_key = str(api_key).strip()

        if not clean_api_key:
            raise ValueError(
                "api_key không được để trống."
            )

        if max_output_tokens < 256:
            raise ValueError(
                "max_output_tokens phải lớn hơn "
                "hoặc bằng 256."
            )

        self.model_name = model_name

        self.client = genai.Client(
            api_key=clean_api_key
        )

        self.generation_config = (
            types.GenerateContentConfig(
                temperature=temperature,
                top_p=top_p,
                max_output_tokens=max_output_tokens,
                thinking_config=types.ThinkingConfig(
                    thinking_budget=thinking_budget
                ),
            )
        )

    @staticmethod
    def _enum_value(
        value: Any,
    ) -> str:
        if value is None:
            return "UNKNOWN"

        enum_value = getattr(
            value,
            "value",
            value,
        )

        return str(enum_value)

    @staticmethod
    def _extract_text(
        candidate: Any,
    ) -> str:
        content = getattr(
            candidate,
            "content",
            None,
        )

        parts = getattr(
            content,
            "parts",
            None,
        ) or []

        text_parts: list[str] = []

        for part in parts:
            text = getattr(
                part,
                "text",
                None,
            )

            if text:
                text_parts.append(
                    str(text)
                )

        return "".join(
            text_parts
        ).strip()

    @staticmethod
    def _usage_summary(
        response: Any,
    ) -> str:
        usage = getattr(
            response,
            "usage_metadata",
            None,
        )

        if usage is None:
            return ""

        field_names = (
            "prompt_token_count",
            "candidates_token_count",
            "thoughts_token_count",
            "total_token_count",
        )

        values: list[str] = []

        for field_name in field_names:
            value = getattr(
                usage,
                field_name,
                None,
            )

            if value is not None:
                values.append(
                    f"{field_name}={value}"
                )

        return ", ".join(values)

    def generate(
        self,
        prompt: str,
    ) -> str:
        clean_prompt = str(
            prompt
        ).strip()

        if not clean_prompt:
            raise ValueError(
                "prompt không được để trống."
            )

        response = (
            self.client.models.generate_content(
                model=self.model_name,
                contents=clean_prompt,
                config=self.generation_config,
            )
        )

        candidates = list(
            getattr(
                response,
                "candidates",
                None,
            )
            or []
        )

        if not candidates:
            prompt_feedback = getattr(
                response,
                "prompt_feedback",
                None,
            )

            block_reason = self._enum_value(
                getattr(
                    prompt_feedback,
                    "block_reason",
                    None,
                )
            )

            raise RuntimeError(
                "Gemini không trả về candidate. "
                f"block_reason={block_reason}"
            )

        candidate = candidates[0]

        finish_reason = self._enum_value(
            getattr(
                candidate,
                "finish_reason",
                None,
            )
        )

        text = self._extract_text(
            candidate
        )

        if finish_reason != "STOP":
            usage_summary = self._usage_summary(
                response
            )

            message = (
                "Gemini không kết thúc bình thường: "
                f"finish_reason={finish_reason}"
            )

            if usage_summary:
                message += (
                    f" ({usage_summary})"
                )

            raise RuntimeError(message)

        if not text:
            raise RuntimeError(
                "Gemini kết thúc nhưng không trả "
                "về nội dung văn bản."
            )

        return text