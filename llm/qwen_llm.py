from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from llm.base_llm import BaseLLM


class QwenLLM(BaseLLM):
    """
    Local Qwen LLM wrapper using llama-server.

    The model is loaded once by llama-server.
    This class sends prompts to the local
    OpenAI-compatible chat completions endpoint.
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8080",
        model_name: str = "qwen",
        *,
        temperature: float = 0.2,
        top_p: float = 0.8,
        top_k: int = 20,
        max_output_tokens: int = 1024,
        timeout: int = 180,
        disable_thinking: bool = True,
    ) -> None:
        self.base_url = str(base_url).rstrip("/")
        self.model_name = str(model_name).strip()

        if not self.model_name:
            raise ValueError(
                "model_name must not be empty."
            )

        if max_output_tokens < 1:
            raise ValueError(
                "max_output_tokens must be >= 1."
            )

        if timeout < 1:
            raise ValueError(
                "timeout must be >= 1."
            )

        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.top_k = int(top_k)
        self.max_output_tokens = int(
            max_output_tokens
        )
        self.timeout = int(timeout)
        self.disable_thinking = bool(
            disable_thinking
        )

    def _build_prompt(
        self,
        prompt: str,
    ) -> str:
        clean_prompt = str(prompt).strip()

        if not clean_prompt:
            raise ValueError(
                "prompt must not be empty."
            )

        if self.disable_thinking:
            return (
                "/no_think\n"
                + clean_prompt
            )

        return clean_prompt

    @staticmethod
    def _extract_text(
        response: dict[str, Any],
    ) -> str:
        choices = response.get(
            "choices",
            [],
        )

        if not choices:
            raise RuntimeError(
                "Qwen returned no choices."
            )

        message = choices[0].get(
            "message",
            {},
        )

        text = str(
            message.get(
                "content",
                "",
            )
        ).strip()

        if not text:
            raise RuntimeError(
                "Qwen returned empty content."
            )

        return text

    def generate(
        self,
        prompt: str,
    ) -> str:
        final_prompt = self._build_prompt(
            prompt
        )

        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": final_prompt,
                }
            ],
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "max_tokens": (
                self.max_output_tokens
            ),
            "stream": False,
        }

        request_data = json.dumps(
            payload,
            ensure_ascii=False,
        ).encode("utf-8")

        request = urllib.request.Request(
            (
                self.base_url
                + "/v1/chat/completions"
            ),
            data=request_data,
            headers={
                "Content-Type": (
                    "application/json; "
                    "charset=utf-8"
                )
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(
                request,
                timeout=self.timeout,
            ) as response:
                response_data = json.loads(
                    response.read().decode(
                        "utf-8"
                    )
                )

        except urllib.error.HTTPError as error:
            detail = error.read().decode(
                "utf-8",
                errors="replace",
            )

            raise RuntimeError(
                "Qwen server HTTP error "
                f"{error.code}: {detail}"
            ) from error

        except urllib.error.URLError as error:
            raise RuntimeError(
                "Cannot connect to Qwen server at "
                f"{self.base_url}. "
                "Make sure llama-server is running."
            ) from error

        return self._extract_text(
            response_data
        )
