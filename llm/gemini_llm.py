import google.generativeai as genai
from llm.base_llm import BaseLLM

class GeminiLLM(BaseLLM):
    def __init__(self, api_key, model_name="gemini-2.5-flash"):
        genai.configure(api_key=api_key)

        self.model = genai.GenerativeModel(
            model_name
        )
        self.generation_config = {
            "temperature": 0.1,
            "top_p": 0.8,
            "max_output_tokens": 2048
        }
    def generate(self, prompt: str):
        response = self.model.generate_content(
            prompt,
            generation_config=self.generation_config
        )

        return response.text