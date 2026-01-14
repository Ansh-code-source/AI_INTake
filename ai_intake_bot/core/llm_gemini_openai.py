import os
from openai import OpenAI

from .llm import BaseLLM
from .llm_registry import register_llm


class GeminiOpenAICompatLLM(BaseLLM):
    def __init__(
        self,
        model: str = "gemini-1.5-flash",
        api_key: str | None = None,
    ):
        key = api_key or os.getenv("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("Gemini API key not provided")

        self.client = OpenAI(
            api_key=key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
        self.model = model

    def generate(self, prompt: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": prompt}
            ],
        )
        return resp.choices[0].message.content


# ✅ registration happens LAST
register_llm("gemini", GeminiOpenAICompatLLM)
