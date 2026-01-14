"""Simple LLM abstraction used by engines.

Provides a FakeLLM for tests and a minimal adapter interface for real LLMs.
"""
from typing import Optional, List
import json
import os


class BaseLLM:
    """Minimal LLM interface used by engines."""

    def generate(self, prompt: str) -> str:
        raise NotImplementedError


class FakeLLM(BaseLLM):
    """A deterministic fake LLM that returns JSON responses for testing.

    It supports an optional `responses` mapping (substring -> JSON-serializable
    object) so tests can simulate different LLM outputs depending on the prompt.
    It stores the last prompt and a history of prompts for inspection.
    """

    def __init__(self, responses: Optional[dict] = None):
        self.last_prompt: Optional[str] = None
        self.prompts_history: List[str] = []
        self.responses = responses or {}

    def generate(self, prompt: str) -> str:
        self.last_prompt = prompt
        self.prompts_history.append(prompt)
        # Find a matching response by substring key
        for key, resp in self.responses.items():
            if key.lower() in prompt.lower():
                return json.dumps(resp)

        # Default response includes signals and actions
        response = {
            "reply": "This is a canned reply",
            "structured_data": {"example_field": "example_value"},
            "signals": {"empathy": 0.8, "clarity": 0.9, "aggression": 0.1, "urgency": 0.2},
            "candidate_score": 0.85,
            "recommended_selection": True,
            "recommended_actions": [
                {"type": "ALERT", "priority": "HIGH", "reason": "Test: needs attention"}
            ],
        }
        return json.dumps(response)


class OpenAIChatLLM(BaseLLM):
    """Adapter for the OpenAI Python client. This is opt-in: instantiate and
    pass to `IntakeBot.set_llm()` to use a real LLM. It will not be used by
    default, and tests are skipped unless environment variables are present.

    Requires the `openai` package (OpenAI client) and an env var `OPENAI_API_KEY`.
    """

    def __init__(self, model: str = "gpt-4o", api_key: Optional[str] = None):
        try:
            from openai import OpenAI
        except Exception as e:
            raise ImportError("openai package is required to use OpenAIChatLLM") from e

        # Prefer explicit api_key param; otherwise rely on environment
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY must be set to use OpenAIChatLLM")

        self.client = OpenAI()
        self.model = model

    def generate(self, prompt: str) -> str:
        # We use the chat completions API; send the whole prompt as a system message.
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": prompt}],
        )
        # New OpenAI client uses choices[0].message.content
        content = resp.choices[0].message.content
        return content
