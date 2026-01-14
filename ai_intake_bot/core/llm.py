from typing import Optional, List
import json


class BaseLLM:
    def generate(self, prompt: str) -> str:
        raise NotImplementedError


class FakeLLM(BaseLLM):
    def __init__(self, responses: Optional[dict] = None):
        self.responses = responses or {}
        self.prompts_history: List[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts_history.append(prompt)
        return json.dumps({
            "reply": "This is a canned reply",
            "signals": {
                "empathy": 0.8,
                "clarity": 0.9,
                "aggression": 0.1,
                "urgency": 0.2,
            }
        })
