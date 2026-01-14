from typing import List, Dict, Any


class ExpertEvalSession:
    def __init__(self, persona: str, problem: Dict[str, Any]):
        self.persona = persona
        self.problem = problem
        self.history: List[Dict[str, str]] = []
        self.turns = 0
        self.finished = False

    def add_bot(self, text: str):
        self.history.append({"role": "bot", "text": text})
        self.turns += 1

    def add_user(self, text: str):
        self.history.append({"role": "user", "text": text})
