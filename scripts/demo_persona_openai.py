#!/usr/bin/env python3
"""Demo: Persona expert_eval using OpenAI (opt-in).

Requirements:
- Set OPENAI_API_KEY in your environment
- Optional: set OPENAI_MODEL (default gpt-4o)

This script demonstrates injecting a real LLM (OpenAIChatLLM) into IntakeBot
and running an `expert_eval` persona flow. It does not persist secrets.
"""
import os
import json
from dotenv import load_dotenv

load_dotenv()

from ai_intake_bot.core.engine import IntakeBot
from ai_intake_bot.core.llm import OpenAIChatLLM
from ai_intake_bot.core.personas import Problem


def main():
    model = os.getenv("OPENAI_MODEL", "gpt-4o")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required to run this demo")

    llm = OpenAIChatLLM(model=model, api_key=api_key)

    bot = IntakeBot(
        mode="persona",
        template="expert_eval",
        persona="expert_reviewer",
        problem={
            "description": "User cannot login to the account",
            "emotional_state": "frustrated",
            "goals": ["restore access"],
            "constraints": ["verify identity via email"],
        },
        api_key="sk-DEMO",
        qdrant_url=None,
        qdrant_api_key=None,
        files=None,
        selection_probability=0.6,
        enable_alerts=True,
        extra_system_prompt="Be concise and produce JSON that matches the SDK output contract",
    )

    bot.set_llm(llm)

    out = bot.handle("Please evaluate this user's message and suggest next steps")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
