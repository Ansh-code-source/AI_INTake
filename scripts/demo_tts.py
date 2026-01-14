#!/usr/bin/env python3
"""Demo: TTS (voice) demo using macOS 'say' or pyttsx3 (opt-in).

This demo is local-only and does not use remote services. It demonstrates how
`IntakeBot.set_voice()` can be used to speak responses.
"""
import os
from dotenv import load_dotenv
load_dotenv()

from ai_intake_bot.core.engine import IntakeBot
from ai_intake_bot.core.llm import FakeLLM
from ai_intake_bot.core.voice import MacOSSayVoice, PyTTSX3Voice, NoOpVoice


def main():
    bot = IntakeBot(
        mode="persona",
        template="survey",
        persona="support_agent",
        problem=None,
        api_key="sk-test",
        qdrant_url=None,
        qdrant_api_key=None,
        files=None,
        selection_probability=None,
        enable_alerts=False,
        extra_system_prompt=None,
    )
    bot.set_llm(FakeLLM())

    # Prefer macOS say; fall back to pyttsx3; ultimately NoOp
    try:
        v = MacOSSayVoice()
    except Exception:
        try:
            v = PyTTSX3Voice()
        except Exception:
            v = NoOpVoice()

    bot.set_voice(v)

    out = bot.handle("Hello, please provide a short greeting")
    print("Reply:", out["reply"])
    print("Voice used:", v.__class__.__name__)


if __name__ == '__main__':
    main()
