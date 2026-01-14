import os
import pytest

from ai_intake_bot.core.llm import OpenAIChatLLM


@pytest.mark.skipif(os.getenv("RUN_OPENAI_INTEGRATION") != "1", reason="OpenAI integration disabled")
def test_openai_chat_llm_integration():
    # This test runs only when RUN_OPENAI_INTEGRATION=1 and OPENAI_API_KEY is set.
    model = os.getenv("OPENAI_MODEL", "gpt-4o")
    llm = OpenAIChatLLM(model=model)
    out = llm.generate("Say hello in one sentence.")
    assert isinstance(out, str)
    assert len(out) > 0
