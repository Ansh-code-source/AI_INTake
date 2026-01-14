from ai_intake_bot.core.engine import IntakeBot
from ai_intake_bot.core.llm import FakeLLM


def test_persona_engine_returns_contract():
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
    # Inject FakeLLM to capture the prompt
    bot.set_llm(FakeLLM())
    out = bot.handle("Hello, I need help")
    assert isinstance(out, dict)
    assert "reply" in out and "structured_data" in out
    assert isinstance(out["signals"], dict)


def test_expert_eval_injects_problem_prompt():
    bot = IntakeBot(
        mode="persona",
        template="expert_eval",
        persona="expert_reviewer",
        problem={"description": "Customer cannot login", "emotional_state": "frustrated", "goals": ["restore access"]},
        api_key="sk-test",
        qdrant_url=None,
        qdrant_api_key=None,
        files=None,
        selection_probability=None,
        enable_alerts=True,
        extra_system_prompt=None,
    )
    fake = FakeLLM()
    bot.set_llm(fake)
    _ = bot.handle("Please evaluate this scenario")
    # Role-play call should include the problem; the FakeLLM stores prompt history
    assert any("Customer cannot login" in p for p in fake.prompts_history)
