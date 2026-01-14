from ai_intake_bot.core.engine import IntakeBot
from ai_intake_bot.core.llm import FakeLLM


def test_rag_engine_local_retrieval_injects_grounding():
    # Create a small text file as a document
    import tempfile, pathlib

    tmp = tempfile.NamedTemporaryFile(delete=False, mode="w", encoding="utf-8", suffix=".txt")
    tmp.write("Invoice 1234: customer reports incorrect charge on billing statement.")
    tmp.close()

    fake = FakeLLM(responses={"use the following retrieved documents": {"reply": "Grounded reply"}})

    bot = IntakeBot(
        mode="rag",
        template="complaint",
        persona="support_agent",
        problem=None,
        api_key="sk-test",
        qdrant_url=None,
        qdrant_api_key=None,
        files=[tmp.name],
        selection_probability=0.5,
        enable_alerts=True,
        extra_system_prompt=None,
    )
    bot.set_llm(fake)
    out = bot.handle("What happened to my invoice?")
    assert out["reply"] == "Grounded reply"
