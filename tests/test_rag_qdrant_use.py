from types import SimpleNamespace
from ai_intake_bot.core.engine import IntakeBot
from ai_intake_bot.core.llm import FakeLLM


class FakeQdrant:
    def __init__(self):
        self.from_docs_called = False
        self.cleanup_called = False

    def from_documents(self, documents, embedding_model=None, collection_name=None):
        self.from_docs_called = True
        return self

    def similarity_search(self, query, k=5):
        # Return objects with page_content attribute
        return [SimpleNamespace(page_content=f"Found chunk about {query} - {i}", metadata={"source": "fake"}) for i in range(3)]

    def cleanup(self):
        self.cleanup_called = True


def test_rag_engine_uses_qdrant(monkeypatch, tmp_path):
    # Create a small text file as a document
    doc = tmp_path / "doc.txt"
    doc.write_text("Invoice 1234: customer reports incorrect charge on billing statement.")

    # Monkeypatch the QdrantVectorStore to use our FakeQdrant
    import ai_intake_bot.rag.vectorstore as vmod

    monkeypatch.setattr(vmod, "QdrantVectorStore", lambda url, api_key=None, collection_name=None: FakeQdrant())

    fake = FakeLLM(responses={"use the following retrieved documents": {"reply": "Grounded reply from Qdrant"}})

    bot = IntakeBot(
        mode="rag",
        template="complaint",
        persona="support_agent",
        problem=None,
        api_key="sk-test",
        qdrant_url="http://localhost:6333",
        qdrant_api_key=None,
        files=[str(doc)],
        selection_probability=0.5,
        enable_alerts=True,
        extra_system_prompt=None,
    )
    bot.set_llm(fake)
    out = bot.handle("What happened to my invoice?")

    # Check the LLM received grounding text from Qdrant results
    assert fake.last_prompt is not None
    assert "Found chunk about" in fake.last_prompt
    # Validate output
    assert out["reply"] == "Grounded reply from Qdrant"
