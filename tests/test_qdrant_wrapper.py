import os
import pytest
from ai_intake_bot.rag.vectorstore import QdrantVectorStore


def test_qdrant_wrapper_not_installed():
    # If langchain_qdrant isn't installed, constructing QdrantVectorStore should raise ImportError
    try:
        import langchain_qdrant  # type: ignore
        # If the package is installed in the test env, the constructor should succeed
        QdrantVectorStore(url="http://localhost:6333")
    except Exception:
        # Otherwise, ensure we raise ImportError as expected
        with pytest.raises(ImportError):
            QdrantVectorStore(url="http://localhost:6333")


@pytest.mark.skipif(os.getenv("RUN_QDRANT_INTEGRATION") != "1", reason="Integration test; enable when a Qdrant server and langchain packages are available")
def test_qdrant_wrapper_from_documents_integration():
    # Integration test skeleton: requires langchain_qdrant, langchain_openai, and a running Qdrant server
    texts = ["Doc about billing.", "Invoice details and charge info."]
    q = QdrantVectorStore(url="http://localhost:6333", collection_name="test_collection", preserve=True)
    q.from_documents(texts)
    results = q.similarity_search("billing question")
    assert isinstance(results, list)
    q.cleanup()
