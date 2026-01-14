"""Embedding interfaces and fake embeddings for testing."""
from typing import List


class FakeEmbeddings:
    """Deterministic fake embeddings for tests.

    Produces small fixed-length vectors based on character codes.
    """

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        vectors = []
        for t in texts:
            # Simple deterministic embedding: sum of ords mod constants
            s = sum(ord(c) for c in t)
            vectors.append([float((s % (i + 3)) / (i + 3)) for i in range(8)])
        return vectors

    # Compatibility shim for langchain-style embeddings
    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        # Documents may be strings or objects with `.page_content`
        texts = []
        for d in documents:
            if isinstance(d, str):
                texts.append(d)
            else:
                texts.append(getattr(d, "page_content", str(d)))
        return self.embed_texts(texts)

    def embed_query(self, text: str) -> List[float]:
        # Single-query embedding for similarity search
        return self.embed_texts([text])[0]


# Adapter for real embedding providers would be added in Phase 10
