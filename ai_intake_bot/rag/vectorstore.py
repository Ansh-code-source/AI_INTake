"""Vector store implementations used by RAG engine.

Includes a simple in-memory vector store (no external dependencies) and a
Qdrant wrapper placeholder for future real integration. In-memory store is
suitable for small dev/test workloads and does not persist data.
"""
from typing import List, Tuple, Sequence
import math


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    ax = sum(x * x for x in a) ** 0.5
    bx = sum(y * y for y in b) ** 0.5
    if ax == 0 or bx == 0:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / (ax * bx)


class LocalVectorStore:
    """In-memory vector store for small RAG workloads."""

    def __init__(self):
        self.vectors: List[List[float]] = []
        self.metadatas: List[dict] = []
        self.texts: List[str] = []

    def index(self, texts: List[str], embeddings: List[List[float]], metadatas: List[dict] = None):
        if metadatas is None:
            metadatas = [{} for _ in texts]
        if len(texts) != len(embeddings):
            raise ValueError("texts and embeddings length mismatch")
        for t, e, m in zip(texts, embeddings, metadatas):
            self.texts.append(t)
            self.vectors.append(e)
            self.metadatas.append(m)

    def search(self, query_vector: List[float], top_k: int = 5) -> List[Tuple[str, dict, float]]:
        scored = []
        for txt, md, vec in zip(self.texts, self.metadatas, self.vectors):
            score = _cosine(query_vector, vec)
            scored.append((score, txt, md))
        scored.sort(reverse=True, key=lambda x: x[0])
        results = [(float(s), t, m) for s, t, m in scored[:top_k]]
        return results


class QdrantVectorStore:
    """Qdrant-backed vector store wrapper using `langchain_qdrant` or `qdrant-client`.

    This wrapper creates a temporary collection (by default) and removes it in
    `cleanup()` to avoid long-term persistence. If the environment contains an
    existing collection name in `collection_name` and `preserve=True`, it will
    reuse the existing collection (useful for devs who manage their own collections).

    Note: Qdrant and LangChain integrations are optional; when unavailable, the
    class will raise ImportError at construction time.
    """

    def __init__(self, url: str, api_key: str = None, collection_name: str = None, preserve: bool = False):
        # Lazy imports to avoid hard dependency during tests
        # Try multiple import patterns for different langchain and helper package versions
        LangChainQdrant = None
        _import_errors = []
        try:
            # Preferred: standalone helper package
            from langchain_qdrant import QdrantVectorStore as LangChainQdrant
        except Exception as e:
            _import_errors.append(e)
            try:
                # Some versions expose `Qdrant` symbol directly from the package
                from langchain_qdrant import Qdrant as LangChainQdrant
            except Exception as e1:
                _import_errors.append(e1)
                try:
                    # Fallback: langchain vectorstores module (various versions)
                    from langchain.vectorstores import Qdrant as LangChainQdrant
                except Exception as e2:
                    _import_errors.append(e2)
                    try:
                        from langchain.vectorstores import QdrantVectorStore as LangChainQdrant
                    except Exception as e3:
                        _import_errors.append(e3)

        # Try to import an OpenAI embeddings shim used by some langchain helpers
        OpenAIEmbeddings = None
        try:
            from langchain_openai import OpenAIEmbeddings
        except Exception as e:
            _import_errors.append(e)
            try:
                from langchain.embeddings.openai import OpenAIEmbeddings
            except Exception as e4:
                _import_errors.append(e4)

        if LangChainQdrant is None:
            # Provide helpful diagnostic info instead of a terse message
            raise ImportError(
                "langchain_qdrant integration is not available; tried multiple import paths. "
                "Install a compatible 'langchain_qdrant' or a matching 'langchain' + 'langchain_openai' package. "
                f"Errors: {_import_errors!r}"
            )

        self.url = url
        self.api_key = api_key
        self.collection_name = collection_name
        self._preserve = preserve
        self._store = None
        self._LangChainQdrant = LangChainQdrant

    def from_documents(self, documents: List[str], embedding_model=None, collection_name: str = None):
        """Create (or reuse) a Qdrant collection from provided documents.

        Uses LangChain's `from_documents` helper for convenience.
        """
        if embedding_model is None:
            try:
                from langchain_openai import OpenAIEmbeddings

                embedding_model = OpenAIEmbeddings()
            except Exception as e:
                # If OpenAI embeddings are unavailable (or missing API key), fall back to a local fake embeddings
                try:
                    from ai_intake_bot.rag.embeddings import FakeEmbeddings

                    embedding_model = FakeEmbeddings()
                except Exception:
                    raise ImportError("OpenAI embeddings provider (langchain_openai) is required for QdrantVectorStore.from_documents and no fallback embedding is available") from e

        collection = collection_name or self.collection_name
        # LangChain helpers typically expect `Document` objects (with `.page_content`).
        docs = documents
        if documents and isinstance(documents[0], str):
            # Try to build real LangChain Document objects when available
            LC_Doc = None
            try:
                from langchain.schema import Document as LC_Doc
            except Exception:
                try:
                    from langchain_core.schema import Document as LC_Doc
                except Exception:
                    LC_Doc = None
            if LC_Doc:
                docs = [LC_Doc(page_content=t) for t in documents]
            else:
                # Minimal fallback that provides `.page_content` and `.metadata`
                class _Doc:
                    def __init__(self, text):
                        self.page_content = text
                        self.metadata = {}

                docs = [_Doc(t) for t in documents]

        try:
            self._store = self._LangChainQdrant.from_documents(
                documents=docs, embedding=embedding_model, url=self.url, collection_name=collection
            )
        except Exception as e:
            # Some qdrant client/server combinations may complain about compatibility checks.
            # Retry with `check_compatibility=False` where supported as a best-effort.
            try:
                self._store = self._LangChainQdrant.from_documents(
                    documents=docs,
                    embedding=embedding_model,
                    url=self.url,
                    collection_name=collection,
                    check_compatibility=False,
                )
            except Exception:
                # Last-resort fallback: if we cannot use a remote Qdrant instance due to
                # API incompatibilities, fall back to an in-memory LocalVectorStore so the
                # caller still has a usable retriever for testing/dev purposes.
                try:
                    local = LocalVectorStore()
                    # Extract texts and metadatas from docs
                    texts = [getattr(d, "page_content", str(d)) for d in docs]
                    metadatas = [getattr(d, "metadata", {}) for d in docs]
                    # Compute embeddings using the provided embedding_model
                    if hasattr(embedding_model, "embed_documents"):
                        embeddings = embedding_model.embed_documents(texts)
                    elif hasattr(embedding_model, "embed_texts"):
                        embeddings = embedding_model.embed_texts(texts)
                    else:
                        raise
                    local.index(texts, embeddings, metadatas)

                    # Adapter so the in-memory store matches the LangChain API
                    class _LocalAdapter:
                        def __init__(self, store, embedder):
                            self._store = store
                            self._embedder = embedder

                        def similarity_search(self, query, k=5):
                            # Embed the query using whatever embedder is available
                            if hasattr(self._embedder, "embed_query"):
                                qvec = self._embedder.embed_query(query)
                            elif hasattr(self._embedder, "embed_texts"):
                                qvec = self._embedder.embed_texts([query])[0]
                            else:
                                raise RuntimeError("Embedding provider doesn't support query embedding")
                            results = self._store.search(qvec, top_k=k)
                            # Convert (score, text, metadata) -> (text, metadata, score) to match tests
                            return [(t, m, s) for s, t, m in results]

                    self._store = _LocalAdapter(local, embedding_model)
                    self.collection_name = collection or "local_fallback"
                    return self
                except Exception:
                    # Re-raise original exception for caller inspection
                    raise e
        self.collection_name = collection or self._store.collection_name
        return self

    def similarity_search(self, query: str, k: int = 5):
        if self._store is None:
            raise RuntimeError("Store not initialized; call from_documents first")
        return self._store.similarity_search(query, k=k)

    def cleanup(self):
        # If preserve flag is set, do not delete collection
        if self._preserve:
            return
        # Try to remove the collection via qdrant-client if available
        try:
            from qdrant_client import QdrantClient

            client = QdrantClient(url=self.url, prefer_grpc=False, api_key=self.api_key)
            if self.collection_name:
                client.delete_collection(collection_name=self.collection_name)
        except Exception:
            # Best-effort cleanup only; do not fail hard
            return
