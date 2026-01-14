"""Retriever builder used by RAG engine.

Provides a simple in-memory retriever suitable for tests and small dev runs.
"""
from typing import List
from .embeddings import FakeEmbeddings
from .vectorstore import LocalVectorStore


class Retriever:
    def __init__(self, vectorstore: LocalVectorStore, embeddings: FakeEmbeddings):
        self.vs = vectorstore
        self.emb = embeddings

    def retrieve(self, query: str, top_k: int = 5) -> List[str]:
        vec = self.emb.embed_texts([query])[0]
        results = self.vs.search(vec, top_k=top_k)
        texts = []
        for item in results:
            # support both (score, text, meta) and (text, meta, score)
            if isinstance(item[0], float):
                _, text, _ = item
            else:
                text, _, _ = item
            texts.append(text)
        return texts


def build_retriever_from_texts(texts: List[str], embeddings: FakeEmbeddings = None) -> Retriever:
    emb = embeddings or FakeEmbeddings()
    vs = LocalVectorStore()
    embs = emb.embed_texts(texts)
    vs.index(texts, embs)
    return Retriever(vs, emb)
