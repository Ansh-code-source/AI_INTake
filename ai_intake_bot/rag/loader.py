"""Document loaders for RAG.

Provides a PDF loader using `langchain` when available; otherwise falls back to
reading plain text files. Loaders return a list of strings (document contents).
"""
from typing import List


def load_documents(paths: List[str]) -> List[str]:
    texts: List[str] = []
    try:
        # Lazy import of langchain's loader to avoid hard dependency in tests
        from langchain.document_loaders import PyPDFLoader

        for p in paths:
            if p.lower().endswith(".pdf"):
                loader = PyPDFLoader(p)
                docs = loader.load()
                for d in docs:
                    texts.append(d.page_content)
            else:
                with open(p, "r", encoding="utf-8") as fh:
                    texts.append(fh.read())
    except Exception:
        # Fallback: read as plain text
        for p in paths:
            with open(p, "r", encoding="utf-8") as fh:
                texts.append(fh.read())

    return texts
