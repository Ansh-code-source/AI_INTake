"""Text splitters for RAG document chunking.

Prefer `RecursiveCharacterTextSplitter` from `langchain_text_splitters` when
available; otherwise provide a simple fallback splitter.
"""
from typing import List


def split_text_simple(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """A naive splitter that slices text into overlapping chunks."""
    if not text:
        return []
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + chunk_size, length)
        chunks.append(text[start:end])
        start = max(end - overlap, end) if end < length else end
        if start == end:
            break
    return chunks


def split_documents_texts(texts: List[str], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    """Split multiple document texts into chunks using best available splitter."""
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = []
        for t in texts:
            docs = splitter.split_text(t)
            chunks.extend(docs)
        return chunks
    except Exception:
        # Fallback to simple splitter
        chunks = []
        for t in texts:
            chunks.extend(split_text_simple(t, chunk_size=chunk_size, overlap=chunk_overlap))
        return chunks