#!/usr/bin/env python3
"""Demo: RAG with LangChain loaders, splitter, OpenAI embeddings, and Qdrant.

This script follows the pattern you provided but adapts it into a safe, runnable
demo. It is opt-in and requires:
- QDRANT_URL (e.g., http://localhost:6333)
- OPENAI_API_KEY
- langchain_qdrant and langchain_openai installed

Note: the script creates an ephemeral collection by default; you can set
VECTOR_COLLECTION_NAME to reuse an existing collection.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

VECTOR_DB_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
VECTOR_COLLECTION_NAME = os.getenv("VECTOR_COLLECTION_NAME", "ai_intake_demo")
PDF_PATH = os.getenv("DEMO_PDF_PATH") or str(Path(__file__).parent / "nodejs.pdf")

if __name__ == "__main__":
    # Lazy imports to avoid hard dependency unless script is run
    try:
        from langchain_community.document_loaders import PyPDFLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_openai import OpenAIEmbeddings
        from langchain_qdrant import QdrantVectorStore
        from openai import OpenAI
    except Exception as e:
        raise RuntimeError("This demo requires langchain_qdrant, langchain_openai, PyPDFLoader, and OpenAI packages") from e

    loader = PyPDFLoader(PDF_PATH)
    docs = loader.load()
    print(f"Loaded {len(docs)} documents from {PDF_PATH}")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=400)
    texts = text_splitter.split_documents(docs)
    print(f"Split into {len(texts)} chunks")

    openai_client = OpenAI()
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

    vector_store = QdrantVectorStore.from_documents(
        documents=texts, embedding=embeddings, url=VECTOR_DB_URL, collection_name=VECTOR_COLLECTION_NAME
    )

    user_query = input("Ask something about the PDF: ")
    search_results = vector_store.similarity_search(query=user_query)

    context = "\n\n\n".join([
        f"Page Content: {res.page_content}\nPage Number: {res.metadata.get('page_label')}\nFile Location: {res.metadata.get('source')}"
        for res in search_results
    ])

    SYSTEM_PROMPT = f"""
 You are an assistant. Answer the user's query based only on the following context retrieved from a PDF file (include page references where appropriate).

 Context:
 {context}
"""

    response = openai_client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o"),
        messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_query}],
    )

    print(response.choices[0].message.content)
