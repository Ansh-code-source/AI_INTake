from ai_intake_bot.rag.embeddings import FakeEmbeddings
from ai_intake_bot.rag.vectorstore import LocalVectorStore


def test_local_vectorstore_index_and_search():
    texts = ["This is a test document.", "Another document about billing."]
    emb = FakeEmbeddings()
    vectors = emb.embed_texts(texts)
    vs = LocalVectorStore()
    vs.index(texts, vectors)

    qvec = emb.embed_texts(["billing question"])[0]
    results = vs.search(qvec, top_k=2)
    assert len(results) == 2
    assert isinstance(results[0][0], float)
    assert isinstance(results[0][1], str)
