from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder

def get_chroma_retriever(persist_directory: str, embedding_model_name: str, collection_name: str = "faq_collection", k: int = 8):
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
    vector_db = Chroma(
        persist_directory=persist_directory,
        collection_name=collection_name,
        embedding_function=embeddings,
    )
    retriever = vector_db.as_retriever(search_kwargs={"k": k})
    return retriever

def rerank(query: str, retrieved_docs, top_k_re_rank=3, cross_encoder_model="cross-encoder/ms-marco-MiniLM-L-6-v2"):
    # Prepare cross encoder
    cross_encoder = CrossEncoder(cross_encoder_model)
    # Prepare pairs for reranking
    pairs = [(query, doc.page_content) for doc in retrieved_docs]
    if not pairs:
        return []
    # Cross-encode scores
    scores = cross_encoder.predict(pairs)
    # Attach scores and rerank
    reranked = [
        (score, doc) for score, doc in sorted(zip(scores, retrieved_docs), key=lambda x: x[0], reverse=True)
    ]
    return reranked[:top_k_re_rank]

def similarity_search_with_reranking(retriever, query: str, rerank_top_n=3):
    print(f"Running similarity search for query: {query}")
    retrieved_docs = retriever.invoke(query)
    print(f"DEBUG: Retrieved {len(retrieved_docs)} candidates (pre-rerank)")

    if not retrieved_docs:
        print("No documents found matching this query.")
        return

    # Re-rank
    reranked = rerank(query, retrieved_docs, top_k_re_rank=rerank_top_n)
    print(f"Top {rerank_top_n} re-ranked results:")
    for i, (score, doc) in enumerate(reranked):
        print(f"Result {i+1}: Score: {score:.4f}")
        print(doc.page_content)
        print("-" * 40)

if __name__ == "__main__":
    PERSIST_DIRECTORY = r"C:\Users\Aman Sheikh\Documents\GitHub\e_commerce\chromadb_FAQ"
    EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    COLLECTION_NAME = "faq_collection"
    TOP_K_RETRIEVER = 8
    TOP_K_RERANK = 3

    retriever = get_chroma_retriever(PERSIST_DIRECTORY, EMBEDDING_MODEL_NAME, COLLECTION_NAME, TOP_K_RETRIEVER)
    user_query = "within how many days can I return a product?"
    similarity_search_with_reranking(retriever, user_query, TOP_K_RERANK)
