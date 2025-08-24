from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


def get_chroma_retriever(persist_directory: str, embedding_model_name: str, collection_name: str = "default_collection", k: int = 3):

    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

    vector_db = Chroma(
        persist_directory=persist_directory,
        collection_name=collection_name,
        embedding_function=embeddings,
    )

    # print("Collection name:", vector_db._collection.name)
    # print("Printing all document IDs for sanity check:")
    # try:
    #     print(vector_db._collection.get(limit=5))  # Show first 5 docs' info
    # except Exception as e:
    #     print("Error fetching docs:", e)


    retriever = vector_db.as_retriever(search_kwargs={"k": k})
    return retriever


def similarity_search(retriever, query: str):
    print(f"Running similarity search for query: {query}")
    results = retriever.invoke(query)
    # print("DEBUG: Retrieved", len(results), "results")
    # if not results:
    #     print("No documents found matching this query.")
    for i, doc in enumerate(results):
        print(f"Result {i+1}:")
        print(doc.page_content)
        print("-" * 40)


if __name__ == "__main__":
    
    PERSIST_DIRECTORY = r"C:\Users\Aman Sheikh\Documents\GitHub\e_commerce\embeddings\chromadb_FAQ"
    EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    COLLECTION_NAME = "faq_collection"
    TOP_K = 3

    retriever = get_chroma_retriever(PERSIST_DIRECTORY, EMBEDDING_MODEL_NAME, COLLECTION_NAME, TOP_K)

    user_query = "Within how many days can I return my order?"
    similarity_search(retriever, user_query)
