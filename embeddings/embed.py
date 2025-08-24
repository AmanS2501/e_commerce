from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema.runnable import RunnableLambda, RunnableSequence

pdf_path = r"C:\Users\Aman Sheikh\Documents\GitHub\e_commerce\data\FAQs.pdf"
loader = PyPDFLoader(pdf_path)
load_docs_runnable = RunnableLambda(lambda _: loader.load())

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=70
)
splitter_runnable = RunnableLambda(text_splitter.split_documents)

embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

def store_to_chroma(docs):
    db = Chroma.from_documents(
        docs,
        embeddings,
        persist_directory="chromadb_FAQ",
        collection_name="faq_collection"
    )
    db.persist()
    print(f"Stored {db._collection.count()} documents in collection {db._collection.name}")
    return db

store_runnable = RunnableLambda(store_to_chroma)

pipeline = RunnableSequence(
    load_docs_runnable,
    splitter_runnable,
    store_runnable,
)

vector_db = pipeline.invoke(None)
print("FAQ embeddings stored in ChromaDB!")
