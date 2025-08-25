# from scraper.pdf_scraper import load_documents  # Import from your scraper module
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_chroma import Chroma

# # ---- Configuration ----
# DEFAULT_FILE_PATHS = [
#     "data/FAQs.pdf"
# ]
# PERSIST_DIRECTORY = "chromadb_FAQ"
# COLLECTION_NAME = "faq_collection"
# EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# # ---- Document Loading ----
# print("[INFO] Loading documents using scraper...")
# documents = load_documents(DEFAULT_FILE_PATHS)
# print(f"[INFO] Loaded {len(documents)} documents.")

# if not documents:
#     raise ValueError("No documents loaded. Please check your file paths and PDF content.")

# # ---- Text Splitting ----
# print("[INFO] Splitting documents into chunks...")
# splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=70)
# chunks = splitter.split_documents(documents)
# print(f"[INFO] Created {len(chunks)} chunks.")

# # ---- Embedding Setup ----
# embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

# # ---- Vector Store and Persist ----
# print("[INFO] Storing chunks in ChromaDB...")
# db = Chroma.from_documents(
#     chunks,
#     embeddings,  # Correct usage: embeddings as 2nd argument
#     persist_directory=PERSIST_DIRECTORY,
#     collection_name=COLLECTION_NAME,
# )

# # db.persist()
# # print(f"[SUCCESS] {len(chunks)} chunks embedded and stored in collection '{COLLECTION_NAME}' at '{PERSIST_DIRECTORY}'.")

# # Optional: Print stored doc count for sanity check
# print(f"Collection doc count: {db._collection.count()}")
# print(f"Sample docs: {db._collection.get(limit=3)}")



from scraper.pdf_scraper import load_and_chunk_documents_to_json
from langchain.text_splitter import NLTKTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


DEFAULT_FILE_PATHS = [
    "data/FAQs.pdf"
]
PERSIST_DIRECTORY = "chromadb_FAQ"
COLLECTION_NAME = "faq_collection"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


print("[INFO] Loading documents using scraper...")
documents = load_and_chunk_documents_to_json(DEFAULT_FILE_PATHS)
print(f"[INFO] Loaded {len(documents)} documents.")

if not documents:
    raise ValueError("No documents loaded. Please check your file paths and PDF content.")


embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


print("[INFO] Storing chunks in ChromaDB...")
db = Chroma.from_documents(
    documents,
    embeddings,
    persist_directory=PERSIST_DIRECTORY,
    collection_name=COLLECTION_NAME,
)
print(f"Collection doc count: {db._collection.count()}")
