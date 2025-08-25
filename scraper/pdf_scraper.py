import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
import requests
from io import BytesIO
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import PyPDF2
from typing import List
import json
import uuid



HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
}

# Default file paths if not imported from config
DEFAULT_FILE_PATHS = [
    "data/FAQs.pdf"
    # "documents/manual.txt",
    # "documents/guide.pdf"
]

VECTOR_DB_DIR = "vector_store"

def clean_text(text: str) -> str:
    """Clean text by removing extra whitespace"""
    return ' '.join(text.split())

def read_pdf_file(file_path: str) -> str:
    """Read text content from PDF file"""
    try:
        print(f"[INFO] Reading PDF: {file_path}")
        text = ""
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return clean_text(text)
    except Exception as e:
        print(f"[ERROR] Failed to read PDF {file_path}: {e}")
        return ""

def read_text_file(file_path: str) -> str:
    """Read text content from text file"""
    try:
        print(f"[INFO] Reading text file: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        return clean_text(text)
    except Exception as e:
        print(f"[ERROR] Failed to read text file {file_path}: {e}")
        return ""

def fetch_file_content(file_path_or_url: str) -> str:
    """Fetch content from local file or URL based on file extension."""
    from io import BytesIO
    from urllib.parse import urlparse

    is_url = file_path_or_url.startswith("http://") or file_path_or_url.startswith("https://")
    file_extension = Path(urlparse(file_path_or_url).path).suffix.lower()

    try:
        if file_extension == '.pdf':
            if is_url:
                print(f"[INFO] Downloading PDF from URL: {file_path_or_url}")
                response = requests.get(file_path_or_url, headers=HEADERS, timeout=10)
                response.raise_for_status()
                pdf_stream = BytesIO(response.content)
                pdf_reader = PyPDF2.PdfReader(pdf_stream)
                text = ""
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                return clean_text(text)
            else:
                return read_pdf_file(file_path_or_url)

        elif file_extension in ['.txt', '.md']:
            if is_url:
                print(f"[INFO] Downloading text file from URL: {file_path_or_url}")
                response = requests.get(file_path_or_url, headers=HEADERS, timeout=10)
                response.raise_for_status()
                return clean_text(response.text)
            else:
                return read_text_file(file_path_or_url)

        else:
            print(f"[ERROR] Unsupported file type: {file_extension}")
            return ""

    except Exception as e:
        print(f"[ERROR] Failed to fetch file content from {file_path_or_url}: {e}")
        return ""


def split_text_into_sentence_chunks(text: str, chunk_size: int = 1) -> list:
    """
    Splits text into chunks consisting of `chunk_size` sentences each.
    Sentences are defined by splitting on '. ' (period + space).
    """
    # Split based on '. ', keep sentences clean and re-add period when joining
    sentences = [s.strip() for s in text.split('. ') if s.strip()]
    chunks = []
    for i in range(0, len(sentences), chunk_size):
        chunk = '. '.join(sentences[i:i + chunk_size])
        if not chunk.endswith('.'):
            chunk += '.'  # Add period if missing
        chunks.append(chunk)
    return chunks


def load_and_chunk_documents_to_json(file_paths: list = None, json_save_path: str = "data/chunked_docs.json", chunk_size: int = 1) -> list:
    """
    Loads text from given file paths, chunks by sentences, saves each chunk with unique id to JSON file,
    and returns list of langchain Document objects.
    """
    if file_paths is None:
        file_paths = DEFAULT_FILE_PATHS
    
    documents = []
    with open(json_save_path, "w", encoding="utf-8") as json_file:
        for file_path in file_paths:
            text = fetch_file_content(file_path)
            if not text:
                continue
            # Metadata for source file
            meta = {"source": file_path, "type": Path(file_path).suffix}
            # Sentence chunking
            chunks = split_text_into_sentence_chunks(text, chunk_size=chunk_size)
            for chunk in chunks:
                doc_id = str(uuid.uuid4())
                # Write to JSONL file
                json.dump({
                    "id": doc_id,
                    "page_content": chunk,
                    "metadata": meta
                }, json_file, ensure_ascii=False)
                json_file.write("\n")
                # Add to documents list
                documents.append(Document(page_content=chunk, metadata=meta))
    print(f"[INFO] Loaded, chunked, and saved {len(documents)} sentence chunks to {json_save_path}")
    return documents
