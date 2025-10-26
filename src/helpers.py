from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from typing import List


def pdf_text_extractor(data):
    loader = DirectoryLoader(
        data,
        glob = "*.pdf",
        loader_cls = PyPDFLoader
    )
    documents = loader.load()
    return documents 


def filter_min_docs(docs: List[Document]) -> List[Document]:
    min_docs: List[Document] = []
    for doc in docs:
        src = doc.metadata.get("source")
        min_docs.append(
            Document(
            page_content = doc.page_content,
            metadata = {"source": src}
            )
        )
    return min_docs


def text_split(min_docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 20
    )
    texts_chunk = text_splitter.split_documents(min_docs)
    return texts_chunk

def download_embeddings():
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name
        )
    return embeddings