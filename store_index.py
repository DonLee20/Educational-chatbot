from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from src.helpers import pdf_text_extractor, filter_min_docs, text_split, download_embeddings
import os

load_dotenv()
load_dotenv(dotenv_path=".env")

# Fetch the keys from environment
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

pinecone_api_key = os.getenv("PINECONE_API_KEY")

pc = Pinecone(api_key=pinecone_api_key)

extracted_data = pdf_text_extractor('D:\CHATBOT-01\Educational-chatbot\data')
min_docs = filter_min_docs(extracted_data)
texts_chunk = text_split(min_docs)

embedding = download_embeddings()

index_name = "educational-chatbot"

if not pc.has_index(index_name):
    pc.create_index(
        name = index_name,
        dimension = 384,
        metric = "cosine",
        spec = ServerlessSpec(
            cloud = "aws", region = "us-east-1"
            )
    )
index = pc.Index(index_name)

vector_store = PineconeVectorStore.from_documents(
    documents = texts_chunk,
    embedding = embedding,
    index_name = index_name
)

