# ingest_data.py

import json
from pathlib import Path
from langchain_community.vectorstores import ElasticsearchStore
from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from dotenv import load_dotenv

# Load .env if you're using OpenAI API key
load_dotenv()

# === 1. Load JSON Data ===
DATA_PATH = Path("../data/hallucinated/production_knowledge.json")

with open(DATA_PATH, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

# Convert to LangChain Document format
docs = [
    Document(page_content=item["text"], metadata=item["metadata"])
    for item in raw_data
]

# === 2. Choose Embedding Model ===
# Option A: OpenAI
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


# Option B: Local (uncomment if using SBERT)
# from langchain.embeddings import HuggingFaceEmbeddings
# embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# === 3. Connect to Elasticsearch ===
ELASTICSEARCH_URL = "http://localhost:9200"
INDEX_NAME = "producer_assistant_knowledge"

# Initialize store and index documents
vectorstore = ElasticsearchStore.from_documents(
    docs,
    embedding=embedding,
    es_url=ELASTICSEARCH_URL,
    index_name=INDEX_NAME
)

print(f"✅ Indexed {len(docs)} documents into Elasticsearch index: {INDEX_NAME}")
