import os
from langchain_community.vectorstores import ElasticsearchStore
from langchain_community.chat_models import ChatOllama
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

# --- 1. Load environment variables if needed ---
load_dotenv()

# --- 2. Elasticsearch connection params ---
ELASTICSEARCH_URL = os.environ.get("ELASTICSEARCH_URL", "http://localhost:9200")
INDEX_NAME = os.environ.get("INDEX_NAME", "producer_assistant_knowledge")

# --- 3. Reconstruct the Vectorstore for retrieval ---
embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
embedding = HuggingFaceEmbeddings(model_name=embedding_model_name)

vectorstore = ElasticsearchStore(
    es_url=ELASTICSEARCH_URL,
    index_name=INDEX_NAME,
    embedding=embedding,
)

# --- 4. Set up Local LLM with Ollama ---
llm = ChatOllama(model="mistral")  # or "llama3", "openhermes", etc.

# --- 5. Set up conversational memory (optional, but nice!) ---
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key='answer',
)

# --- 6. Set up the Retrieval QA Chain ---
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    memory=memory,
    return_source_documents=True,
)

# --- 7. Simple REPL loop for user queries ---
print("🎛️ Producer Assistant Q&A")
print("Type 'exit' to quit.\n")

while True:
    question = input("🧑‍🎤 You: ")
    if question.strip().lower() in ["exit", "quit"]:
        break
    # result = qa_chain({"question": question})
    result = qa_chain.invoke({"question": question})
    print("\n🤖 Assistant:", result['answer'].strip())
    # Optionally print sources
    print("\n--- Sources ---")
    for doc in result['source_documents']:
        print("•", doc.metadata)
    print()

print("Bye! 👋")