import os
import streamlit as st
from langchain_community.vectorstores import ElasticsearchStore
from langchain_community.chat_models import ChatOllama
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

ELASTICSEARCH_URL = os.environ.get("ELASTICSEARCH_URL", "http://localhost:9200")
INDEX_NAME = os.environ.get("INDEX_NAME", "producer_assistant_knowledge")


def create_chain():
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = ElasticsearchStore(
        es_url=ELASTICSEARCH_URL,
        index_name=INDEX_NAME,
        embedding=embedding,
    )
    llm = ChatOllama(model="mistral")
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        return_source_documents=True,
    )


if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = create_chain()

st.title("\U0001F399\ufe0f Producer's Assistant")

question = st.text_input("Ask a question about music production:")

if question:
    result = st.session_state.qa_chain.invoke({"question": question})
    st.markdown("**Answer:** " + result["answer"].strip())
    st.markdown("---")
    st.markdown("**Sources**")
    for doc in result["source_documents"]:
        st.write(doc.metadata)
