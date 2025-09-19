import os
import streamlit as st
from dotenv import load_dotenv
from transformers import pipeline
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone

# Load environment
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
HF_EMBEDDING_MODEL = os.getenv("HF_EMBEDDING_MODEL")

# Init Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

# Embeddings + retriever
embeddings = HuggingFaceEmbeddings(model_name=HF_EMBEDDING_MODEL)
vector_store = PineconeVectorStore(index=index, embedding=embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# Local LLM (Flan-T5)
qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-base", tokenizer="google/flan-t5-base")

# Streamlit UI
st.title("📘 Local RAG Chatbot (Flan-T5 + Pinecone)")

if "messages" not in st.session_state:
    st.session_state.messages = [SystemMessage(content="You are a helpful assistant using local Flan-T5 with Pinecone RAG.")]

# Display history
for msg in st.session_state.messages:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)

# Chat input
prompt = st.chat_input("Ask me something about your PDFs...")
if prompt:
    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    # Retrieve context
    docs = retriever.invoke(prompt)
    context = "\n".join([d.page_content for d in docs])

    system_prompt = f"Context: {context}\n\nQuestion: {prompt}\nAnswer concisely:"
    response = qa_pipeline(system_prompt, max_length=256, do_sample=True, temperature=0.3)[0]["generated_text"]

    # Show assistant response
    st.session_state.messages.append(AIMessage(content=response))
    with st.chat_message("assistant"):
        st.markdown(response)
