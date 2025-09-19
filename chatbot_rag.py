import os
import streamlit as st
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from huggingface_hub import InferenceClient

load_dotenv()

st.title("📘 PDF RAG Chatbot")

# Load ENV
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
HF_MODEL_ID = os.getenv("HF_MODEL_ID")
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
HF_EMBEDDING_MODEL = os.getenv("HF_EMBEDDING_MODEL")

# Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

# Embeddings + Vector store
embeddings = HuggingFaceEmbeddings(model_name=HF_EMBEDDING_MODEL)
vector_store = PineconeVectorStore(index=index, embedding=embeddings)

# Hugging Face client
llm_client = InferenceClient(model=HF_MODEL_ID, token=HF_TOKEN)

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = [SystemMessage("You are an assistant for answering questions using context from PDFs.")]

# Render past messages
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(msg.content)

# User input
prompt = st.chat_input("Ask me something from your PDFs...")

if prompt:
    # Show user input
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append(HumanMessage(prompt))

    # Retrieve context from Pinecone
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(prompt)
    context_text = "\n\n".join(d.page_content for d in docs)

    system_prompt = f"""
    You are a helpful assistant. Use the following context from PDFs to answer:

    Context:
    {context_text}

    If the answer isn't in the context, say "I don't know."
    Keep answers concise.
    """

    # Call Hugging Face model
    response = llm_client.chat.completions.create(
        model=HF_MODEL_ID,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        max_tokens=300,
    )

    answer = response.choices[0].message["content"]

    # Show response
    with st.chat_message("assistant"):
        st.markdown(answer)
    st.session_state.messages.append(AIMessage(answer))
