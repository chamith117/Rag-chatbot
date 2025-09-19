import os
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone

# Load env vars
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
HF_EMBEDDING_MODEL = os.getenv("HF_EMBEDDING_MODEL")

# Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

# Embeddings
embeddings = HuggingFaceEmbeddings(model_name=HF_EMBEDDING_MODEL)

# Vector store retriever
vector_store = PineconeVectorStore(index=index, embedding=embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 4})

query = "What is in the PDF?"
docs = retriever.invoke(query)

print("🔎 Retrieved results:")
for d in docs:
    print("-", d.page_content[:200], "...")
