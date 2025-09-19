import os
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
HF_EMBEDDING_MODEL = os.getenv("HF_EMBEDDING_MODEL")

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

embeddings = HuggingFaceEmbeddings(model_name=HF_EMBEDDING_MODEL)
vector_store = PineconeVectorStore(index=index, embedding=embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

query = "What is Sri Lanka?"
docs = retriever.invoke(query)

print("\n🔎 Retrieved context:")
for d in docs:
    print("-", d.page_content[:200], "...")
