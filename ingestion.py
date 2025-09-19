import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

# ENV variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
HF_EMBEDDING_MODEL = os.getenv("HF_EMBEDDING_MODEL")

# Load PDFs
docs_folder = "documents"
documents = []
for file in os.listdir(docs_folder):
    if file.endswith(".pdf"):
        loader = PyPDFLoader(os.path.join(docs_folder, file))
        documents.extend(loader.load())

# Split text
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(documents)

# Embeddings
embeddings = HuggingFaceEmbeddings(model_name=HF_EMBEDDING_MODEL)

# Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

if PINECONE_INDEX_NAME not in [i["name"] for i in pc.list_indexes()]:
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=384,  # dimension of MiniLM embeddings
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(PINECONE_INDEX_NAME)

# Store in Pinecone
vectorstore = PineconeVectorStore(index=index, embedding=embeddings)
vectorstore.add_documents(splits)

print(f"✅ Successfully ingested {len(splits)} chunks into Pinecone index '{PINECONE_INDEX_NAME}'")
