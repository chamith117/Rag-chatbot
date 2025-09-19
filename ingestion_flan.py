import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD")
PINECONE_REGION = os.getenv("PINECONE_REGION")
HF_EMBEDDING_MODEL = os.getenv("HF_EMBEDDING_MODEL")

# --- Pinecone init ---
pc = Pinecone(api_key=PINECONE_API_KEY)

# Create index if it does not exist
if PINECONE_INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=384,  # embedding size of all-MiniLM-L6-v2
        metric="cosine",
        spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
    )

index = pc.Index(PINECONE_INDEX_NAME)

# --- Load PDFs ---
loader = DirectoryLoader("./documents", glob="*.pdf", loader_cls=PyPDFLoader)
docs = loader.load()

# --- Split into chunks ---
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)

# --- Embeddings ---
embeddings = HuggingFaceEmbeddings(model_name=HF_EMBEDDING_MODEL)

# --- Push to Pinecone ---
vector_store = PineconeVectorStore.from_documents(chunks, embeddings, index_name=PINECONE_INDEX_NAME)

print(f"✅ Ingested {len(chunks)} chunks into Pinecone index '{PINECONE_INDEX_NAME}'")
