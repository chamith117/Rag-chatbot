**Create ".env" file **

```
# ---- Pinecone ----
PINECONE_API_KEY=add your key
PINECONE_ENVIRONMENT=us-east-1
PINECONE_INDEX_NAME=add your index name
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1
PINECONE_NAMESPACE=default

# ---- Hugging Face ----
HUGGINGFACEHUB_API_TOKEN=add your token
HF_MODEL_ID=mistralai/Mistral-7B-Instruct-v0.2

# ---- Embeddings ----
HF_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# ---- Retrieval ----
K_RETRIEVAL=4

# ---- Local Models ----
LOCAL_LLM_MODEL=google/flan-t5-base
LOCAL_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

```


1. Create a virtual environment

```
python -m venv venv
```

2. Activate the virtual environment

```
venv\Scripts\Activate
(or on Mac): source venv/bin/activate
```
3. Install libraries

```
pip install -r requirements.txt
```

4. Create accounts

- Create a free account on Pinecone: https://www.pinecone.io/
- Create an API key for huggingface: https://huggingface.co/

5. Add API keys to .env file

- Rename .env.example to .env
- Add the API keys for Pinecone and OpenAI to the .env file

<h3>Executing the scripts</h3>

1. Open a terminal in VS Code

2. Execute the following command:

```

python ingestion.py
python retrieval.py
streamlit run chatbot_rag.py
```
<h3>For Local-Model</h3>

1. Open a terminal in VS Code

2. Execute the following command:

```

python ingestion_flan.py
python retrieval_flan.py
streamlit run chatbot_flan.py
```
