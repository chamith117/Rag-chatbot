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
