import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

client = InferenceClient(
    model=os.environ.get("HF_MODEL_ID", "mistralai/Mistral-7B-Instruct-v0.2"),
    token=os.environ.get("HUGGINGFACEHUB_API_TOKEN"),
)

response = client.chat.completions.create(
    model=os.environ.get("HF_MODEL_ID", "mistralai/Mistral-7B-Instruct-v0.2"),
    messages=[{"role": "user", "content": "Hello! Can you explain what RAG is in 2 sentences?"}],
    max_tokens=200,
)

print(response.choices[0].message["content"])
