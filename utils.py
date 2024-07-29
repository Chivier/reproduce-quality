from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

import requests
import json

class TextEmbedding:
    def __init__(self):
        self.model = SentenceTransformer('Alibaba-NLP/gte-large-en-v1.5', trust_remote_code=True)
    
    def convert_to_embedding(self, sentences):
        embeddings = self.model.encode(sentences)
        return embeddings
    
    def compute_similarity(self, sentence1, sentence2):
        embeddings = self.model.encode([sentence1, sentence2])
        print(cos_sim(embeddings[0], embeddings[1]))

# Test    
# * sentences = ['That is a happy person', 'That is a very happy person']
# * model = SentenceTransformer('Alibaba-NLP/gte-large-en-v1.5', trust_remote_code=True)
# * embeddings = model.encode(sentences)
# * print(cos_sim(embeddings[0], embeddings[1]))


def get_ollama_response(prompt, model="llama3.1:70b", temperature=0, stream=False):
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "options": {
            "temperature": temperature
        },
        "stream": stream
    }

    headers = {
        'Content-Type': 'application/json'
    }

    response = requests.post(url, headers=headers, data=json.dumps(payload))
    
    if response.status_code == 200:
        response_data = response.json()
        return response_data.get("response", "")
    else:
        raise Exception(f"Request failed with status code {response.status_code}")

# Example usage
# * model = "llama3.1:70b"
# * prompt = "Why is the sky blue?"
# * 
# * response_text = get_ollama_response(prompt, model)
# * print(response_text)