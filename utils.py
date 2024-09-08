from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

import requests
import json
import uuid
import numpy as np

def merge_spans(intervals):
    """合并有交集的区间"""
    intervals.sort(key=lambda x: x[0])
    merged = []
    for interval in intervals:
        if not merged or merged[-1][1] < interval[0]:
            merged.append(interval)
        else:
            merged[-1][1] = max(merged[-1][1], interval[1])
    return merged

def merge_labeled_spans(spacy_list, text, return_positions=False):
    """按照标签合并有交集的区间"""
    merged_list = []
    for s, e, label in spacy_list:
        if merged_list and merged_list[-1][1] == s and merged_list[-1][2] == label:
            merged_list[-1][1] = e
        else:
            merged_list.append([s, e, label])
    if return_positions:
        return merged_list
    merged_list = {text[s:e] for s, e, _ in merged_list}
    return merged_list

# def encrypte_noun_text(text, spacy_model, return_ents=True):
#     """标签匿名化函数，输出标签匿名后的文本，可选是否返回对应的实体列表"""
#     doc = spacy_model(text)
#     spacy_list = [(token.idx, token.idx + len(token), token.pos_) for token in doc if token.pos_ in ['NOUN', 'PROPN']]
#     spacy_list = merge_labeled_spans(spacy_list, text, return_positions=True)
#     positions = np.array([ent[:2] for ent in spacy_list])
#     labels = [ent[2] for ent in spacy_list]
#     ner_list = {}
#     for i in range(len(spacy_list)):
#         s, e = positions[i]
#         original_text = text[s:e]
#         unique_id = str(uuid.uuid4())
#         ner_list[unique_id] = original_text
#         text = text[:s] + f'<{unique_id}>' + text[e:]
#         positions[i:,:] = positions[i:,:] + len(unique_id) + 2 - (e - s)
#     if return_ents:
#         return text, ner_list
#     else:
#         return text

def encrypte_noun_text(text, spacy_model, uuid_map):
    """标签匿名化函数，输出标签匿名后的文本，可选是否返回对应的实体列表"""
    doc = spacy_model(text)
    spacy_list = [(token.idx, token.idx + len(token), token.pos_) for token in doc if token.pos_ in ['NOUN', 'PROPN']]
    spacy_list = merge_labeled_spans(spacy_list, text, return_positions=True)
    positions = np.array([ent[:2] for ent in spacy_list])
    for i in range(len(spacy_list)):
        s, e = positions[i]
        original_text = text[s:e]
        if original_text in uuid_map.values():
            unique_id = next(key for key, value in uuid_map.items() if value == original_text)
        else:
            unique_id = str(uuid.uuid4())
            uuid_map[unique_id] = original_text
        text = text[:s] + f'<{unique_id}>' + text[e:]
        positions[i:,:] = positions[i:,:] + len(unique_id) + 2 - (e - s)
    return text

# def my_get_labelled_text(text, spacy_model, return_ents=True):
#     """标签匿名化函数，输出标签匿名后的文本，可选是否返回对应的实体列表"""
#     doc = spacy_model(text)
#     spacy_list = [(token.idx, token.idx + len(token), token.pos_) for token in doc if token.pos_ in ['NOUN', 'PROPN']]
#     spacy_list = merge_labeled_spans(spacy_list, text, return_positions=True)
#     positions = np.array([ent[:2] for ent in spacy_list])
#     labels = [ent[2] for ent in spacy_list]
#     ner_list = {}
#     for i in range(len(spacy_list)):
#         s, e = positions[i]
#         original_text = text[s:e]
#         unique_id = str(uuid.uuid4())
#         ner_list[unique_id] = original_text
#         text = text[:s] + f'<{unique_id}>' + text[e:]
#         positions[i:,:] = positions[i:,:] + len(unique_id) + 2 - (e - s)
#     if return_ents:
#         return text, ner_list
#     else:
#         return text

class TextEmbedding:
    def __init__(self):
        #self.model = SentenceTransformer('Alibaba-NLP/gte-large-en-v1.5')
        #self.model = SentenceTransformer('Alibaba-NLP/gte-large-en-v1.5', trust_remote_code=True)
        self.model = SentenceTransformer('nvidia/NV-Embed-v2', trust_remote_code=True)
        self.model.max_seq_length = 4096
        self.model.tokenizer.padding_side="right"

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
