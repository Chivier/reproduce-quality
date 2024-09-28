import json
from datetime import datetime
import random
from nerif.nerif_core import *
import numpy as np

# Vector DB
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from qdrant_client.models import PointStruct

import platform

if platform.system() == 'Linux':
    from utils import *
    import torch
    from raptor import BaseSummarizationModel, BaseQAModel, BaseEmbeddingModel, RetrievalAugmentationConfig
    from transformers import AutoTokenizer, pipeline

question_file_count = 300
question_position = "./parsed_data/quality_v1.0.1"

def build_vectordb():
    client = QdrantClient(url="http://localhost:6333")
    vectordb_points = []
    chunk_id = 0
    for file_id in range(question_file_count):
        print(f"File {file_id}")

        article_chunk_text_path = f"{question_position}/article_{file_id}_chunks.txt"
        article_chunk_embedding_path = f"{question_position}/article_{file_id}_chunks.npy"

        article_chunks = []
        with open(article_chunk_text_path, "r") as f:
            for line in f:
                article_chunks.append(line)
        article_embeddings = np.load(article_chunk_embedding_path)
        client.create_collection(
            collection_name=f"quality_rag_article_{file_id}",
            vectors_config=VectorParams(size=4096, distance=Distance.COSINE),
        )
        for i, chunk in enumerate(article_chunks):
            chunk_id += 1
            point = PointStruct(id=chunk_id, vector=article_embeddings[i], payload={"text": chunk})
            vectordb_points.append(point)

    client.upsert(
        collection_name=f"quality_rag",
        points=vectordb_points,
    )
    return client


# Load questions
def qa_test(client):
    question_count = 0
    correct_count = 0
    
    # log_file : "eval_1_{datetime}.log"
    log_filename = f"eval_1_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
    
    for file_id in range(question_file_count):
        file_path = f"{question_position}/question_{file_id}.jsonl"
        with open (file_path, "r") as f:
            for line in f:
                data = json.loads(line)
                original_question = data["question"]
                correct_answer = data["answer"]

                # find question in <QUESTION></QUESTION>
                question = original_question.split("<QUESTION>")[1].split("</QUESTION>")[0]
                # find options in <OPTIONS></OPTIONS>
                options = original_question.split("<OPTIONS>")[1].split("</OPTIONS>")[0].split("\n")
                options = [option.strip() for option in options]
                options = [option for option in options if option]
                
                selector = NerifMatchString(choices=options, model="gpt-4o")
                # export OPENAI_PROXY_URL="https://localhost:11434/"
                # export OPENAI_API_BASE="https://localhost:11434/v1/"
                # selector = NerifMatchString(choices=options, model="ollama/llama3.1-70b-instruct")
                answer = selector.match(question)

                # ( convert question -> embedding )
                # # convert question to embedding
                # question_embedding = xxxx
                # # 
                # search_result = client.query_points(
                #     collection_name="quality_rag",
                #     query=question_embedding,
                #     with_payload=True,
                #     limit=3
                # ).points

                print(search_result)
                # article = 
                # question_withrag = .... # add top k
                # answer_withrag = selector.match(question_withrag)
                
                print(answer)
                print(correct_answer)
                break
                        
    print(f"Correct count: {correct_count}/{question_count}")
    print(f"Score: {correct_count/question_count}")
                
if "__main__" == __name__:
    client = build_vectordb()
    qa_test(client)