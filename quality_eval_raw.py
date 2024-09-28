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

# Load questions
def qa_test():
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
                answer = selector.match(question)

                if answer == correct_answer:
                    correct_count += 1
                question_count += 1
                        
    print(f"Correct count: {correct_count}/{question_count}")
    print(f"Score: {correct_count/question_count}")
                
if "__main__" == __name__:
    qa_test()