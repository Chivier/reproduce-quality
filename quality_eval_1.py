from utils import *
from datetime import datetime
import random
from nerif.nerif_core import *
# Zero shot evaluation

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
        print(f"File {file_id}")
        file_path = f"{question_position}/question_{file_id}.jsonl"
        with open (file_path, "r") as f:
            for line in f:
                data = json.loads(line)
                original_question = data["question"]
                correct_answer = data["answer"]
                # find question in <QUESTION></QUESTION>
                question = data["question"].split("<QUESTION>")[1].split("</QUESTION>")[0]
                print(question)
                # find options in <OPTIONS></OPTIONS>
                options = question.split("<OPTIONS>")[1].split("</OPTIONS>")[0].split("\n")
                # parse options by newline  
                options = [option.strip() for option in options]
                print(options)
                
                selector = NerifMatchString(choices=options, model="gpt-4o-mini")
                answer = selector.match(question)
                print(answer)
                print(correct_answer)
                exit(0)
                        
    print(f"Correct count: {correct_count}/{question_count}")
    print(f"Score: {correct_count/question_count}")
                
if "__main__" == __name__:
    qa_test()
