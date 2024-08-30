from utils import *
from datetime import datetime
import random
# Zero shot evaluation

import torch
from raptor import BaseSummarizationModel, BaseQAModel, BaseEmbeddingModel, RetrievalAugmentationConfig, RetrievalAugmentation
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM
import logging
import os
import time


class AlibabaEmbedding(BaseEmbeddingModel):
    def __init__(self):
        self.model = TextEmbedding()
    
    def create_embedding(self, text):
        return self.model.convert_to_embedding(text)


class Llama3_1_70b(BaseQAModel):
    def __init__(self, model="meta-llama/Meta-Llama-3.1:70B"):
        pass
        
    def answer_question(self, context, question, max_tokens=150):
        try:
            # Try to answer the question and spit out an answer
            input_text = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
            answer = get_ollama_response(input_text)
            return answer
        except Exception as e:
            # For some error that may happen
            logging.error(f"Error generating answer: {e}")
            return ""
        
question_file_count = 30
# question_position = "./parsed_data/v1.0.1"

question_position = "./10percentquestions"


common_heading = "You are a helpful assistant. You can help me by answering my questions.\n"
common_prompt = "Give me the option that can answer this problem the best. Only give me the id of the best OPTION. If you cannot find the answer, give the option that you think is the best. ONLY RETURN THE ID!!!\n"

def format_answer(answer):
    answer = answer.strip().lower()
    # answer is a number
    if answer.isdigit():
        return int(answer)
    # if answer is a sentence, and there is only 1 number in the sentence
    if len([int(s) for s in answer.split() if s.isdigit()]) == 1:
        return int([int(s) for s in answer.split() if s.isdigit()][0])
    return -1
    
# Load questions
def qa_test():
    question_count = 0
    correct_count = 0
    indexing_time = 0
    
    log_file : "raptor_eval_1_{datetime}.log"
    log_filename = f"raptor_eval_1_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
    Llama3170b = Llama3_1_70b()
    embedder = AlibabaEmbedding()
    custom_config = RetrievalAugmentationConfig(tr_top_k=20,tb_max_tokens=2000,qa_model=Llama3170b, embedding_model=embedder)
    RA = RetrievalAugmentation(config=custom_config)
    with open("/home/cluster/Projects/quality_benchmark/merged_txt/merged_txt.txt", "r") as file:
        text = file.read()
        start_time = time.time()
        RA.add_documents(text)
        end_time = time.time()

    indexing_time += (end_time-start_time)
    log_file = open(log_filename, "a")
    log_file.write(f"Index Building Time: {indexing_time}\n")
    log_file.close()

    
    for file_id in range(question_file_count):
        print(f"File {file_id}")
        file_path = f"{question_position}/question_{file_id}.jsonl"
        with open (file_path, "r") as f:
            for line in f:
                get_answer = False
                data = json.loads(line)
                question = data["question"]
                correct_answer = data["answer"]
                question = common_heading + question + common_prompt
                # RAPTOR Implementation starts here:
                
                # Config Retrieval Augmentation
                
                max_retries = 10
                temperature = 0
                while not get_answer:
                    # RAPTOR Answer
                    answer = RA.answer_question(question=question)
                    
                    #answer = get_ollama_response(question, temperature=temperature)
                    formated_answer = format_answer(answer)
                    if formated_answer == -1:
                        print("Invalid answer")
                        print(f"Answer: {answer}")
                        max_retries -= 1
                        if max_retries == 0:
                            print("Max retries reached")
                            # use random answer
                            # count options between <OPTIONS></OPPTIONS> in quesiton
                            options = question.split("<OPTIONS>")[1].split("</OPTIONS>")[0].split("\n")
                            options_count = len(options)
                            formated_answer = random.randint(0, options_count-1)
                            get_answer = True
                            question_count += 1
                            if formated_answer == correct_answer:
                                correct_count += 1
                        temperature += 0.1
                        continue
                    else:
                        get_answer = True
                        question_count += 1
                        if formated_answer == correct_answer:
                            correct_count += 1
                        log_file = open(log_filename, "a")
                        log_file.write(f"Eval_answer: {answer}, Correct_answer: {correct_answer}\n")
                        log_file.close()
                        
    print(f"Correct count: {correct_count}/{question_count}")
    print(f"Score: {correct_count/question_count}")
                
if "__main__" == __name__:
    qa_test()