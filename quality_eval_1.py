from utils import *
from datetime import datetime
import random
# Zero shot evaluation

import torch
from raptor import BaseSummarizationModel, BaseQAModel, BaseEmbeddingModel, RetrievalAugmentationConfig
from transformers import AutoTokenizer, pipeline

question_file_count = 300
question_position = "./parsed_data/quality_v1.0.1"

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
    
    # log_file : "eval_1_{datetime}.log"
    log_filename = f"eval_1_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
    
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
                
                max_retries = 10
                temperature = 0
                # while not get_answer:
                #     answer = get_ollama_response(question, temperature=temperature)
                #     formated_answer = format_answer(answer)
                #     if formated_answer == -1:
                #         print("Invalid answer")
                #         print(f"Answer: {answer}")
                #         max_retries -= 1
                #         if max_retries == 0:
                #             print("Max retries reached")
                #             # use random answer
                #             # count options between <OPTIONS></OPPTIONS> in quesiton
                #             options = question.split("<OPTIONS>")[1].split("</OPTIONS>")[0].split("\n")
                #             options_count = len(options)
                #             formated_answer = random.randint(0, options_count-1)
                #             get_answer = True
                #             question_count += 1
                #             if formated_answer == correct_answer:
                #                 correct_count += 1
                #         temperature += 0.1
                #         continue
                #     else:
                #         get_answer = True
                #         question_count += 1
                #         if formated_answer == correct_answer:
                #             correct_count += 1
                #         log_file = open(log_filename, "a")
                #         log_file.write(f"Eval_answer: {answer}, Correct_answer: {correct_answer}\n")
                #         log_file.close()
                print(question)
                exit(0)
                        
    print(f"Correct count: {correct_count}/{question_count}")
    print(f"Score: {correct_count/question_count}")
                
if "__main__" == __name__:
    qa_test()
