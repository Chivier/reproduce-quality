from datetime import datetime
import random
import json
import os
from utils import *
import argparse
from collections import Counter
import string
import re

# 你可能需要根据具体需求调整这些导入
# from utils import *
# from raptor import BaseSummarizationModel, BaseQAModel, BaseEmbeddingModel, RetrievalAugmentationConfig
# from transformers import AutoTokenizer, pipeline

question_file_count = 888  
question_position = "./parsed_data/qasper"  # Qasper数据集的解析位置
results_path = "./results/no_retrieval/qasper"  # 保存日志文件的路径
predictions_path = "./results/no_retrieval/qasper/predictions.jsonl"  # 保存预测结果的路径

# 确保结果目录存在
os.makedirs(results_path, exist_ok=True)

common_heading = "You are a helpful assistant. You can help me by answering my questions.\n"
common_prompt = "Please provide the best answer to the following question. If you cannot find the answer, respond with the best guess. Only provide the answer without any explanation or analysis.\n"

def format_answer(answer):
    answer = answer.strip().lower()
    return answer

# 载入问题并进行测试
def qa_test():
    question_count = 0
    correct_count = 0
    
    # 日志文件
    log_filename = os.path.join(results_path, f"eval_qasper_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log")
    predictions = []

    for file_id in range(question_file_count):
        print(f"Processing File {file_id}")
        file_path = f"{question_position}/question_{file_id}.jsonl"
        with open(file_path, "r") as f:
            for line in f:
                get_answer = False
                data = json.loads(line)
                question = data["question"]
                correct_answer = data["answer"]
                question_id = data["question_id"]
                question = common_heading + question + common_prompt
                
                max_retries = 10
                temperature = 0
                while not get_answer:
                    answer = get_ollama_response(question, temperature=temperature)
                    formated_answer = format_answer(answer)
                    if not formated_answer:
                        print("Invalid answer")
                        print(f"Answer: {answer}")
                        max_retries -= 1
                        if max_retries == 0:
                            print("Max retries reached")
                            # 使用随机答案或默认答案
                            formated_answer = "No valid answer"
                            get_answer = True
                            question_count += 1
                        temperature += 0.1
                        continue
                    else:
                        get_answer = True
                        question_count += 1
                        if formated_answer == correct_answer:
                            correct_count += 1
                        with open(log_filename, "a") as log_file:
                            log_file.write(f"Eval_answer: {answer}, Correct_answer: {correct_answer}\n")
                        
                predictions.append({
                    "question_id": question_id,
                    "predicted_answer": formated_answer,  # 使用模型的回答
                    "predicted_evidence": []  # 如果有证据，需要在这里填充
                })

    # 保存预测结果
    with open(predictions_path, "w") as f:
        for prediction in predictions:
            f.write(json.dumps(prediction) + "\n")
    
    print(f"Correct count: {correct_count}/{question_count}")
    print(f"Score: {correct_count/question_count}")

if "__main__" == __name__:
    qa_test()
