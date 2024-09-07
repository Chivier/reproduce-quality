import json
import os
import numpy as np
import tiktoken
from utils import *

# 路径配置
gQasperPath = './datasets/qasper/'
gTrainPath = os.path.join(gQasperPath, 'qasper-train-v0.3.json')
gParsedDataPath = './parsed_data/qasper/'

# 读取 JSON 文件
def ReadJson(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

# 解析单个条目
def ParseEntry(entry_id, entry):
    # Meta data
    article = "# " + entry["title"]
    article += "\n---\n"
    article += "id: " + entry_id + "\n"
    article += "title: " + entry["title"] + "\n"
    article += "abstract: " + entry["abstract"] + "\n"
    article += "---\n"
    article += "\n".join([p for section in entry["full_text"] for p in section["paragraphs"] if p])

    questions = []
    for qa in entry["qas"]:
        new_question = "<QUESTION>"
        new_question += qa["question"]
        new_question += "</QUESTION>\n"
        new_question += "<OPTIONS>\n"
        option_id = 1
        for answer in qa["answers"]:
            if answer["answer"]["free_form_answer"]:
                option_text = answer["answer"]["free_form_answer"]
            elif answer["answer"]["extractive_spans"]:
                option_text = ", ".join(answer["answer"]["extractive_spans"])
            elif answer["answer"]["yes_no"] is not None:
                option_text = "Yes" if answer["answer"]["yes_no"] else "No"
            else:
                option_text = "Unanswerable"
            new_question += str(option_id) + ". " + option_text + "\n"
            option_id += 1
        new_question += "</OPTIONS>\n"
        new_answer = qa["answers"][0]["answer"]["free_form_answer"] if qa["answers"] else "Unanswerable"
        questions.append((new_question, new_answer, qa["question_id"]))

    return (article, questions)

# 解析数据集
def ParseData():
    data = ReadJson(gTrainPath)
    os.makedirs(gParsedDataPath, exist_ok=True)
    for idx, (entry_id, entry) in enumerate(data.items()):
        article, questions = ParseEntry(entry_id, entry)
        article_file = open(gParsedDataPath + 'article_' + str(idx) + '.txt', 'w', encoding='utf-8')
        article_file.write(article)
        article_file.close()
        question_file = open(gParsedDataPath + 'question_' + str(idx) + '.jsonl', 'w', encoding='utf-8')
        for qa in questions:
            question, answer, question_id = qa
            qa_json = {}
            qa_json['question'] = question
            qa_json['answer'] = answer
            qa_json['question_id'] = question_id
            question_file.write(json.dumps(qa_json, ensure_ascii=False))
            question_file.write('\n')
        question_file.close()
        print('Parsed', idx, 'entries')

# 切分数据
def CutData():
    file_count = len([f for f in os.listdir(gParsedDataPath) if f.startswith('article_') and f.endswith('.txt')])
    cutting_length = 2000
    overlapping = 400
    encoding = tiktoken.get_encoding("cl100k_base")
    embedding_converter = TextEmbedding()

    for i in range(file_count):
        # 读取文章文件
        article_file = open(gParsedDataPath + 'article_' + str(i) + '.txt', 'r', encoding='utf-8')
        article = article_file.read()
        
        # 切分文章为块
        article_chunk_filename = gParsedDataPath + 'article_' + str(i) + '_chunks.txt'
        vector_chunk_filename = gParsedDataPath + 'article_' + str(i) + '_chunks.npy'
        text_chunks = []
        vector_chunks = []
        chunk_position = 0
        
        article_tokenized = encoding.encode(article)
        article_chunk_file = open(article_chunk_filename, 'w', encoding='utf-8')
        
        while chunk_position < len(article_tokenized):
            # 确定块的边界
            chunk_end = min(chunk_position + cutting_length, len(article_tokenized))
            chunk = article_tokenized[chunk_position:chunk_end]
            chunk_position += cutting_length - overlapping
            text_chunk = encoding.decode(chunk)
            text_chunks.append(text_chunk)
            article_chunk_file.write(repr(text_chunk))
            article_chunk_file.write('\n')
        
        # 将文本块转换为向量块
        vector_chunk = embedding_converter.convert_to_embedding(text_chunks)
        np.save(vector_chunk_filename, vector_chunk)
        
        # 关闭文件
        article_file.close()
        article_chunk_file.close()
        
        print('Cut', i, 'entries')
    
if "__main__" == __name__:
    ParseData()
    CutData()
    print('Done')
