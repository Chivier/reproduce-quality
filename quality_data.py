# Tasks in this script:
# 1. Load the data from quality dataset
# download: `git clone https://github.com/nyu-mll/quality`
# 2. Load and parse dataset
# Read Json
import json
import os
from utils import *
import numpy as np
import re
import tiktoken

gDataPath = './quality/data/v1.0.1/'
gTrainPath = './quality/data/v1.0.1/QuALITY.v1.0.1.htmlstripped.train'
gParsedDataPath = './parsed_data/v1.0.1/'

# Read Json
def ReadLine(filename, line_id):
    with open(filename) as f:
        for i, line in enumerate(f):
            if i == line_id:
                data = json.loads(line)
                return data
    return None


def ParseLine(json_data):
    # new_json['article_id'] = json_data['article_id']
    # new_json['set_unique_id'] = json_data['set_unique_id']
    # new_json['batch_num'] = json_data['batch_num']
    # new_json['writer_id'] = json_data['writer_id']
    # new_json['source'] = json_data['source']
    # new_json['title'] = json_data['title']
    # new_json['year'] = json_data['year']
    # new_json['author'] = json_data['author']
    # new_json['topic'] = json_data['topic']
    # new_json['article'] = json_data['article']

    # Meta data
    article = "# " + json_data["title"]
    article += "\n---\n"
    article += "article_id: " + json_data["article_id"] + "\n"
    article += "set_unique_id: " + json_data["set_unique_id"] + "\n"
    article += "batch_num: " + json_data["batch_num"] + "\n"
    article += "writer_id: " + json_data["writer_id"] + "\n"
    article += "source: " + json_data["source"] + "\n"
    article += "year: " + str(json_data["year"]) + "\n"
    article += "author: " + json_data["author"] + "\n"
    article += "topic: " + json_data["topic"] + "\n"
    article += "---\n"
    article += json_data["article"]

    questions = []
    for item in json_data["questions"]:
        new_question = "<QUESTION>"
        new_question += item["question"]
        new_question += "</QUESTION>\n"
        new_question += "<OPTIONS>\n"
        option_id = 1
        for option in item["options"]:
            new_question += str(option_id) + ". " + option + "\n"
            option_id += 1
        new_question += "</OPTIONS>\n"
        new_answer = item["gold_label"]
        questions.append((new_question, new_answer))

    return (article, questions)

# Parse data
def ParseData():
    total_line = 300
    for i in range(total_line):
        line = ReadLine(gTrainPath, i)
        article, questions = ParseLine(line)
        article_file = open(gParsedDataPath + 'article_' + str(i) + '.txt', 'w')
        article_file.write(article)
        article_file.close()
        question_file = open(gParsedDataPath + 'question_' + str(i) + '.jsonl', 'w')
        for qa in questions:
            question, answer = qa
            qa_json = {}
            qa_json['question'] = question
            qa_json['answer'] = answer
            question_file.write(json.dumps(qa_json))
            question_file.write('\n')
        question_file.close()
        print('Parsed', i, 'lines')

# Cut the data
def CutData():
    file_count = 300
    cutting_length = 2000
    overlapping = 400
    encoding = tiktoken.get_encoding("cl100k_base")
    embedding_converter = TextEmbedding()

    for i in range(file_count):
        # Read the article file
        article_file = open(gParsedDataPath + 'article_' + str(i) + '.txt', 'r')
        article = article_file.read()
        
        # Cut the article into chunks
        article_chunk_filename = gParsedDataPath + 'article_' + str(i) + '_chunks.txt'
        vector_chunk_filename = gParsedDataPath + 'article_' + str(i) + '_chunks.npy'
        text_chunks = []
        vector_chunks = []
        chunk_begin = 0
        
        article_tokenized = encoding.encode(article)
        article_chunk_file = open(article_chunk_filename, 'w')
        
        chunk_position = 0
        while chunk_position < len(article_tokenized):
            # Determine the chunk boundaries
            chunk_end = min(chunk_position + cutting_length, len(article_tokenized))
            chunk = article_tokenized[chunk_position:chunk_end]
            chunk_position += cutting_length - overlapping
            text_chunk = encoding.decode(chunk)
            text_chunks.append(text_chunk)
            article_chunk_file.write(repr(text_chunk))
            article_chunk_file.write('\n')
        
        # Convert text chunks to vector chunks
        vector_chunk = embedding_converter.convert_to_embedding(text_chunks)
        print(vector_chunk)
        print(vector_chunk.shape)
        np.save(vector_chunk_filename, vector_chunk)
        
        # Close the files
        article_file.close()
        article_chunk_file.close()
        
        print('Cut', i, 'lines')
    
if "__main__" == __name__:
    ParseData()
    CutData()
    print('Done')
