# coding: utf-8
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'False'
import json
import re
from utils import encrypte_noun_text



def hide_text(raw_input, spacy_model, uuid_map):
    return encrypte_noun_text(raw_input, spacy_model, uuid_map)

# def restore_text(text, ner_list):
#     """根据ner_list将匿名化文本还原"""
#     for unique_id, original_text in ner_list.items():
#         text = text.replace(f'<{unique_id}>', original_text)
#     return text

def process_file(file_path, spacy_model, uuid_map):
    """处理单个文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    processed_lines = []
    for line in lines:
        doc = spacy_model(line.strip())
        if len(doc) > 500:
            chunks = [doc[i:i + 500].text for i in range(0, len(doc), 500)]
        else:
            chunks = [line.strip()]
        
        encrypted_chunks = []
        for chunk in chunks:
            encrypted_chunk = hide_text(chunk, spacy_model, uuid_map)
            encrypted_chunks.append(encrypted_chunk)
        
        processed_lines.append(' '.join(encrypted_chunks))
    
    # 去除空行
    processed_lines = [line for line in processed_lines if line.strip()]

    return processed_lines

def main_decryption(folder):
    import spacy
    # 加载spaCy模型
    nlp = spacy.load("en_core_web_trf")
    # 设置文件夹路径
    input_folder = folder
    output_folder = folder
    uuid_file = os.path.join(folder, 'encrypt_1.json')

    # Load or initialize UUID mapping
    if os.path.exists(uuid_file):
        with open(uuid_file, 'r', encoding='utf-8') as f:
            uuid_map = json.load(f)
    else:
        uuid_map = {}

    files_to_process = []
    for file_name in os.listdir(input_folder):
        match = re.match(r'article_(\d+)_chunks\.txt', file_name)
        if match:
            file_id = int(match.group(1))
            files_to_process.append((file_id, file_name))
    
    # 按文件ID排序
    files_to_process.sort(key=lambda x: x[0])

    # Process each file in the input folder
    for file_id, file_name in files_to_process:
        print(f'Processing file {file_id}...')
        input_file_path = os.path.join(input_folder, file_name)
        output_file_path = os.path.join(output_folder, f'article_{file_id}_encrypt_1_chunks.txt')
        
        processed_lines = process_file(input_file_path, nlp, uuid_map)
        
        with open(output_file_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(processed_lines))
    
    # Save the UUID mapping
    with open(uuid_file, 'w', encoding='utf-8') as f:
        json.dump(uuid_map, f, ensure_ascii=False, indent=4)
        
def delete_encrypted_files(folder):
    # 获取文件夹中的所有文件
    files_in_folder = os.listdir(folder)
    
    # 遍历文件列表，删除符合条件的文件
    for file_name in files_in_folder:
        if file_name.startswith('article_') and file_name.endswith('_encrypt_1_chunks.txt'):
            file_path = os.path.join(folder, file_name)
            try:
                os.remove(file_path)
                print(f'Deleted: {file_path}')
            except Exception as e:
                print(f'Error deleting {file_path}: {e}')

# 示例使用
folder = './parsed_data/dummy'

main_decryption(folder)
# delete_encrypted_files(folder)
