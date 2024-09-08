# coding: utf-8
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'False'
from typing import Dict
import numpy as np
import spacy
from utils import my_get_labelled_text, encrypte_noun_text



# specify langauge
lang = 'en'


# def hide_text(raw_input, spacy_model):

#     return my_get_labelled_text(raw_input, spacy_model, return_ents=True)

def hide_text(raw_input, spacy_model, uuid_map):
    return encrypte_noun_text(raw_input, spacy_model, uuid_map)

def restore_text(text, ner_list):
    """根据ner_list将匿名化文本还原"""
    for unique_id, original_text in ner_list.items():
        text = text.replace(f'<{unique_id}>', original_text)
    return text



if __name__ == '__main__':
    # load models
    print('loading model...')
    uuid_map = {}
    spacy_model = spacy.load(f'{lang}_core_web_trf')
    while True:
        # input text
        raw_input = input('\033[1;31minput:\033[0m ')
        if raw_input == 'q':
            print('quit')
            break
        # hide
        hidden_text = hide_text(raw_input, spacy_model, uuid_map)
        print('\033[1;31mhidden text:\033[0m ', hidden_text)



