# coding: utf-8
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'False'
from typing import Dict
import numpy as np
import spacy
from utils import my_get_labelled_text



# specify langauge
lang = 'en'


def hide_text(raw_input, spacy_model):

    return my_get_labelled_text(raw_input, spacy_model, return_ents=True)

def restore_text(text, ner_list):
    """根据ner_list将匿名化文本还原"""
    for unique_id, original_text in ner_list.items():
        text = text.replace(f'<{unique_id}>', original_text)
    return text



if __name__ == '__main__':
    # load models
    print('loading model...')

    spacy_model = spacy.load(f'{lang}_core_web_trf')
    while True:
        # input text
        raw_input = input('\033[1;31minput:\033[0m ')
        if raw_input == 'q':
            print('quit')
            break
        # hide
        hidden_text, ner_list = hide_text(raw_input, spacy_model)
        recovered_text = restore_text(hidden_text, ner_list)
        print('\033[1;31mhidden text:\033[0m ', hidden_text)
        print('\033[1;31moriginal text:\033[0m ', recovered_text)
        print('\033[1;31mYes!\033[0m ') if recovered_text == raw_input else None
