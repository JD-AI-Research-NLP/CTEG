import json
import re
import collections
import string
import sys
import os
import torch
import ipdb
import numpy as np
from tqdm import tqdm 
from scipy import sparse
from stanfordcorenlp import StanfordCoreNLP 
import nltk
from nltk.corpus import stopwords
nlp = StanfordCoreNLP(r'stanford-corenlp-full-2018-10-05')

with open('data/ne_train_middle.json') as f:
    all_ne = json.loads(f.read()) 
with open('data/train_middle.json') as f:
    all_data = json.loads(f.read())     
all_ne_location = {}
for aid in tqdm(all_ne.keys()):
    all_ne_location[aid] = []
    if len(all_data[aid]['items']) == 0: continue
    cur_art = all_data[aid]['items'][0][0][0]
    sentences = cur_art.lower().split('.')
    cur_ne = all_ne[aid]
  
    for cur_question_ne in cur_ne:
        cur_question_ne_location = []
        for sent in sentences:
            cur_sent_ne_location = []
            sent_words = nlp.word_tokenize(sent+'.')
            for ne in cur_question_ne:
                if ne in sent_words:
                    cur_sent_ne_location.append(sent_words.index(ne) + 1)
            cur_question_ne_location.append(cur_sent_ne_location)
        all_ne_location[aid].append(cur_question_ne_location)
print(len(all_ne_location))                
with open('data/nelocation_train_middle.json','w') as f:
    f.write(json.dumps(all_ne_location))
    
nlp.close()


        
    
        

    

     


    