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

max_tag_num = 0
with open('data/dtree_train_high.json') as f:
    all_dtree = json.loads(f.read()) 
with open('data/nelocation_train_high.json') as f:
    all_ne_location = json.loads(f.read()) 
with open('data/train_high.json') as f:
all_dtag = {}    
for aid in tqdm(all_ne_location.keys()):
    if aid not in all_dtree.keys(): continue
    #cur_dtree = all_dtree[aid]
    cur_ne_location = all_ne_location[aid]
    cur_art = all_data[aid]['items'][0][0][0]
    sentences = cur_art.lower().split('.')    
    all_dtag[aid] = []
    cur_questions = all_data[aid]['items']
    for ne4q, question in zip(cur_ne_location, cur_questions):
        cur_q_ne = []
        for  ne4s, sent in zip(ne4q, sentences):
            sent_words = nlp.word_tokenize(sent+'.')
            cur_sent_dtag = []
            for i in range(len(sent_words)):
                #ipdb.set_trace()
                if len(ne4s)==0:
                    cur_sent_dtag.append('100')
                else:
                    cur_w_pos = 100
                    for ne_w in ne4s:
                        if abs(i-ne_w) < abs(cur_w_pos):
                            cur_w_pos = i-ne_w
                    cur_sent_dtag.append(cur_w_pos)
    
            if len(ne4s) != 0:
                for loc in ne4s:
                    cur_sent_dtag[loc-1] = 'self'
                for dtag in dtree4s:
                    
                    if dtag[1] in ne4s:
                        cur_sent_dtag[dtag[2]-1] = dtag[0]
                        if 'other' in cur_sent_dtag[dtag[2]-1]: cur_sent_dtag[dtag[2]-1].remove('other')
                    if dtag[2] in ne4s:
                        cur_sent_dtag[dtag[1]-1] = dtag[0]
                        if 'other' in cur_sent_dtag[dtag[1]-1]: cur_sent_dtag[dtag[1]-1].remove('other')

            cur_q_ne.append(cur_sent_dtag)
        all_dtag[aid].append(cur_q_ne)
print(len(all_dtag))
with open('data/position_train_high.json','w') as f:
    f.write(json.dumps(all_dtag))
nlp.close()               
            
    
    
    
    
#with open('data/dtag_dev_middle.json','w') as f:
#    f.write(json.dumps(all_dtag))



        
    
        

    

     


    