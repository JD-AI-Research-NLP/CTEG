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
stw = stopwords.words('english')+['!', ',' ,'.' ,'?' ,'-s' ,'-ly' ,'</s> ', 's', '_']
nlp = StanfordCoreNLP(r'stanford-corenlp-full-2018-10-05')
with open('data/train_high.json') as f:
    ori_data = json.loads(f.read()) #有答案节点

all_dependency_result = {}
for aid in tqdm(ori_data.keys()):
    if len(ori_data[aid]['items'])==0: ipdb.set_trace()
    ipdb.set_trace()    
    article = ori_data[aid]['items'][0][0][0] 
    sentences = article.split('.')
    cur_dependency_result = []
    for sent in sentences:
        parse4sent = nlp.dependency_parse(sent + '.')
        cur_dependency_result.append(parse4sent)
    all_dependency_result[aid] = cur_dependency_result
nlp.close()

print(len(all_dependency_result))
with open('data/dtree_train_high.json','w') as f:
    f.write(json.dumps(all_dependency_result))



        
    
        

    

     


    