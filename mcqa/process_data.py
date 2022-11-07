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

dirpath = './RACE/dev/middle'
files = os.listdir(dirpath)

all_data = {}
for fpath in tqdm(files):
    with open(dirpath + '/' + fpath) as f:
        ori_data = eval(f.read()) #有答案节点
    answers = ori_data['answers']
    options = ori_data['options']
    questions = ori_data['questions']
    for item,q in zip(options,questions):
        if 'buildings' in item and 'paper' in item:
            print(q)
            ipdb.set_trace()

    article = ori_data['article']
    aid = ori_data['id']
    
    new_answers = []
    for ans in answers:
        if ans == 'A':
            new_answers.append(0)
        elif ans == 'B':
            new_answers.append(1)
        elif ans == 'C':
            new_answers.append(2)
        elif ans == 'D':
            new_answers.append(3)
        else:
            ipdb.set_trace()
    
    all_data[aid] = {}
    all_data[aid]['items'] = []
    all_data[aid]['labels'] = new_answers
    for q,ops in zip(questions,options):
        curq_item = []
        for op in ops:
            curq_item.append((article,q,op))
        all_data[aid]['items'].append(curq_item)
        
    
print(len(all_data))
fp = open('data/test_high','w')
fp.write(json.dumps(all_data))
fp.close()

    
        

    

     


    