"""Extract pre-computed feature vectors from a PyTorch BERT model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
#from bert import pytorch_pretrained_bert
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import argparse
import collections
import logging
import json
import re
import ipdb
from collections import Iterable
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
import os
from pytorch_pretrained_bert import BertTokenizer
from pytorch_pretrained_bert import BertModel
from transformers import AlbertConfig, AlbertModel,AlbertTokenizer
from transformers import ElectraForPreTraining, ElectraTokenizerFast
from transformers import AutoModel, AutoTokenizer





class InputExample(object):

    def __init__(self, unique_id, text_a, text_b):
        self.unique_id = unique_id
        self.text_a = text_a
        self.text_b = text_b


class InputFeatures(object):


    def __init__(self, unique_id, tokens, input_ids, input_mask, input_type_ids):
        self.unique_id = unique_id
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_type_ids = input_type_ids


def convert_examples_to_features(examples, seq_length, tokenizer):
   
    
    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
 
        if tokens_b:

            _truncate_seq_pair(tokens_a, tokens_b, seq_length - 3)
        else:

            if len(tokens_a) > seq_length - 2:
                tokens_a = tokens_a[0:(seq_length - 2)]

        tokens = []
        input_type_ids = []
        tokens.append("[CLS]")
        input_type_ids.append(0)
        for token in tokens_a:
            
     
            tokens.append(token)
            input_type_ids.append(0)
        tokens.append("[SEP]")
        input_type_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                input_type_ids.append(1)
            tokens.append("[SEP]")
            input_type_ids.append(1)
       
        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < seq_length:
            input_ids.append(0)
            input_mask.append(0)
            input_type_ids.append(0)

        assert len(input_ids) == seq_length
        assert len(input_mask) == seq_length
        assert len(input_type_ids) == seq_length
        features.append(
            InputFeatures(
                unique_id=example.unique_id,
                tokens=tokens,
                input_ids=input_ids,
                input_mask=input_mask,
                input_type_ids=input_type_ids))
    return features,tokens


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
 

    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def read_examples(input_list):
    examples = []
    unique_id = 0
    for ins in input_list:
            ins = ins.strip()
            text_a = None
            text_b = None
            m = re.match(r"^(.*) \|\|\| (.*)$", ins)
            if m is None:
                text_a = ins
            else:
                text_a = m.group(1)
                text_b = m.group(2)
            examples.append(
                InputExample(unique_id=unique_id, text_a=text_a, text_b=text_b))
            unique_id += 1
    return examples



class Embedding(nn.Module):

    def __init__(self, word_vec_mat, max_length=100, word_embedding_dim=768, dpos_embedding_dim=50, dmask_embedding_dim=50):
        nn.Module.__init__(self)
        self.max_length = max_length
        self.word_embedding_dim = word_embedding_dim
        self.dpos_embedding_dim = dpos_embedding_dim
        self.dmask_embedding_dim = dmask_embedding_dim
       
        self.bert_token = AutoTokenizer.from_pretrained('./bert.base')
        self.bert =  AutoModel.from_pretrained('./bert.base')
        


    def forward(self, inputs,sent,gate):      
      
        examples = read_examples(sent) 
        features,tokens = convert_examples_to_features(examples=examples, seq_length=self.max_length, tokenizer=self.bert_token)
      
        unique_id_to_feature = {}
        for feature in features:
            unique_id_to_feature[feature.unique_id] = feature 
      
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long).cuda()
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long).cuda()
        all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long).cuda()
      
        attention_mask = torch.ones_like(all_input_ids)
        attention_mask = torch.where(all_input_ids==0,all_input_ids,attention_mask)
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        attention_mask = attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        attention_mask = (1.0 - attention_mask) * -10000.0        
        
                           
        embedding_output = self.bert.embeddings(all_input_ids , token_type_ids=None)
        
        head_mask = [None]*12
        gate = gate.unsqueeze(1).expand(gate.size(0),12,gate.size(1),gate.size(2))
    
        all_encoder_layers = self.bert.encoder(embedding_output,attention_mask,head_mask,gate)
        bert = all_encoder_layers[-1]

  
        return bert,all_input_mask


