# -*- encoding: utf-8 -*-

import re
from collections import defaultdict
from sklearn.metrics import precision_recall_fscore_support
import os
import sys
import json
import random
import ipdb
import torch
import numpy as np
from tqdm import tqdm
from torch.nn import CrossEntropyLoss, MSELoss
from torch.autograd import Variable
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertTokenizer, BertModel

import torch.optim as optim
max_cqnum = 0
#os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"
#===============================================Data=====================================================
class DataManager:
    def __init__(self):
        
        dp = './bertbase'
        self.tokenizer = BertTokenizer.from_pretrained('./bertbase')
        self.index = 0
        dev_fnm = './data/dev_high.json'
        train_fnm = './data/dev_high.json'
         
        dev_qids, dev_pairs, dev_labels = self.load_dat_new(dev_fnm)
        train_qids, train_pairs, train_labels = self.load_dat_new(train_fnm)

        self.dev_samples = self.trans_samples(dev_qids, dev_pairs, dev_labels)
        self.train_samples = self.trans_samples(train_qids, train_pairs, train_labels)
  
    
        #with open('data/samples_dtag_poses_test_high.json','r') as f:
          #  f.write(json.dumps(self.dev_samples))
            #self.dev_samples_high = json.loads(f.read())
       # with open('data/samples_dtag_poses_train_high.json','r') as f:
          #  f.write(json.dumps(self.train_samples))      
            #self.train_samples_high = json.loads(f.read())
        #with open('data/samples_dtag_poses_test_middle.json','r') as f:
          #  f.write(json.dumps(self.dev_samples))
           # self.dev_samples_middle = json.loads(f.read())
        #with open('data/samples_dtag_poses_train_middle.json','r') as f:
          #  f.write(json.dumps(self.train_samples))      
            #self.train_samples_middle = json.loads(f.read()) 
            
        self.dev_samples = self.dev_samples_middle# + self.dev_samples_middle
        self.train_samples = self.train_samples_high + self.train_samples_middle
        
        self.dev_num = len(self.dev_samples)
        self.dev_idxs = list(range(self.dev_num))  

        self.train_num = len(self.train_samples)
        self.train_idxs = list(range(self.train_num))        
                

        #ipdb.set_trace()
        
        print('all data size:')
        print(len(self.dev_samples),len(self.train_samples))
        #print('true train size = {}; false train size = {}'.format(len(self.true_train_samples),len(self.false_train_samples)))
        #ipdb.set_trace()
    def resample(self):
        #ipdb.set_trace()
        print('sample training data...')
        false_data = random.sample(self.false_train_samples, int(0.045 * len(self.false_train_samples)))
        self.train_samples = self.true_train_samples + false_data
        
        self.dev_num = len(self.dev_samples)
        self.dev_idxs = list(range(self.dev_num))  

        self.train_num = len(self.train_samples)
        self.train_idxs = list(range(self.train_num))    
        print(self.dev_num,self.train_num)
        
    def load_dat_new(self, fnm, full=False):
    
        pairs = []
        qids = []
        labels = []
      
        f = open(fnm,'r')
        data = json.loads(f.read())
        f.close()
        
        for qid in tqdm(data.keys()):
            items = data[qid]['items']
            cur_labels = data[qid]['labels']
            new_labels = []
            for la in cur_labels:
                if la == 0:
                    new_la = [1,0,0,0]
                elif la == 1:
                    new_la = [0,1,0,0]
                elif la == 2:
                    new_la = [0,0,1,0]
                elif la == 3:
                    new_la = [0,0,0,1]
                else:
                    ipdb.set_trace()
                new_labels = new_labels + new_la
            for item in items:
                for context in item:
                    article = context[0]
                    q = context[1]
                    op = context[2]                
                    pairs.append((article, q + '[SEP]' + op))
                    qids.append(qid)

            labels = labels + new_labels
            #ipdb.set_trace()
            
        
        
        print(len(qids),len(pairs),len(labels),fnm)
        return qids,pairs,labels

    def trans_samples(self, qids,pairs, labels,task_token="[unused3]"):

        samples = []

       
        for qid, (article, option), label in tqdm(zip(qids, pairs, labels)):
           # ipdb.set_trace()
            
            option_tok = self.tokenizer.tokenize(option)  
            article_tok = self.tokenizer.tokenize(article)
            #ipdb.set_trace()
            tokens = ["[CLS]"] + option_tok + ['[SEP]'] + article_tok
            
            input_type_ids = [0] * (len(option_tok) + 2) + [1] * len(article_tok)
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            samples.append((qid, input_ids, input_type_ids, label))
            #ipdb.set_trace()  
    
        return samples

    def iter_batches(self, which="train", samples=None, batch_size=None):

        if which == 'train':
            smp_idxs = self.train_idxs
          #  random.shuffle(smp_idxs)
            samples = self.train_samples
            batch_size = 32
        elif which == 'dev':
            smp_idxs = self.dev_idxs
            samples = self.dev_samples
            batch_size = 16
        elif which == 'test':
            smp_idxs = self.test_idxs
            samples = self.test_samples
            batch_size = 20
        else:
            raise Exception('which should be in [train, dev]!')
        batch_word_idxs, batch_type_idxs, batch_labels, batch_qid = [], [], [], []
        batch_dtags = []
        batch_poses = []
        end_idx = smp_idxs[-1]
     
        for smp_idx in smp_idxs:
            
            smp_infos = samples[smp_idx]
            qid, w_idxs, type_idxs, label, dtag, pos = smp_infos[:]
          #  ipdb.set_trace()
            batch_word_idxs.append(w_idxs)
            batch_type_idxs.append(type_idxs)
            batch_labels.append(label)
            batch_dtags.append(dtag)
            batch_qid.append(qid)
            batch_poses.append(pos)
            
            if len(batch_word_idxs) == batch_size or smp_idx == end_idx:
               # ipdb.set_trace()
                #max_len = min(max([len(_) for _ in batch_word_idxs]),512)  
                max_len = 512
                batch_word_idxs = self.padding_seq(batch_word_idxs, max_len=max_len)
                batch_type_idxs = self.padding_seq(batch_type_idxs, max_len=max_len)
                batch_dtags = self.padding_seq(batch_dtags, max_len=max_len)
                batch_poses = self.padding_seq(batch_poses, max_len=max_len)
                batch_labels = np.array(batch_labels)
                batch_word_idxs = np.array(batch_word_idxs)
                batch_type_idxs = np.array(batch_type_idxs)
                batch_dtags = np.array(batch_dtags)
                batch_poses = np.array(batch_poses)
               # ipdb.set_trace()
                yield batch_word_idxs, batch_type_idxs, batch_labels, batch_dtags, batch_poses, batch_qid
                batch_word_idxs, batch_type_idxs, batch_labels, batch_dtags, batch_poses, batch_qid = [], [], [], [], [], []

    
    def padding_seq(self, idxs, max_len=None, pad_unit=0):
       
        padded_idxs = []
        for seq in idxs:
            seq = seq[:max_len]
            padding_len = max_len - len(seq)
            for _ in range(padding_len):
                seq.append(pad_unit)
            padded_idxs.append(seq)
        return padded_idxs
    

#===============================================Model===================================================
class SubModel(nn.Module):
    def __init__(self):
        super(SubModel, self).__init__()
        pass
        
        self.model = ModelDefine()
        self.fc1 = nn.Linear(768,1)
        self.fc2 = nn.Linear(768,1)
        self.tanh = nn.Tanh()
        
    def forward(self, w_idxs1, type_idxs, mask_idxs, dtags, dtag_mask, poses):
        logits = self.model(w_idxs1, type_idxs, mask_idxs, dtags, dtag_mask, poses)
        #logits = logits.view(-1,4)
      #  ipdb.set_trace()
        logits1 = self.fc1(logits).view(-1,4)
        logits2 = self.tanh(self.fc2(logits)).view(-1,4)
        return logits1, logits2
    
class ModelDefine(nn.Module):
    def __init__(self):
        super(ModelDefine, self).__init__()
        pass

        path = './bertbase'
        self.bert = BertModel.from_pretrained(path)  # 定义一个模型
        self.fc = nn.Linear(768,1)
        self.dtag_fc = nn.Linear(230,1)        
        self.dtag_embedding = nn.Embedding(43, 64, padding_idx=0)
        self.pos_embedding = nn.Embedding(202, 64, padding_idx=0)
        self.dtag_encoder = TransformerEncoder(512,128,230)
        self.drop_out = nn.Dropout(0.1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, w_idxs1, type_idxs, mask_idxs, dtags, dtag_mask, poses):
        #ipdb.set_trace()
        dtag_emb = self.dtag_embedding(dtags)
        pos_emb = self.pos_embedding(poses)
        demb = torch.cat([dtag_emb, pos_emb],dim=-1)
        dtag_hiddens = self.dtag_encoder(demb,dtag_mask)
        dtag_gate = self.sigmoid(self.dtag_fc(dtag_hiddens)).squeeze(-1)        
        dtag_gate = dtag_gate.unsqueeze(1).expand(dtag_gate.size(0),512,512)
        dtag_gate = dtag_gate.unsqueeze(1).expand(dtag_gate.size(0),12,dtag_gate.size(1),dtag_gate.size(2))
        embedding_output = self.bert.embeddings(w_idxs1, type_idxs)
       
        extended_attention_mask = mask_idxs.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        #extended_attention_mask = None
        head_mask = [None] * 12
        
        encoded_layers = self.bert.encoder(embedding_output,extended_attention_mask, head_mask)
        
        last_layer = encoded_layers[-1]

        #max_pooling_fts = F.max_pool1d(last_layer.transpose(1, 2).contiguous(), kernel_size=last_layer.size(1)).squeeze(-1)
        logits = self.fc(max_pooling_fts)

        return logits

class TransformerEncoder(nn.Module):
    def __init__(self,max_length, dim_proj,hidden_size):
        super(TransformerEncoder, self).__init__()
        pass
        self.max_length = max_length
        self.hidden_size = hidden_size
        self.dim_proj =dim_proj
        self.head_count = 2
        self.dim_FNN = 256
        self.act_fn = torch.nn.ReLU()
        self.num_layers = 1
        self.dropout_rate = 0.5
       # self.position = PositionalEncoding(dim_proj,max_length)
        self.pool = nn.MaxPool1d(512)
        self.mlp_final = MLP(self.dim_proj,self.hidden_size)
        self._init_params()
        self.fc = nn.Linear(self.hidden_size,self.hidden_size)
    def _init_params(self):
        self.transformer = torch.nn.ModuleList([TransformerEncoderBlock(self.dim_proj) for _ in range(self.num_layers)])
        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, inp, mask):
        
        rval = []
       
        pre_output = inp 
      #  ipdb.set_trace()
        masks = torch.matmul(mask[:, :, None], mask[:, None, :]).float() #(400,40,40)
     
        for i in range(self.num_layers):
            
            cur_output = self.transformer[i](pre_output, masks)
            rval.append(cur_output)
            pre_output = cur_output  #(400,40,75)
        
        pre_output = self.mlp_final(pre_output)
        x_for_att = pre_output 
   
        x = self.pool(pre_output.transpose(1, 2))		
        x_mask = x.transpose(1,2)
     
        return x_for_att
      
class TransformerEncoderBlock(nn.Module):
     def __init__(self,dim_proj):
        super(TransformerEncoderBlock, self).__init__()
        pass

        self.dim_proj = dim_proj
        self.head_count = 2  #8
        self.dim_FNN = 256      #1024
        self.act_fn = torch.nn.ReLU()
        self.dropout_rate = 0.5

        self._init_params()

     def _init_params(self):
        self.w_1 = nn.Conv1d(self.dim_proj, self.dim_FNN, 1) # position-wise
        self.w_2 = nn.Conv1d(self.dim_FNN, self.dim_proj, 1) 
        self.multi_head_attention = Multi_Head_Attention(self.dim_proj)
        self.linear_proj_context = MLP(self.dim_proj, self.dim_proj)
        self.layer_norm_context = LayerNorm(self.dim_proj)
        self.position_wise_fnn = MLP(self.dim_proj, self.dim_FNN)
        self.linear_proj_intermediate = MLP(self.dim_FNN,self.dim_proj)
        self.layer_norm_intermediate = LayerNorm(self.dim_proj)
        self.layer_norm_final = LayerNorm(self.dim_proj)
        self.dropout = nn.Dropout(self.dropout_rate)
     
     def forward(self, inp, mask):
		
        output = self.multi_head_attention(inp, inp, inp, mask = mask)
       #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        
        residual = output
        context = output.transpose(1, 2)
        context = self.w_2(F.relu(self.w_1(context)))
        context = context.transpose(1, 2)
        context = self.dropout(context)
        context = self.layer_norm_intermediate(context + residual)
        
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    
        inp = self.linear_proj_context(inp)
        context = self.linear_proj_context(context)   # MLP(self.dim_proj, self.dim_proj)
        context = self.dropout(context)
        res_inp = self.layer_norm_context(inp + context)
        
        rval = self.act_fn(self.position_wise_fnn(res_inp))
        rval = self.linear_proj_intermediate(rval)
        
        res_rval = self.layer_norm_final(rval + res_inp)
        
        return res_rval

class Multi_Head_Attention(nn.Module):
    def __init__(self,dim_proj):
        super(Multi_Head_Attention, self).__init__()
        pass

        self.dim_proj = dim_proj
        self.head_count = 2
        self.dim_per_head = int(self.dim_proj/self.head_count)
        self.dropout_rate = 0.5

        self._init_params()

    def _init_params(self):
        self.linear_key = nn.Linear(self.dim_proj, self.head_count * self.dim_per_head)
        self.linear_value = nn.Linear(self.dim_proj, self.head_count * self.dim_per_head)
        self.linear_query = nn.Linear(self.dim_proj, self.head_count * self.dim_per_head)
		#nn.init.normal_(self.linear_key.weight, mean=0, std=np.sqrt(2.0 / (self.dim_proj + self.dim_per_head)))
		#nn.init.normal_(self.linear_value.weight, mean=0, std=np.sqrt(2.0 / (self.dim_proj + self.dim_per_head)))
		#nn.init.normal_(self.linear_query.weight, mean=0, std=np.sqrt(2.0 / (self.dim_proj + self.dim_per_head)))
        self.layer_norm = nn.LayerNorm(self.dim_proj)
        self.fc = nn.Linear(self.head_count * self.dim_per_head, self.dim_proj)
        self.dropout = nn.Dropout(self.dropout_rate)
		#nn.init.xavier_normal_(self.fc.weight)
        self.attention = ScaledDotProductAttention(temperature=np.power(self.dim_per_head, 0.5))
        self.softmax = nn.Softmax(dim=-1)
	
    def forward(self, key, value, query, mask = None):
		# key: batch X key_len X hidden
		# value: batch X value_len X hidden
		# query: batch X query_len X hidden
		# mask: batch X query_len X key_len
        batch_size = key.size()[0]  #400
        residual = query  
        #print key.size()   (400,40,60)
       
        key_ = self.linear_key(key)
        #print key_.size()   #(400,40,75)
        value_ = self.linear_value(value)
        query_ = self.linear_query(query)
        #print key_.size()
		#Q=V=K

		#key_ = key_.reshape(batch_size, -1, self.head_count, self.dim_per_head).transpose(1, 2)
		#value_ = value_.reshape(batch_size, -1, self.head_count, self.dim_per_head).transpose(1, 2)
		#query_ = query_.reshape(batch_size, -1, self.head_count, self.dim_per_head).transpose(1, 2)
		#__import__("ipdb").set_trace()
		#attention_scores = torch.matmul(query_, key_.transpose(2, 3))
		#attention_scores = attention_scores / math.sqrt(float(self.dim_per_head))

		#if mask is not None:
			#mask = mask.unsqueeze(1).expand_as(attention_scores)
			#attention_scores = attention_scores.masked_fill(1 - mask, -1e18)

		#attention_probs = self.softmax(attention_scores)
		#attention_probs = self.dropout(attention_probs)

		#context = torch.matmul(attention_probs, value_)
		#context = context.transpose(1, 2).reshape(batch_size, -1, self.head_count * self.dim_per_head)

        #"=====================================transformer===================================="
        key_ = key_.reshape(batch_size, -1, self.head_count, self.dim_per_head)
        value_ = value_.reshape(batch_size, -1, self.head_count, self.dim_per_head)
        query_ = query_.reshape(batch_size, -1, self.head_count, self.dim_per_head)
       # print key_.size()  (400,40,6,10)
                                                   
        query_ = query_.permute(2, 0, 1, 3).contiguous().view(-1, 512, self.dim_per_head) # (n*b) x lq x dk
        key_ = key_.permute(2, 0, 1, 3).contiguous().view(-1, 512, self.dim_per_head) # (n*b) x lk x dk
        value_  = value_ .permute(2, 0, 1, 3).contiguous().view(-1, 512, self.dim_per_head) # (n*b) x lv x dv
        #print key_.size()   (2400,40,10)
        #ipdb.set_trace()
        mask = mask.repeat(self.head_count, 1, 1).byte() # (n*b) x .. x ..
        output, attn = self.attention(query_,key_ ,value_,mask = mask)
		
        output = output.view(self.head_count, batch_size, 512, self.dim_per_head)
        output = output.permute(1, 2, 0, 3).contiguous().view(batch_size, 512, -1) # b x lq x (n*dv)
        
        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)
		#"=====================================transformer===================================="


        return output

class ScaledDotProductAttention(nn.Module):


    def __init__(self, temperature, attn_dropout=0.1):
        super(ScaledDotProductAttention,self).__init__()
        pass
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, qu, ke, va, mask=None):
   
        attn = torch.bmm(qu, ke.transpose(1, 2))
        attn = attn / self.temperature
 
    
        if mask is not None:
       
            attn = torch.where(mask == 0 , torch.full_like(mask, 1e-8), mask).type(torch.cuda.FloatTensor)
   
        attn = self.softmax(attn)
  
        attn = self.dropout(attn)
    
        output = torch.bmm(attn, va)
    

        return output, attn

      
class MLP(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(MLP, self).__init__()
        pass
        self.dim_in = dim_in
        self.dim_out = dim_out

        self._init_params()

    def _init_params(self):
        self.mlp = nn.Linear(in_features = self.dim_in,out_features = self.dim_out)

    def forward(self, inp):
        proj_inp = self.mlp(inp)
        return proj_inp
      
      
class LayerNorm(nn.Module):
    """Layer Normalization class"""
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        pass
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class PositionalEncoding(nn.Module):
    
    def __init__(self, d_model, max_seq_len):

        super(PositionalEncoding, self).__init__()
        pass
        
    
        position_encoding = np.array([
          [pos / np.power(10000, 2.0 * (j // 2) / d_model) for j in range(d_model)]
          for pos in range(max_seq_len)])
   
        position_encoding[:, 0::2] = np.sin(position_encoding[:, 0::2])
        position_encoding[:, 1::2] = np.cos(position_encoding[:, 1::2])
        pad_row = torch.zeros([1, d_model])
        position_encoding = torch.from_numpy(position_encoding)
        position_encoding = torch.cat([pad_row, position_encoding.float()],dim=0)

        self.position_encoding = nn.Embedding(max_seq_len + 1, d_model)
        self.position_encoding.weight = nn.Parameter(position_encoding,
                                                     requires_grad=False)
    def forward(self, input_len):
    
        max_len = torch.max(input_len)
        tensor = torch.cuda.LongTensor if input_len.is_cuda else torch.LongTensor
        input_pos = tensor(
          [list(range(1, len + 1)) + [0] * (max_len - len) for len in input_len])
        return self.position_encoding(input_pos)       
              
       
    
class Model:
    def __init__(self, lr=5e-5, device=None):

        
        self.submodel = SubModel() 
        if torch.cuda.is_available():
            self.submodel.cuda()

        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            self.submodel.model =  nn.DataParallel(self.submodel.model)  
        self.train_loss = AverageMeter()
        self.updates = 0
        opt_layers = list(range(0, 12, 1))
        #self.loss = nn.BCEWithLogitsLoss()
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = optim.Adamax([p for p in self.submodel.parameters() if p.requires_grad], lr=lr)
        self.dt = DataManager()
        #self.dt.resample()
        self.flag = 0
        self.softmax = nn.Softmax(dim=-1)
        self.KL = torch.nn.KLDivLoss(reduction='batchmean')
       

    def train(self):

        for i in range(30):
           # if i>0:
           #     self.dt.resample()
            print("===" * 10)
            print("epoch%d" % i)
            for batch in tqdm(self.dt.iter_batches(which="train")):
                batch_size = len(batch[0])
                self.submodel.train()         
                word_idxs, type_idxs, labels, dtags, poses = [Variable(torch.from_numpy(e)).long().to('cuda') for e in batch[:-1]]              
                dtag_mask = (dtags != 0).float()
                attention_mask = (word_idxs > 0.5).long().to('cuda')
                logits1, logits2 = self.submodel(word_idxs, type_idxs, attention_mask, dtags, dtag_mask, poses)
                logits1 = logits1.view(-1,4)
                logits2 = logits2.view(-1,4)
                labels = labels.view(-1,4)
                _,cos_labels = torch.max(labels,dim=-1)
                loss1 = self.loss(logits1, cos_labels)
                _,pred = torch.max(logits1,dim=-1)
                mask = (pred != cos_labels).float().unsqueeze(-1)
                loss2 = self.loss(logits2, pred)
                pr = self.softmax(logits1) * mask
                pf = self.softmax(logits2) * mask
                loss = loss1+loss2-self.KL(pr,pf)

                self.train_loss.update(loss.item(), batch_size)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.updates += 1
            print("epoch {}, train loss={}".format(i, self.train_loss.avg))
            self.train_loss.reset() 
            torch.cuda.empty_cache()
            self.validate(epoch=i)

               
    def validate(self, which="dev", epoch=-1):
        global max_cqnum
        softmax_func = nn.Softmax(dim=-1) 
        all_question_num =0
        correct_question = 0
        
        for batch in tqdm(self.dt.iter_batches(which=which)):
            batch_size = len(batch[0])
            all_question_num = all_question_num + (batch_size/4)
            self.submodel.eval()
            word_idxs, type_idxs, labels, dtags, poses = [Variable(torch.from_numpy(e)).long().to('cuda') for e in batch[:-1]]
            dtag_mask = (dtags != 0).float()
            batch_qid = batch[-1]
            attention_mask = (word_idxs > 0.5).long().to('cuda')
            logits,_ = self.submodel(word_idxs, type_idxs,attention_mask,dtags,dtag_mask,poses) 
        
            logits = logits.view(-1,4)
            _,cos_logits = torch.max(softmax_func(logits),dim=-1)
            labels = labels.view(-1,4)
            _,cos_labels = torch.max(labels,dim=-1)

            correct_question += sum(cos_labels == cos_logits).item()
            
            
        
        
        
        print('correct question num = {}'.format(correct_question))
        print("question level acc={}".format(correct_question/all_question_num))
        if correct_question >= max_cqnum:
            max_cqnum = correct_question
           
            self.save("models/race_cdt_6.26.pt", epoch)

    

    def save(self, filename, epoch):
        params = {
            'state_dict': {
                'network': self.submodel.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'updates': self.updates
            },
            'epoch': epoch
        }
        torch.save(params, filename)

    def resume(self, filename):
        
        checkpoint = torch.load(filename)
        state_dict = checkpoint['state_dict']
        self.submodel.load_state_dict(state_dict['network'])
        self.submodel.to('cuda')
        return self.submodel



if __name__ == '__main__':
    synonym_model = Model()
    synonym_model.train()
