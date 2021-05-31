import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch import optim
from . import network
import sys
import ipdb
sys.path.append('..')
import fewshot_re_kit
from torch import autograd, optim, nn
import numpy as np
from torch.autograd import Variable
from pytorch_pretrained_bert.tokenization import BertTokenizer
from fewshot_re_kit.network import embedding
from fewshot_re_kit.network import encoder
      
      
class TransformerSentenceEncoder(nn.Module):
  
    def __init__(self,word_vec_mat, max_length=100, word_embedding_dim=768, dpos_embedding_dim=50,dmask_embedding_dim=50,hidden_size=230):
        nn.Module.__init__(self)
        self.hidden_size = hidden_size
        self.max_length = max_length
        self.embedding = network.embedding.Embedding(word_vec_mat, max_length, word_embedding_dim, dpos_embedding_dim,dmask_embedding_dim)
        self.dmask1_embedding = nn.Embedding(2*self.max_length,dmask_embedding_dim,padding_idx=0)
        self.dmask2_embedding = nn.Embedding(2*self.max_length,dmask_embedding_dim,padding_idx=0)
        self.dpos1_embedding = nn.Embedding(2*self.max_length,50,padding_idx=0)
        self.dpos2_embedding = nn.Embedding(2*self.max_length,50,padding_idx=0)        
        self.encoder2 = network.encoder.TransformerEncoder(max_length,4*dmask_embedding_dim,hidden_size)
        self.fc = nn.Linear(self.hidden_size,1)
        self.bpool = nn.MaxPool1d(max_length)


     
    def forward(self, inputs,sent,N,K):
        dpos1 = inputs['dpos1']
        dpos2 = inputs['dpos2']
        dpos1_emb = self.dpos1_embedding(dpos1)
        dpos2_emb  = self.dpos2_embedding(dpos2)  
        dmask1 = inputs['dmask1']
        dmask2 = inputs['dmask2']

        demb1 = self.dmask1_embedding(dmask1)
        demb2 = self.dmask2_embedding(dmask2)
        demb = torch.cat([demb1,demb2,dpos1_emb,dpos2_emb],dim=-1)
        m = torch.ones(dmask1.size())
   
     
        den_tran1 = self.encoder2(demb,mask=m,gate=None)
   
        gate = torch.sigmoid(self.fc(den_tran1)).squeeze(-1)
        finalgate=gate.cpu()

        gate = gate.unsqueeze(1).expand(gate.size(0),100,100)
    
       
        x_emb,mask = self.embedding(inputs,sent,gate.cuda())
     
        x = self.bpool(x_emb.transpose(1,2))
    
        return x,finalgate
