import sys
sys.path.append('..')
import fewshot_re_kit
import torch
from torch import autograd, optim, nn
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np


class Proto(fewshot_re_kit.framework.FewShotREModel):
    
    def __init__(self, sentence_encoder, hidden_size=768):
        fewshot_re_kit.framework.FewShotREModel.__init__(self, sentence_encoder)
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(5, 5)
        self.fc2 = nn.Linear(5, 5)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()
     
        self.index = 0
        self.allre = 0
        self.one = 0
        self.two = 0
        self.three = 0
        
    def __dist__(self, x, y, dim):
        #ipdb.set_trace()
        return (torch.pow(x - y, 2)).sum(dim)

    def __batch_dist__(self, S, Q):
        #ipdb.set_trace()
        return self.__dist__(S.unsqueeze(1), Q.unsqueeze(2), 3)
    
    def forward(self, support, query, N, K, Q,label,target_class):
        '''
        support: Inputs of the support set.
        query: Inputs of the query set.
        N: Num of classes
        K: Num of instances for each class in the support set
        Q: Num of instances for each class in the query set
        ''' 

             
        
        orisent_s = support['sent']
        sent_s = [x for z in orisent_s for y in z for x in y]
        support,_= self.sentence_encoder(support,sent_s,N,K) 
        orisent_q = query['sent']
        sent_q = [x for z in orisent_q for x in z]
        query,gate= self.sentence_encoder(query,sent_q,N,Q) # (B * N * Q, D)
        support = support.view(-1, N, K, self.hidden_size) # (B, N, K, D)
        query  = query.view(-1, N * Q, self.hidden_size) # (B, N * Q, D)
    
  
  
        B = support.size(0) # Batch size
        NQ = query.size(1) # Num of instances for each batch in the query set
        D = support.size(-1)
        support = torch.mean(support, 2) # Calculate prototype for each class
        # Prototypical Networks 


        logits = -self.__batch_dist__(support, query)
        mask = torch.zeros(label.size())
        
        
        if self.training:
            label2 = torch.clone(label)
            logits1 = logits
            logits2 = self.tanh(self.fc2(logits))
            _, pred = torch.max(logits1.view(-1, N), -1)
           
            for i in range(0,label.size(1)):
                if label[0][i] != pred[i]:
                    label2[0][i] = pred[i]
                    mask[0][i] = 1
        else:
         
            logits1 = logits
            logits2 = logits
            _, pred = torch.max(logits1.view(-1, N), -1)
            label2 = label
       
        mask = mask.unsqueeze(-1).cuda()
  
      
        return logits1, pred ,label2, logits2, mask
