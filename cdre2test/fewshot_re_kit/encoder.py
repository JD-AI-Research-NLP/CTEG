import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from torch import optim
import numpy as np
import ipdb


class Encoder(nn.Module):
    def __init__(self, max_length,word_embedding_dim=768,dpos_embedding_dim=50,dmask_embedding_dim=50,hidden_size=230):
        nn.Module.__init__(self)

        self.max_length = max_length
        self.hidden_size = hidden_size
        self.embedding_dim = word_embedding_dim+2*dpos_embedding_dim
        self.conv = nn.Conv1d(self.embedding_dim, self.hidden_size, 3, padding=1)
        self.pool = nn.MaxPool1d(40)
        self.bpool = nn.MaxPool1d(max_length)
        #self.cost = nn.MSELoss(size_average=False) 
        # For PCNN
        self.mask_embedding = nn.Embedding(4, 3)
        self.mask_embedding.weight.data.copy_(torch.FloatTensor([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]]))
        self.mask_embedding.weight.requires_grad = False
        self._minus = -100
        self.fc = nn.Linear(self.embedding_dim, self.embedding_dim)
        #self.segfc = nn.Linear(self.hidden_size,self.hidden_size)
        self.attfc = nn.Linear(self.hidden_size,self.hidden_size)
        self.dfc = nn.Linear(200,self.hidden_size)
        self.dropout = nn.Dropout(0.3)
    def forward(self, inputs):
        return self.cnn(inputs)
    

        
    def cnn(self, inputs):
        inputs = F.relu(self.fc(inputs))
        x = self.conv(inputs.transpose(1, 2))   #x(400,230,40)
        x = F.relu(x)        
        x = self.pool(x)
        return x.squeeze(2) # n x hidden_size

    def pcnn(self, inputs):
    
        inputs = F.relu(self.fc(inputs))
   
        x = self.conv(inputs.transpose(1, 2))#400,230,40
     
        #========================================================================
        '''
        attmask = mask.unsqueeze(1).repeat(1,100,1)
        finalmask = mask.unsqueeze(1)*mask.unsqueeze(2)
        x_foratt = x.transpose(1,2)
        demb = torch.cat([demb1,demb2],-1)
        demb = self.dfc(demb)
        x_foratt = self.attfc(x_foratt)
        demb = self.attfc(demb)
        #__import__("ipdb").set_trace()       
        att_score = (x_foratt.unsqueeze(2) * demb.unsqueeze(1)).sum(-1)
        attinf = torch.full(att_score.size(),float("-inf")).cuda()         
        att_score = torch.where(attmask==1,att_score,attinf)
        att_score = torch.softmax(att_score,dim=-1)*finalmask
        x_att = (x_foratt.unsqueeze(-2)* att_score.unsqueeze(-1)).sum(-2)#.unsqueeze(-1)
        x_att = self.bpool(x_att.transpose(1,2))
        '''
        #================================================================================================
      
        x_nopool = x        #x_cat = torch.cat([x,bert.transpose(1,2)],1)
 
        x_ori = self.bpool(x)
  
        '''
        x_segatt = self.segfc(x_att.transpose(1,2))
        x_nopool = self.segfc(x_nopool.transpose(1,2))
        finalatt = (x_segatt*x_nopool).sum(-1)
        attinf = torch.full(finalatt.size(),float("-inf")).cuda()
        finalatt = torch.where(mask == 1, finalatt,attinf)
        finalatt = F.softmax(finalatt,dim = -1)
        finalx = (x_nopool * finalatt.unsqueeze(-1)).transpose(1,2).sum(-1).unsqueeze(-1)
        #finalx = self.bpool(finalx)
       
        
        #==================================bert segment attention begin=====================================
        x_nopool = x_nopool.transpose(1,2)
        q_seg = torch.zeros(30,x_nopool.size(0),x_nopool.size(2))
        one = torch.ones(segmask.size()).cpu()
        zero = torch.zeros(segmask.size()).cpu()
        segmask = segmask.cpu()
        for i in range(1,30):
          mask = torch.where(segmask == i,one,zero)
          q_per_seg = x_nopool * mask.unsqueeze(-1).cuda()
          q_per_seg = torch.mean(q_per_seg,1)
          q_seg[i-1] = q_per_seg
        
        q_seg = q_seg.transpose(0,1).cuda()
        q_seg = self.segfc(q_seg)
        x_att = self.segfc(x_att.transpose(1,2))
        segatt_score = torch.tanh(q_seg*x_att).sum(-1)
        #ipdb.set_trace()
        lenth,_ = segmask.max(-1)
        padmask = torch.zeros(segatt_score.size()).cuda()
        for i in range(0,padmask.size(0)):
          padmask[i][0:lenth[i]] = 1
        #ipdb.set_trace()
        padinf = torch.full(padmask.size(),float('-inf')).cuda()
        segatt_score = torch.where(padmask == 1,segatt_score,padinf)
        segatt_score = F.softmax(segatt_score,dim=-1)  
        att_score_final = torch.zeros(segatt_score.size(0),100).cuda()        
        for i in range(1,30):
          mask = torch.where(segmask == i,one,zero).cuda()
          att_score_perseg = segatt_score[:,i-1].unsqueeze(1).repeat(1,100)
          att_score_final = att_score_final + att_score_perseg * mask
        att_query = (x_nopool * att_score_final.unsqueeze(-1)).sum(-2)
        '''
               
        return x_ori
      

class TransformerEncoder(nn.Module):
    def __init__(self,max_length, dim_proj,hidden_size=230):
        super(TransformerEncoder, self).__init__()

        self.dim_proj =dim_proj
        self.head_count = 2
        self.dim_FNN = 256
        self.act_fn = torch.nn.ReLU()
        self.num_layers = 1
        self.dropout_rate = 0.5
       # self.position = PositionalEncoding(dim_proj,max_length)
        self.pool = nn.MaxPool1d(100)
        self.mlp_final = MLP(self.dim_proj,hidden_size)
        self._init_params()
        self.fc = nn.Linear(hidden_size,hidden_size)
    def _init_params(self):
        self.transformer = torch.nn.ModuleList([TransformerEncoderBlock(self.dim_proj) for _ in range(self.num_layers)])
        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, inp, mask = None,dmask = None,gate = None):
        
        rval = []
       
        pre_output = inp 
    
        mask = mask.float()
        masks = torch.matmul(mask[:, :, None], mask[:, None, :]).float() #(400,40,40)
     
        for i in range(self.num_layers):
            
            cur_output = self.transformer[i](pre_output, masks,gate=gate)
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
     
     def forward(self, inp, mask,gate):
		
        output = self.multi_head_attention(inp, inp, inp, mask = mask,gate=gate)
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
	
    def forward(self, key, value, query, mask = None,gate=None):
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
                                                   
        query_ = query_.permute(2, 0, 1, 3).contiguous().view(-1, 100, self.dim_per_head) # (n*b) x lq x dk
        key_ = key_.permute(2, 0, 1, 3).contiguous().view(-1, 100, self.dim_per_head) # (n*b) x lk x dk
        value_  = value_ .permute(2, 0, 1, 3).contiguous().view(-1, 100, self.dim_per_head) # (n*b) x lv x dv
        #print key_.size()   (2400,40,10)
        #ipdb.set_trace()
        mask = mask.repeat(self.head_count, 1, 1).byte() # (n*b) x .. x ..
        output, attn = self.attention(query_,key_ ,value_,mask = mask,gate=gate)
		
        output = output.view(self.head_count, batch_size, 100, self.dim_per_head)
        output = output.permute(1, 2, 0, 3).contiguous().view(batch_size, 100, -1) # b x lq x (n*dv)
        
        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)
		#"=====================================transformer===================================="


        return output

class ScaledDotProductAttention(nn.Module):


    def __init__(self, temperature, attn_dropout=0.1):
        super(ScaledDotProductAttention,self).__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, qu, ke, va, mask=None,gate=None):
   
        attn = torch.bmm(qu, ke.transpose(1, 2))
        attn = attn / self.temperature
 
        if not gate is None:
     
            gate = gate.repeat(2, 1, 1)
            attn = attn * gate
        if mask is not None:
       
            attn = torch.where(mask == 0 , torch.full_like(mask, 1e-8), mask).type(torch.cuda.FloatTensor)
   
        attn = self.softmax(attn)
  
        attn = self.dropout(attn)
    
        output = torch.bmm(attn, va)
    

        return output, attn

      
class MLP(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(MLP, self).__init__()

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
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class PositionalEncoding(nn.Module):
    
    def __init__(self, d_model, max_seq_len):
        """初始化。
        
        Args:
            d_model: 一个标量。模型的维度，论文默认是512
            max_seq_len: 一个标量。文本序列的最大长度
        """
        super(PositionalEncoding, self).__init__()
        
    
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
        """神经网络的前向传播。

        Args:
          input_len: 一个张量，形状为[BATCH_SIZE, 1]。每一个张量的值代表这一批文本序列中对应的长度。

        Returns:
          返回这一批序列的位置编码，进行了对齐。
        """
        
    
        max_len = torch.max(input_len)
        tensor = torch.cuda.LongTensor if input_len.is_cuda else torch.LongTensor
        input_pos = tensor(
          [list(range(1, len + 1)) + [0] * (max_len - len) for len in input_len])
        return self.position_encoding(input_pos)