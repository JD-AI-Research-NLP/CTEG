import json
import os
import multiprocessing
import numpy as np
import random
import torch

import ipdb
from stanfordcorenlp import StanfordCoreNLP
from torch.autograd import Variable
from math import log
from pytorch_pretrained_bert.tokenization import BertTokenizer
#nlp = StanfordCoreNLP(r'./stanford-corenlp-full-2018-10-05')

class FileDataLoader:
    def next_batch(self, B, N, K, Q):
        '''
        B: batch size.
        N: the number of relations for each batch
        K: the number of support instances for each relation
        Q: the number of query instances for each relation
        return: support_set, query_set, query_label
        '''
        raise NotImplementedError

class JSONFileDataLoader(FileDataLoader):
    def _load_preprocessed_file(self):
        name_prefix = '.'.join(self.file_name.split('/')[-1].split('.')[:-1])
    
        word_vec_name_prefix = '.'.join(self.word_vec_file_name.split('/')[-1].split('.')[:-1])
        processed_data_dir = '_processed_data'
        if not os.path.isdir(processed_data_dir):
            return False
        word_npy_file_name = os.path.join(processed_data_dir, name_prefix + '_word.npy')
        sent_npy_file_name = os.path.join(processed_data_dir, name_prefix + '_sent.npy')
        dmask1_npy_file_name = os.path.join(processed_data_dir, name_prefix + '_dmask1.npy')
        dmask2_npy_file_name = os.path.join(processed_data_dir, name_prefix + '_dmask2.npy')
        pos1_npy_file_name = os.path.join(processed_data_dir, name_prefix + '_pos1.npy')
        dpos1_npy_file_name = os.path.join(processed_data_dir, name_prefix + '_dpos1.npy')
        pos2_npy_file_name = os.path.join(processed_data_dir, name_prefix + '_pos2.npy')
        dpos2_npy_file_name = os.path.join(processed_data_dir, name_prefix + '_dpos2.npy')
        mask_npy_file_name = os.path.join(processed_data_dir, name_prefix + '_mask.npy')
        length_npy_file_name = os.path.join(processed_data_dir, name_prefix + '_length.npy')
        rel2scope_file_name = os.path.join(processed_data_dir, name_prefix + '_rel2scope.json')
        word_vec_mat_file_name = os.path.join(processed_data_dir, word_vec_name_prefix + '_mat.npy')
        word2id_file_name = os.path.join(processed_data_dir, word_vec_name_prefix + '_word2id.json')
        if not os.path.exists(word_npy_file_name) or \
           not os.path.exists(sent_npy_file_name) or \
           not os.path.exists(pos1_npy_file_name) or \
           not os.path.exists(dpos1_npy_file_name) or \
           not os.path.exists(pos2_npy_file_name) or \
           not os.path.exists(dpos2_npy_file_name) or \
           not os.path.exists(dmask1_npy_file_name) or \
           not os.path.exists(dmask2_npy_file_name) or \
           not os.path.exists(length_npy_file_name) or \
           not os.path.exists(rel2scope_file_name) or \
           not os.path.exists(word_vec_mat_file_name) or \
           not os.path.exists(word2id_file_name):
            return False
        print("Pre-processed files exist. Loading them...")
        self.data_word = np.load(word_npy_file_name)
        self.data_sent = np.load(sent_npy_file_name)
        self.data_dmask1 = np.load(dmask1_npy_file_name)
        self.data_dmask2 = np.load(dmask2_npy_file_name)
        self.data_pos1 = np.load(pos1_npy_file_name)
        self.data_dpos1 = np.load(dpos1_npy_file_name)
        self.data_pos2 = np.load(pos2_npy_file_name)
        self.data_dpos2 = np.load(dpos2_npy_file_name)
        self.data_mask = np.load(mask_npy_file_name)
        self.rel2scope = json.load(open(rel2scope_file_name))
        self.word_vec_mat = np.load(word_vec_mat_file_name)
        self.word2id = json.load(open(word2id_file_name))
        self.name_prefix = '.'.join(self.file_name.split('/')[-1].split('.')[:-1])
        if self.data_word.shape[1] != self.max_length   :
            print("Pre-processed files don't match current settings. Reprocessing...")
            return False
        print("Finish loading")
        return True

    def __init__(self, file_name, word_vec_file_name, max_length=40, case_sensitive=False, reprocess=False, cuda=True):
        '''
        file_name: Json file storing the data in the following format
            {
                "P155": # relation id
                    [
                        {
                            "h": ["song for a future generation", "Q7561099", [[16, 17, ...]]], # head entity [word, id, location]
                            "t": ["whammy kiss", "Q7990594", [[11, 12]]], # tail entity [word, id, location]
                            "token": ["Hot", "Dance", "Club", ...], # sentence
                        },
                        ...
                    ],
                "P177": 
                    [
                        ...
                    ]
                ...
            }
        word_vec_file_name: Json file storing word vectors in the following format
            [
                {'word': 'the', 'vec': [0.418, 0.24968, ...]},
                {'word': ',', 'vec': [0.013441, 0.23682, ...]},
                ...
            ]
        max_length: The length that all the sentences need to be extend to.
        case_sensitive: Whether the data processing is case-sensitive, default as False.
        reprocess: Do the pre-processing whether there exist pre-processed files, default as False.
        cuda: Use cuda or not, default as True.
        '''
        self.file_name = file_name
        self.word_vec_file_name = word_vec_file_name
        self.case_sensitive = case_sensitive
        self.max_length = max_length
        self.cuda = cuda
        self.name_prefix = '.'.join(self.file_name.split('/')[-1].split('.')[:-1])
        #ipdb.set_trace()
        if reprocess or not self._load_preprocessed_file(): # Try to load pre-processed files:
            # Check files
            if file_name is None or not os.path.isfile(file_name):
                raise Exception("[ERROR] Data file doesn't exist")
            if word_vec_file_name is None or not os.path.isfile(word_vec_file_name):
                raise Exception("[ERROR] Word vector file doesn't exist")

            # Load files
            print("Loading data file...")
            self.ori_data = json.load(open(self.file_name, "r"))
            print("Finish loading")
            print("Loading word vector file...")
            self.ori_word_vec = json.load(open(self.word_vec_file_name, "r"))
            #self.ori_word_vec = self.ori_word_vec[0:len(self.ori_word_vec)/10]
            print("Finish loading")
            
            # Eliminate case sensitive
            if not case_sensitive:
                print("Elimiating case sensitive problem...")

                for relation in self.ori_data:
                    for ins in self.ori_data[relation]:
                        for i in range(len(ins['tokens'])):
                            #print  ins['tokens']
                            ins['tokens'][i] = ins['tokens'][i].lower()
                print("Finish eliminating")

      
            # Pre-process word vec
            self.word2id = {}
            self.word_vec_tot = len(self.ori_word_vec)
            UNK = self.word_vec_tot
            
            #print UNK
            BLANK = self.word_vec_tot + 1
            self.word_vec_dim = len(self.ori_word_vec[0]['vec'])
            print("Got {} words of {} dims".format(self.word_vec_tot, self.word_vec_dim))
            print("Building word vector matrix and mapping...")
            #ipdb.set_trace()
            self.word_vec_mat = np.zeros((self.word_vec_tot, self.word_vec_dim), dtype=np.float32)
            for cur_id, word in enumerate(self.ori_word_vec):
                w = word['word']
                if not case_sensitive:
                    w = w.lower()
                self.word2id[w] = cur_id
                self.word_vec_mat[cur_id, :] = word['vec']
                self.word_vec_mat[cur_id] = self.word_vec_mat[cur_id] / np.sqrt(np.sum(self.word_vec_mat[cur_id] ** 2))
            self.word2id['UNK'] = UNK
            self.word2id['BLANK'] = BLANK
            #print self.word_vec_mat[-1]
            print("Finish building")
            #print self.word2id
            # Pre-process data
            print("Pre-processing data...")
            self.instance_tot = 0
            for relation in self.ori_data:
                self.instance_tot += len(self.ori_data[relation])
             
            self.data_word = np.zeros((self.instance_tot, self.max_length), dtype=np.int32)
            self.data_sent = []
            self.data_dmask1 = np.zeros((self.instance_tot, 100),dtype=np.int32)
            self.data_dmask2 = np.zeros((self.instance_tot, 100),dtype=np.int32)
            self.data_pos1 = np.zeros((self.instance_tot, self.max_length), dtype=np.int32)
            self.data_dpos1 = np.zeros((self.instance_tot, 100), dtype=np.int32)
            self.data_pos2 = np.zeros((self.instance_tot, self.max_length), dtype=np.int32)
            self.data_dpos2 = np.zeros((self.instance_tot, 100), dtype=np.int32)
            self.data_mask = np.zeros((self.instance_tot, 100), dtype=np.int32)
            self.data_length = np.zeros((self.instance_tot), dtype=np.int32)
            self.rel2scope = {} # left close right open
            
            
#===============================get dtree and dsent ==========================================================    
                           
            print('tree')          
            for relation in self.ori_data:
              print(relation)
              for ins in self.ori_data[relation]:
                words = ins['tokens']
                sent = " ".join(words)
                self.data_sent.append(sent)
                sent,dtree = self.get_tree(ins)
                fp = open("./dtree/dtree"+relation, "a",encoding="utf-8")       
                for i in range(0,len(dtree)):
                    fp.write(str(dtree[i])+'\n')
                fp.write("#+\n")
                fp.close()
                    
                fp2 = open("./dsent/dsent"+relation,"a",encoding="utf-8")
                fp2.write(sent.replace('\r','').replace('\t','').replace('\n','')+'\n')    
                fp2.close()
                
#==============================get dmask======================================================================
            
            print('damsk')
            all_dmask1 = {}
            all_dmask2 = {}
            for relation in self.ori_data:
                #print(relation)
                dmask1_par,dmask2_par = self.get_mask(relation)
                all_dmask1[relation] = dmask1_par
                all_dmask2[relation] = dmask2_par
            #ipdb.set_trace()
            fp = open("dmask1_"+self.name_prefix,"a")
            fp.write(str(all_dmask1))
            fp.close()
            fp2 = open("dmask2_"+self.name_prefix,"a")
            fp2.write(str(all_dmask2))
            fp2.close()
            #ipdb.set_trace()
            
#==============================get bert tokens===============================================================
                    
            print('berttoken')
            bert_token = BertTokenizer.from_pretrained('./models/bert-base-uncased-vocab.txt')
            fp3 = open('bert_tokens_'+self.name_prefix,'a')
            for relation in self.ori_data:
              for ins in self.ori_data[relation]:
                words = ins['tokens']
                sent = " ".join(words)      
                token = bert_token.tokenize(sent)     
                token.insert(0,'[CLS]')
                token.append('[SEP]')
                #ipdb.set_trace()
                fp3.write(str(token))
                fp3.write('\n')
                #ipdb.set_trace()
            fp3.close()
            
            
            
            print('bert_dmask')
            bert_dmask1,bert_dmask2 = self.get_bert_dmask(self.ori_data)
            self.get_bert_pos_and_bseg(self.ori_data)
            i=0
            for relation in self.ori_data:
                print (relation)
                #__import__("ipdb").set_trace()
                self.rel2scope[relation] = [i, i]
                word_in_r = []
                word_total_r = []
                cur_re_dmask1 = bert_dmask1[relation]
                cur_re_dmask2 = bert_dmask2[relation]
                   
                ins_idx = 0
                #cur_re_ctree_tailword = self.get_ctree_segment(relation) 
                for ins in self.ori_data[relation]:
                    #ipdb.set_trace()
                    head = ins['h'][0]
                    tail = ins['t'][0]
                    pos1 = ins['h'][2][0][0]
                    pos2 = ins['t'][2][0][0]                    
                    words = ins['tokens']
                    #ipdb.set_trace()
                    sent = " ".join(words)
                    self.data_sent.append(sent)
                    cur_ins_dmask1 = cur_re_dmask1[ins_idx] 
                    cur_ins_dmask2 = cur_re_dmask2[ins_idx] 
                    #cur_ins_ctree_tailword = cur_re_ctree_tailword[ins_idx]
                    ins_idx += 1
                    ##ipdb.set_trace()
                    cur_ref_data_word = self.data_word[i]
                    self.data_dmask1[i] = cur_ins_dmask1 
                    self.data_dmask2[i] = cur_ins_dmask2

             

                    if len(words) > max_length:
                        self.data_length[i] = max_length
                    if pos1 >= max_length:
                        pos1 = max_length - 1
                    if pos2 >= max_length:
                        pos2 = max_length - 1
                    pos_min = min(pos1, pos2)
                    pos_max = max(pos1, pos2)
                    for j in range(max_length):
                        self.data_pos1[i][j] = j - pos1 + max_length
                        self.data_pos2[i][j] = j - pos2 + max_length
                    #print(i)
                    i += 1
                self.rel2scope[relation][1] = i 
            
            print("Finish pre-processing")     
            
            print("Storing processed files...")
            name_prefix = '.'.join(file_name.split('/')[-1].split('.')[:-1])
            word_vec_name_prefix = '.'.join(word_vec_file_name.split('/')[-1].split('.')[:-1])
            processed_data_dir = '_processed_data'
            if not os.path.isdir(processed_data_dir):
                os.mkdir(processed_data_dir)

            np.save(os.path.join(processed_data_dir, name_prefix + '_word.npy'), self.data_word)
            np.save(os.path.join(processed_data_dir, name_prefix + '_sent.npy'), self.data_sent)
            np.save(os.path.join(processed_data_dir, name_prefix + '_dmask1.npy'), self.data_dmask1)
            np.save(os.path.join(processed_data_dir, name_prefix + '_dmask2.npy'), self.data_dmask2)
            np.save(os.path.join(processed_data_dir, name_prefix + '_pos1.npy'), self.data_pos1)
            np.save(os.path.join(processed_data_dir, name_prefix + '_dpos1.npy'), self.data_dpos1)
            np.save(os.path.join(processed_data_dir, name_prefix + '_pos2.npy'), self.data_pos2)
            np.save(os.path.join(processed_data_dir, name_prefix + '_dpos2.npy'), self.data_dpos2)
            np.save(os.path.join(processed_data_dir, name_prefix + '_mask.npy'), self.data_mask)
            np.save(os.path.join(processed_data_dir, name_prefix + '_length.npy'), self.data_length)
            json.dump(self.rel2scope, open(os.path.join(processed_data_dir, name_prefix + '_rel2scope.json'), 'w'))
            np.save(os.path.join(processed_data_dir, word_vec_name_prefix + '_mat.npy'), self.word_vec_mat)
            json.dump(self.word2id, open(os.path.join(processed_data_dir, word_vec_name_prefix + '_word2id.json'), 'w'))
            print("Finish storing")
    

    
    
    def get_bert_dmask(self,data):
        fb = open('bert_tokens_'+self.name_prefix,'r')
        fr = open("dmask1_"+self.name_prefix,"r")
        line = fr.readline()
        line = eval(line)
        dmask1 = line
        fr.close()
        fr = open("dmask2_"+self.name_prefix,"r")
        line = fr.readline()
        line = eval(line)
        dmask2 = line    
        fr = open("dmask2id","r")
        line = fr.readline()
        line = eval(line)
        dmask2id = line
        fr.close() 
        bert_dmask1 = {}
        bert_dmask2 = {}
        for relation in data:
           #print(relation) 
           cur_re_dmask1 = dmask1[relation]
           cur_re_dmask2 = dmask2[relation]
           cur_re_bert_dmask1 = np.zeros((700, 100),dtype=np.int32)
           cur_re_bert_dmask2 = np.zeros((700, 100),dtype=np.int32)
           insnum = 0
           #print(relation)
           for ins in range(0,len(data[relation])):              
              bert_token = fb.readline()
              bert_token = eval(bert_token)
              #print(bert_token)
              #ipdb.set_trace()
              cur_ins_dmask1 = cur_re_dmask1[ins]
              cur_ins_dmask2 = cur_re_dmask2[ins]              
              cur_ins_word = data[relation][ins]['tokens']
              #print(cur_ins_word)
              #print(cur_ins_dmask1)
              i = 0
              #ipdb.set_trace()
              for w in range(1,len(bert_token)-1):
                  #print(w,i,cur_ins_dmask1[i])
                  #print(bert_token[w],cur_ins_word[i]) 
                  if (bert_token[w] == cur_ins_word[i])or w==1 :
                     cur_re_bert_dmask1[ins][w] = dmask2id[cur_ins_dmask1[i]]
                     cur_re_bert_dmask2[ins][w] = dmask2id[cur_ins_dmask2[i]]
                     if i+1 <len(cur_ins_word):
                           i += 1
                  else:
                     if bert_token[w-1] == cur_ins_word[i-1]:
                        cur_re_bert_dmask1[ins][w] = dmask2id[cur_ins_dmask1[i]]
                        cur_re_bert_dmask2[ins][w] = dmask2id[cur_ins_dmask2[i]]
                        if i+1 <len(cur_ins_word):
                           i += 1
                     else:
                      if (bert_token[w][0]==cur_ins_word[i][0]) :
                        cur_re_bert_dmask1[ins][w] = dmask2id[cur_ins_dmask1[i]]
                        cur_re_bert_dmask2[ins][w] = dmask2id[cur_ins_dmask2[i]] 
                        if i+1 <len(cur_ins_word):
                           i += 1
                      else:
                        cur_re_bert_dmask1[ins][w] = cur_re_bert_dmask1[ins][w-1]
                        cur_re_bert_dmask2[ins][w] = cur_re_bert_dmask2[ins][w-1]     
              #ipdb.set_trace()
           bert_dmask1[relation] = cur_re_bert_dmask1
           bert_dmask2[relation] = cur_re_bert_dmask2
        fb.close()
        
        return bert_dmask1,bert_dmask2
  
    def get_bert_pos_and_bseg(self,data):
        fb = open('bert_tokens_'+self.name_prefix,'r')
        i = 0
        for relation in data:
          #cur_re_ctree_tailword = self.get_ctree_segment(relation) 
          insnum = 0
          for ins in data[relation]:
            #cur_ins_tailword = cur_re_ctree_tailword[insnum]
            p1 = 0
            p2 = 0
            
            line = fb.readline()
            line = eval(line)
            #ipdb.set_trace()
            self.data_mask[i][:len(line)]=1
            #print (line)
            word = ins['tokens']
            #print (word)
            pos1 = ins['h'][2][0][0]
            pos1_= ins['h'][2][0][-1]
            pos2 = ins['t'][2][0][0]
            pos2_ = ins['t'][2][0][-1]
            new_pos1 = 0
            new_pos2 = 0
            #ipdb.set_trace()
            if word[pos1] in line:
              new_pos1 = line.index(word[pos1])
              line[new_pos1] = 'pad'
            if word[pos2] in line:
              new_pos2 = line.index(word[pos2]) 
              line[new_pos2] = 'pad'
            if (word[pos1] not in line) or (word[pos2] not in line):
              for n in range(0,len(line)):
                if line[n][0] == '#':
                  if p == 0:
                     p = n-1
                  line[n] = line[n-1]+line[n].replace('##','')
                  if line[n] == word[pos1]:
                    new_pos1 = p
                  if line[n] == word[pos2]:
                    new_pos2 = p                    
                else:
                  p = 0
            #ipdb.set_trace()
            ins_len = 100
            if new_pos1 >= ins_len :
               new_pos1 = ins_len - 1
            if new_pos2 >= ins_len:
               new_pos2 = ins_len - 1
            for j in range(ins_len):
               self.data_dpos1[i][j] = j - new_pos1 + 100
               self.data_dpos2[i][j] = j - new_pos2 + 100


 
            #ipdb.set_trace()
            i += 1
            insnum+=1
        fb.close()  
    def get_parse(self,relation) :
    #get tree:
        
        all_tree = []
        fp = open("./dtree/dtree"+relation,"r")
        tree = ['0']
        while len(all_tree) < 700:
              one_tree = []
              tree = fp.readline().replace('\n','').replace('\r','').replace('(','').replace(')','').replace('\'','').split(',')
              while tree != [''] :
                    
                    one_tree.append(tree)
                    tree = fp.readline().replace('\n','').replace('\r','').replace('(','').replace(')','').replace('\'','').split(',')
                    if tree == ['#+']:
                       break
                    
              all_tree.append(one_tree) 
        #
     #get sent:
        all_sent = []
        fp2 = open("./dsent/dsent"+relation,"r")
        sent = fp2.readline().replace('\n','').replace('\r','').split(' ') 
        while sent != [''] :
          all_sent.append(sent)
          sent = fp2.readline().replace('\n','').replace('\r','').split(' ')          
          #print sent

        fp2.close()
        #
        #print len(all_tree),len(all_sent)
        return all_tree,all_sent
        
    
    def get_mask(self,relation):
        all_new_pos1 = []
        all_new_pos2 = []
        all_new_pos1_ = []
        all_new_pos2_ = []
        for ins in self.ori_data[relation]:
            pos1 = ins['h'][2][0][0]
            pos1_= ins['h'][2][0][-1]
            pos2 = ins['t'][2][0][0]
            pos2_ = ins['t'][2][0][-1]
            if pos2_ < pos1 :
               new_pos2 = pos2
               new_pos2_ = pos2_
               new_pos1 = pos1 - pos2_ + pos2
               new_pos1_ = pos1_ - pos2_ + pos2   
            if pos1_ < pos2 :
               new_pos1 = pos1
               new_pos1_ = pos1_
               new_pos2 = pos2 - pos1_ + pos1
               new_pos2_ = pos2_ - pos1_ + pos1
            all_new_pos1.append(new_pos1)
            all_new_pos2.append(new_pos2)
            all_new_pos1_.append(new_pos1_)
            all_new_pos2_.append(new_pos2_)           
        all_tree,all_sent = self.get_parse(relation)
        
        all_mask_e1 = []
        all_mask_e2 = []
#===========================mask e1=======================================================#
        for s in range(0,len(all_sent)):

          one_mask_e1 = []
          
          for w in range(0,len(all_sent[s])):
            #ipdb.set_trace()
            for t in range(0,len(all_tree[s])):
              tag = None
              left_word_pos = int(all_tree[s][t][1].replace(' ',''))-1
              right_word_pos = int(all_tree[s][t][2].replace(' ',''))-1
              ne1 = all_new_pos1_[s]-all_new_pos1[s]
              ne2 = all_new_pos2_[s]-all_new_pos2[s]
              if (left_word_pos== w and right_word_pos == all_new_pos1[s]) or (right_word_pos == w and left_word_pos == all_new_pos1[s]):
                tag = all_tree[s][t][0]
                break
            if tag :
              one_mask_e1.append(tag)
              if w == all_new_pos2[s]:
                for i in range(0,ne2): 
                  one_mask_e1.append(tag)
            else:
              if w == all_new_pos1[s]:
                for i in range(0,ne1+1): 
                  one_mask_e1.append('self') 
              else:
                one_mask_e1.append('others')
                if w == all_new_pos2[s]:
                   for i in range(0,ne2): 
                      one_mask_e1.append('others')
          #print all_sent[s]
          #print one_mask_e1
                   
          all_mask_e1.append(one_mask_e1)      
#===========================mask e2=======================================================#
        for s in range(0,len(all_sent)):
          one_mask_e2 = []
          #__import__("ipdb").set_trace()
          for w in range(0,len(all_sent[s])):
            for t in range(0,len(all_tree[s])):
              tag = None
              left_word_pos = int(all_tree[s][t][1].replace(' ',''))-1
              right_word_pos = int(all_tree[s][t][2].replace(' ',''))-1
              ne1 = all_new_pos1_[s]-all_new_pos1[s]
              ne2 = all_new_pos2_[s]-all_new_pos2[s]
              if (left_word_pos== w and right_word_pos == all_new_pos2[s]) or (right_word_pos == w and left_word_pos == all_new_pos2[s]):
                tag = all_tree[s][t][0]
                break
            if tag :
              one_mask_e2.append(tag)
              if w == all_new_pos1[s]:
                for i in range(0,ne1): 
                  one_mask_e2.append(tag)
            else:
              if w == all_new_pos2[s]:
                for i in range(0,ne2+1): 
                  one_mask_e2.append('self') 
              else:
                one_mask_e2.append('others')
                if w == all_new_pos1[s]:
                   for i in range(0,ne1): 
                      one_mask_e2.append('others')
               
          #print all_sent[s]
          #print one_mask_e2
          all_mask_e2.append(one_mask_e2)    
          #__import__("ipdb").set_trace()
        return all_mask_e1,all_mask_e2
          
        
        
                   
    def get_tree(self,ins):
        head = ins['h'][0]
        tail = ins['t'][0]
        pos1 = ins['h'][2][0][0]
        pos1_= ins['h'][2][0][-1]
        pos2 = ins['t'][2][0][0]
        pos2_ = ins['t'][2][0][-1]
        words = ins['tokens']
        new_e1 = words[pos1]
        new_e2 = words[pos2]
        dtree = []
        #print (words)
        #__import__("ipdb").set_trace()
        if(pos1 != pos1_):
            i= pos1+1
            while i<= pos1_:
               new_e1 = new_e1+'aaa'+words[i]
               i +=1  
            words[pos1]=new_e1    

        if(pos2 != pos2_):
            x = pos2+1
            while x<= pos2_:
                new_e2 = new_e2+'aaa'+words[x]
                x +=1     
            words[pos2]=new_e2
        #print (words)
        if pos2_ < pos1 :
            if  pos1_+1 < len(words) :
               words = words[:pos2+1]+words[pos2_+1: pos1+1]+words[pos1_+1:]
            else :
               words = words[:pos2+1]+words[pos2_+1: pos1+1]
             
        if pos1_ < pos2 :
            if  pos1_+1 < len(words) :
                words =words[:pos1+1]+words[pos1_+1: pos2+1]+words[pos2_+1:]
            else:
                words =words[:pos1+1]+words[pos1_+1: pos2+1]

        #print (words)
                
        sent = " ".join(words).replace('.','')
        dtree = nlp.dependency_parse(sent)
        return sent,dtree
         
        #doc.sentences[0].print_constituencies()
    
    def next_one_nobert(self, N, K, Q, noise_rate=0):
        target_classes = random.sample(self.rel2scope.keys(), N)
        
        
        noise_classes = []
        for class_name in self.rel2scope.keys():
            if not (class_name in target_classes):
                noise_classes.append(class_name)
        support_set = {'word': [],'sent':[],'dmask1':[], 'dmask2':[], 'dpos1': [],'dpos2': [], 'mask': []}
        query_set = {'word': [],'sent':[],'dmask1':[], 'dmask2':[], 'dpos1': [],'dpos2': [], 'mask': []}
        query_label = []
        #ipdb.set_trace()
        for i, class_name in enumerate(target_classes):
            scope = self.rel2scope[class_name]
            indices = np.random.choice(list(range(scope[0], scope[1])), K + Q, False)
            word = self.data_word[indices]
            #ipdb.set_trace()
            sent = np.array(self.data_sent)[indices]
            sent = list(sent)
            dmask1 = self.data_dmask1[indices]
            dmask2 = self.data_dmask2[indices]
            dpos1 = self.data_dpos1[indices]
            dpos2 = self.data_dpos2[indices]
            mask = self.data_mask[indices]
            support_word, query_word, _ = np.split(word, [K, K + Q])
            support_sent, query_sent, _ = np.split(sent, [K, K + Q])
            support_dmask1, query_dmask1, _ = np.split(dmask1,[K, K + Q])
            support_dmask2, query_dmask2, _ = np.split(dmask2,[K, K + Q])
            support_dpos1, query_dpos1, _ = np.split(dpos1, [K, K + Q])
            support_dpos2, query_dpos2, _ = np.split(dpos2, [K, K + Q])
            support_mask, query_mask, _ = np.split(mask, [K, K + Q])
            
            for j in range(K):
                prob = np.random.rand()
                if prob < noise_rate:
                    noise_class_name = noise_classes[np.random.randint(0, len(noise_classes))]
                    scope = self.rel2scope[noise_class_name]
                    indices = np.random.choice(list(range(scope[0], scope[1])), 1, False)
                    word = self.data_word[indices]
                    sent = self.data_sent[indices]
                    dmask1 = self.data_dmask1[indices]
                    dmask2 = self.data_dmask2[indices]
                    dpos1 = self.data_dpos1[indices]
                    dpos2 = self.data_dpos2[indices]
                    mask = self.data_mask[indices]
                    support_word[j] = word
                    support_sent[j] = sent
                    support_dmask1[j] = dmask1
                    support_dmask2[j] = dmask2
                    support_dpos1[j] = dpos1
                    support_dpos2[j] = dpos2
                    support_mask[j] = mask

            support_set['word'].append(support_word)
            support_set['sent'].append(support_sent)
            support_set['dmask1'].append(support_dmask1)
            support_set['dmask2'].append(support_dmask2)
            support_set['dpos1'].append(support_dpos1)
            support_set['dpos2'].append(support_dpos2)
            support_set['mask'].append(support_mask)
            query_set['word'].append(query_word)
            query_set['sent'].append(query_sent)
            query_set['dmask1'].append(query_dmask1)
            query_set['dmask2'].append(query_dmask2)
            query_set['dpos1'].append(query_dpos1)
            query_set['dpos2'].append(query_dpos2)
            query_set['mask'].append(query_mask)
            query_label += [i] * Q
        #__import__("ipdb").set_trace()
        support_set['word'] = np.stack(support_set['word'], 0)
        support_set['sent'] = np.stack(support_set['sent'], 0)
        support_set['dmask1'] = np.stack(support_set['dmask1'],0)
        support_set['dmask2'] = np.stack(support_set['dmask2'],0)
        support_set['dpos1'] = np.stack(support_set['dpos1'], 0)
        support_set['dpos2'] = np.stack(support_set['dpos2'], 0)
        support_set['mask'] = np.stack(support_set['mask'], 0)
        query_set['word'] = np.concatenate(query_set['word'], 0)
        query_set['sent'] = np.concatenate(query_set['sent'], 0)
        query_set['dmask1'] = np.concatenate(query_set['dmask1'], 0)
        query_set['dmask2'] = np.concatenate(query_set['dmask2'], 0)
        query_set['dpos1'] = np.concatenate(query_set['dpos1'], 0)
        query_set['dpos2'] = np.concatenate(query_set['dpos2'], 0)
        query_set['mask'] = np.concatenate(query_set['mask'], 0)
        query_label = np.array(query_label)
        #print query_label
        #print "===================================="

        perm = np.random.permutation(N * Q)
        query_set['word'] = query_set['word'][perm]
        query_set['sent'] = query_set['sent'][perm]
        query_set['dmask1'] = query_set['dmask1'][perm]
        query_set['dmask2'] = query_set['dmask2'][perm]
        query_set['dpos1'] = query_set['dpos1'][perm]
        query_set['dpos2'] = query_set['dpos2'][perm]
        query_set['mask'] = query_set['mask'][perm] 
        query_label = query_label[perm]
        return support_set, query_set, query_label,target_classes

    def next_batch_nobert(self, B, N, K, Q, noise_rate=0):
        support = {'word': [],'sent': [], 'dmask1':[], 'dmask2':[],'dpos1': [], 'dpos2': [], 'mask': []}
        query = {'word': [],'sent': [], 'dmask1':[], 'dmask2':[], 'dpos1': [], 'dpos2': [], 'mask': []}
        label = []
        batch_target = []
        for one_sample in range(B):
            current_support, current_query, current_label ,target_classes= self.next_one_nobert(N, K, Q, noise_rate=noise_rate)
            
            support['word'].append(current_support['word'])
            support['sent'].append(current_support['sent'])
            support['dmask1'].append(current_support['dmask1'])
            support['dmask2'].append(current_support['dmask2']) 
            #support['tag'].append(current_support['tag'])
            support['dpos1'].append(current_support['dpos1'])
            support['dpos2'].append(current_support['dpos2'])
            support['mask'].append(current_support['mask'])
            query['word'].append(current_query['word'])
            query['sent'].append(current_query['sent'])
            query['dmask1'].append(current_query['dmask1'])
            query['dmask2'].append(current_query['dmask2'])
            #query['tag'].append(current_query['tag'])
            query['dpos1'].append(current_query['dpos1'])
            query['dpos2'].append(current_query['dpos2'])
            query['mask'].append(current_query['mask'])
            label.append(current_label)
            batch_target.append(target_classes)
            #batch_target.append(target_classes)
        support['word'] = Variable(torch.from_numpy(np.stack(support['word'], 0)).long().view(-1, self.max_length))
        support['dmask1'] = Variable(torch.from_numpy(np.stack(support['dmask1'], 0)).long().view(-1, 100))
        support['dmask2'] = Variable(torch.from_numpy(np.stack(support['dmask2'], 0)).long().view(-1, 100)) 
        #support['tag'] = Variable(torch.from_numpy(np.stack(support['tag'], 0)).long().view(-1, self.max_length))   
        support['dpos1'] = Variable(torch.from_numpy(np.stack(support['dpos1'], 0)).long().view(-1, 100)) 
        support['dpos2'] = Variable(torch.from_numpy(np.stack(support['dpos2'], 0)).long().view(-1, 100)) 
        support['mask'] = Variable(torch.from_numpy(np.stack(support['mask'], 0)).long().view(-1, 100)) 
        query['word'] = Variable(torch.from_numpy(np.stack(query['word'], 0)).long().view(-1, self.max_length))
        query['dmask1'] = Variable(torch.from_numpy(np.stack(query['dmask1'], 0)).long().view(-1, 100))
        query['dmask2'] = Variable(torch.from_numpy(np.stack(query['dmask2'], 0)).long().view(-1, 100))
        #query['tag'] = Variable(torch.from_numpy(np.stack(query['tag'], 0)).long().view(-1, self.max_length))
        query['dpos1'] = Variable(torch.from_numpy(np.stack(query['dpos1'], 0)).long().view(-1, 100)) 
        query['dpos2'] = Variable(torch.from_numpy(np.stack(query['dpos2'], 0)).long().view(-1, 100))         
        query['mask'] = Variable(torch.from_numpy(np.stack(query['mask'], 0)).long().view(-1, 100)) 
        label = Variable(torch.from_numpy(np.stack(label, 0).astype(np.int64)).long())
        #ipdb.set_trace()
        # To cuda
        if self.cuda:
            for key in support:
              if key != 'sent':
                support[key] = support[key].cuda()
            for key in query:
              if key != 'sent':
                query[key] = query[key].cuda()
            label = label.cuda()
        return support, query, label,batch_target

