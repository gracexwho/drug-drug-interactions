#!/usr/bin/env python
# coding: utf-8

# In[ ]:


## Deepwalk only takes in INTEGERS as nodes


from comet_ml import Experiment

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from random import choice
import urllib.request  # the lib that handles the url stuff
import time

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import zero_one_loss
from itertools import *


torch.cuda.empty_cache()

url = "https://raw.githubusercontent.com/gracexwho/drug-drug-interactions/master/ChCh-Miner_durgbank-chem-chem.tsv"
url_data = urllib.request.urlopen(url) 

G = nx.read_edgelist(url_data)

print(G.number_of_nodes())
print(G.number_of_edges())


# Create an experiment
experiment = Experiment(api_key="yeThLw8MLFuaMF3cVW1b9IsIt",
                        project_name="Node2Vec", workspace="gracexwho")

# Report any information you need by:


# In[ ]:




################# CONTROL ##################

hyper_params = {"learning_rate": 0.05, "epochs": 5000, "num_walks": 500, "walk_length": 5, "window_size": 3}
experiment.log_parameters(hyper_params)

#hyper_params = {"learning_rate": 0.1, "epochs": 5000, "num_walks": 100, "walk_length": 5, "window_size": 3}
#experiment.log_parameters(hyper_params)


################# CONTROL ##################


# In[ ]:


import random
## Always "Restart Kernel and Clear Output if you're going to train again"


id_map = {}
edges = list(G.edges())
#print(edges[0])



def get_id(id_string):
    if id_string in id_map.keys():
        return id_map[id_string]
    else:
        ID = len(id_map)
        id_map[id_string] = ID
        return ID
    
edge_numbers = [(get_id(e[0]), get_id(e[1])) for e in edges]
#print(edge_numbers)


D = nx.Graph()
D.add_edges_from(edge_numbers)
print(D.number_of_edges())


### This generates the edgelist for DeepWalk
nx.write_edgelist(D, 'dw.edgelist')


# In[ ]:


# Generate random walks

import random
## Always "Restart Kernel and Clear Output if you're going to train again"

pairs = []

for i in range(hyper_params['num_walks']):
    current = choice(list(G.nodes()))
    walk = [current]
    y = []
    
    for w in range(hyper_params['walk_length']):
        # walk to an adjacent node
        # error: some adjacent nodes are NOT IN the training set
        c = list(G.adj[current])
        current = choice(c)
        walk.append(current)
    
    # take permutations as closely related within the window size
    y = [permutations(walk[i : i+hyper_params['window_size']], 2) for i in range(len(walk)-hyper_params['window_size'])]
    z = []
    for l in y:
        z.extend(list(l))
    pairs.extend(z)

# remove duplicates
pairs = list(dict.fromkeys(pairs))


class Encoder(nn.Module):
  # should return VECTORS for each node
    def __init__(self):
        super(Encoder, self).__init__()
        
        #self.dropout = nn.Dropout(p=0.2)
        # one layer, return embeds(node_ids) which is a long tensor
        #learnrate might be too big if doesn't decrease
        self.embed = nn.Embedding(G.number_of_nodes(), 64)

    
    def forward(self, x):
        # take the node name as input and 
        x = self.embed(x)
        return x


def process_batch(batch):
    left_ids = torch.cuda.LongTensor([get_id(pair[0]) for pair in batch])
    right_ids = torch.cuda.LongTensor([get_id(pair[1]) for pair in batch])
    neg_ids = torch.cuda.LongTensor([np.random.randint(0, G.number_of_nodes()) for _ in batch])
    
    #print(left_ids)
    left_embeds = model(left_ids)
    right_embeds = model(right_ids)
    neg_embeds = model(neg_ids)
    
    pos_score = torch.mm(torch.t(left_embeds), right_embeds)
    neg_score = torch.mm(torch.t(left_embeds), neg_embeds)
    
    loss = get_loss(pos_score, neg_score)
    return loss
    
                          
def get_loss(pos, neg):
    m = nn.Sigmoid()
    loss = -torch.mean(torch.log(m(pos))) - torch.mean(torch.log(1 - m(neg)))
    return loss



def common_neighbours(T, u, v):
    Unodes = list(T.adj[u])
    Vnodes = list(T.adj[v])
    matches = [x for x in Unodes if x in Vnodes]
    return iter(matches)

def jaccard_index(T, u, v):
    common = len(list(common_neighbours(T, u, v)))
    union = T.degree[u] + T.degree[v] - common
    if union == 0:
        return 0
    else:
        return (common/union)

def adamic_adar(T, u, v):
    common = common_neighbours(T, u, v)
    total = 0
    for c in common:
        total = total + 1/np.log(T.degree[c])
    return total

def CN(T, edges):
    CN = {}
    for (u,v) in edges:
        CN[(u,v)] = len(list(common_neighbours(T,u,v)))
    return CN.values()


def JC(T, edges):
    JC = {}
    for (u,v) in edges:
        JC[(u,v)] = jaccard_index(T, u, v)
    return JC.values()


def AA(T, edges):
    AA = {}
    for (u,v) in edges:
        AA[(u,v)] = adamic_adar(T, u, v)
    return AA.values()


# In[ ]:


# Split graph into training and validation set
import random


url = "https://raw.githubusercontent.com/gracexwho/drug-drug-interactions/master/ChCh-Miner_durgbank-chem-chem.tsv"
url_data = urllib.request.urlopen(url) 
T = nx.read_edgelist(url_data)
V = nx.Graph()

num_val = 20000
num_neg = 20000


val_set = random.sample(list(T.edges()), num_val)
T.remove_edges_from(val_set)
V.add_edges_from(val_set)

T.remove_nodes_from(list(nx.isolates(T)))
# this removes nodes that don't have any neighbors from training graph

num_train = num_neg + T.number_of_edges()


print(T.number_of_edges())
print(T.number_of_nodes())

print("Number of validation edges", num_val)
print("Number of training edges", num_train)


# In[ ]:


model = Encoder()
model.embed.weight.data = (model.embed.weight.data/np.sqrt(64))
model.to('cuda')

optimizer = optim.SGD(model.parameters(), lr=hyper_params['learning_rate'])

epochs = hyper_params['epochs']

#for n in list(list(G.nodes())):
#    get_id(n)

#pairs = [(get_id(pair[0]) , get_id(pair[1])) for pair in pairs]


alpha = 0.9
train_loss = 0
ep = hyper_params['epochs']

for e in range(ep):
    random.shuffle(pairs)
    train_loss = 0
    batch_size = 64
    batch = []
    index=0
    # the reason you can't use training_loss for optimizer is because train_loss isn't defined within the while loop
    # try doing train_loss.back() OUTSIDE of while loop?
    
    while index+batch_size < len(pairs):
        batch = pairs[index:min(index+batch_size, len(pairs))]
        index += batch_size
        
        #print(batch)
        loss = process_batch(batch)
        train_loss = alpha * train_loss + (1-alpha)*loss
        
    optimizer.zero_grad() 
    train_loss.backward()        #change back to loss if retain_graph error
    optimizer.step()
        #print("loss", loss)
        
    if e % 500 == 0:
        print("Training loss: ", train_loss)
        

      
torch.save(model.state_dict(), 'node2vec.pth')
    
print("done")


# In[ ]:


import collections
import subprocess


### deepwalk --format edgelist --input hamilton-lab/dw.edgelist --output dw.embeddings
# This is the code to run deepwalk and generate the embeddings


args = ["deepwalk "]
args.append("--format edgelist ")
args.append("--input hamilton-lab/dw.edgelist ")
args.append("--output dw.embeddings")
        
string =""
for x in args:
    string+=x
subprocess.call(string,shell=True)


############################################################################Now we read the embeddings into python

embeds = pd.read_csv('dw.embeddings', sep=" ", header=None)
embeds.head()
#torch_tensor = torch.tensor(targets_df['targets'].values)

e_parse = embeds.iloc[:, 0:]
e_parse.head()

e_dict = {}


#So basically deepwalk took the edgelist (integers)

for index, row in e_parse.iterrows():
    e_dict[int(row.values[0])] = row.values[1:]


e_dict = collections.OrderedDict(sorted(e_dict.items()))

e_array = np.array(list(e_dict.values()))
e_torch = torch.cuda.FloatTensor(e_array)
print(e_torch.shape)


# In[ ]:


##################### TRAINING LOOP ######################
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import zero_one_loss


start_time = time.time()


class LogisticRegression(torch.nn.Module):
     
    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(64, 1)
        #self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred

    
link_model = LogisticRegression()
criterion = torch.nn.BCELoss(reduction='mean')         # this is already binary cross entropy!
optimizer_link = torch.optim.SGD(link_model.parameters(), lr = 0.05) 
weight = e_torch

train_edges = list(T.edges())
val_edges = list(V.edges())

y_label = [1]*len(train_edges)
neg = [0]*num_neg
y_label = y_label + neg
y_label = np.array(y_label)


negs = random.sample(list(nx.non_edges(T)), num_neg)
train_edges = train_edges + negs


left = [pair[0] for pair in train_edges]
right = [pair[1] for pair in train_edges]
left = [get_id(l) for l in left]
right = [get_id(r) for r in right]

left_ids = [weight[ids] for ids in left]
lf = torch.stack(left_ids)
right_ids = [weight[ids] for ids in right]
rg = torch.stack(right_ids)

#weight = torch.FloatTensor(np.random.rand(weight.shape[0], weight.shape[1]))

dot_prod = torch.bmm(lf.view(num_train, 1, 64), rg.view(num_train, 64, 1)) 
dot_prod = torch.squeeze(dot_prod)
dot_prod = dot_prod.cpu()
dotproduct = dot_prod.numpy()

score = roc_auc_score(y_label, dotproduct)
print("The auc score for dot product is: ", score)




link_model.to('cuda')


element_mul = (lf * rg)
#element_mul = (lf + rg)
element_mul = element_mul.cuda()
y_label_model = torch.cuda.FloatTensor(y_label)

print(element_mul[0])

link_model.train()


# Keep it at 5000, more is overfitting

ra = 10000
for e in range(ra):
    optimizer_link.zero_grad() 
    y_pred = link_model(element_mul)
    y_pred = torch.squeeze(y_pred)
    loss = criterion(y_pred, y_label_model)
    loss.backward() 
    optimizer_link.step() 
    auc = roc_auc_score(y_label, y_pred.cpu().detach().numpy())
    if e % 2000 == 0:
        print("AUC for DW: ", auc)
        print(loss)
    
print("Here's the graph index value:")
test = list(AA(T, train_edges))
print(roc_auc_score(y_label, test))
    
torch.save(link_model.state_dict(), 'linkpred.pth')
        
print("--- %s minutes ---" % ((time.time() - start_time)//60))
    

# If you get CUDA error, just keep running again and it'll work


# In[ ]:


link_model2 = LogisticRegression()
criterion = torch.nn.BCELoss(reduction='mean')         # this is already binary cross entropy!
optimizer_link2 = torch.optim.SGD(link_model2.parameters(), lr = 0.05) 
nv_weight = model.embed.weight.data * 10


left = [pair[0] for pair in train_edges]
right = [pair[1] for pair in train_edges]
left = [get_id(l) for l in left]
right = [get_id(r) for r in right]
left_ids = [nv_weight[ids] for ids in left]
lf = torch.stack(left_ids)
right_ids = [nv_weight[ids] for ids in right]
rg = torch.stack(right_ids)


#weight = torch.FloatTensor(np.random.rand(weight.shape[0], weight.shape[1]))

dot_prod = torch.bmm(lf.view(num_train, 1, 64), rg.view(num_train, 64, 1)) 
dot_prod = torch.squeeze(dot_prod)
dot_prod = dot_prod.cpu()
dotproduct = dot_prod.numpy()

score = roc_auc_score(y_label, dotproduct)
print("The auc score for dot product is: ", score)


link_model2.to('cuda')


element_mul = (lf * rg)
#element_mul = (lf + rg)
element_mul = element_mul.cuda()
y_label_model = torch.cuda.FloatTensor(y_label)

print(element_mul[0])

link_model.train()


# Keep it at 5000, more is overfitting

ra = 10000
for e in range(ra):
    optimizer_link2.zero_grad() 
    y_pred = link_model2(element_mul)
    y_pred = torch.squeeze(y_pred)
    loss = criterion(y_pred, y_label_model)
    loss.backward() 
    optimizer_link2.step() 
    auc = roc_auc_score(y_label, y_pred.cpu().detach().numpy())
    if e % 2000 == 0:
        print("N2V AUC:", auc)
        print(loss)
    
print("Here's the graph index value:")
test = list(AA(T, train_edges))
print(roc_auc_score(y_label, test))
    
torch.save(link_model2.state_dict(), 'linkpred2.pth')


# In[ ]:


########### VALIDATION LOOP ##############
y_hat = []
y_true = [1]*len(val_edges)
neg = [0]*num_val
y_true = y_true + neg
y_true = np.array(y_true)

y_hat2 = []
y_true2 = [1]*len(val_edges)
neg2 = [0]*num_val
y_true2 = y_true2 + neg
y_true2 = np.array(y_true)

negs = random.sample(list(nx.non_edges(V)), num_val)
val_edges = val_edges + negs


link_model.eval()
link_model2.eval()

for (x,y) in val_edges:
    u = weight[get_id(x)]
    v = weight[get_id(y)]
    a = nv_weight[get_id(x)]
    b = nv_weight[get_id(y)]
    em = u * v
    em = em.cuda()
    em2 = a * b
    em2 = em.cuda()
    pred = link_model(em)
    pred2 = link_model2(em2)
    y_hat.append(pred.item())
    y_hat2.append(pred2.item())

print("done")


# In[ ]:


accuracy = roc_auc_score(y_true, y_hat)
accuracy2 = roc_auc_score(y_true2, y_hat2)

jaccard = list(JC(V, val_edges))
jaccard2 = roc_auc_score(y_true, jaccard)

adamic = list(AA(V, val_edges))
adamic2 = roc_auc_score(y_true, adamic)

common = list(CN(V, val_edges))
common2 = roc_auc_score(y_true, common)


print("The auc score for DW elementwise mul: ", accuracy)
print("The AUC for node2vec EM is: ", accuracy2)
print("The auc score for JC index: ", jaccard2)
print("The auc score for AA index: ", adamic2)
print("The auc score for CN index: ", common2)


# In[ ]:


[fpr, tpr, thresh] = roc_curve(y_true, common)

ratioMax = tpr - fpr

fig1 = plt.figure()
common_line, = plt.plot(fpr, tpr, label='CN')

fig1.suptitle('AUC of Local Similarity Indices', fontsize=20)
plt.xlabel('FPR', fontsize=18)
plt.ylabel('TPR', fontsize=16)
plt.axis([0, 1.0, 0, 1.0])                      # set [xmin, xmax, ymin, ymax]


[fpr2, tpr2, thresh2] = roc_curve(y_true, jaccard)
ratioMax = tpr2 - fpr2
jaccard_line, = plt.plot(fpr2, tpr2, label='JC')


[fpr3, tpr3, thresh3] = roc_curve(y_true, adamic)
ratioMax = tpr3 - fpr3
adamic_line, = plt.plot(fpr3, tpr3, label='AA')


plt.legend(handles=[common_line, jaccard_line, adamic_line], loc='center right')


fig1.savefig('Node2Vec_Indices.jpg')

