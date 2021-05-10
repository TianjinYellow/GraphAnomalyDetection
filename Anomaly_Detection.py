#import metis
import torch
import random
import numpy as np
import networkx as nx
from sklearn.model_selection import train_test_split
import collections
from scipy.stats import sem
from sklearn.metrics import accuracy_score
import sklearn
from ssl_utils import encode_onehot
from sklearn_extra.cluster import KMedoids
#from utils import row_normalize
import numpy as np
import numba
from numba import njit
import os
import time

import scipy
import torch.nn as nn
import scipy.sparse as sp
import torch.nn.functional as F
import numpy as np
import torch
import networkx as nx
from sklearn.cluster import KMeans
import torch.nn as nn
import torch.nn.functional as F
import math
import torch
import torch.optim as optim
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
#from deeprobust.graph import utils
from copy import deepcopy
from sklearn.metrics import f1_score
import os
from input import *
import copy
from random import shuffle
import random
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score,precision_score,recall_score,f1_score
from sklearn.metrics import roc_curve
import matplotlib
from matplotlib import pyplot
import argparse
matplotlib.use("Agg")


class NodeDistance:

    def __init__(self, adj,adj_ori, nclass=4):
        """
        :param graph: Networkx Graph.
        """
        self.adj = adj
        self.adj_ori=adj_ori
        self.graph = nx.from_scipy_sparse_matrix(adj)
        self.nclass = nclass

    def get_label(self):
        path_length = dict(nx.all_pairs_shortest_path_length(self.graph, cutoff=self.nclass-1))
        distance = - np.ones((len(self.graph), len(self.graph))).astype(int)

        for u, p in path_length.items():
            for v, d in p.items():
                distance[u][v] = d

        distance[distance==-1] = distance.max() + 1
        distance = np.triu(distance)
        self.distance = distance
        return torch.LongTensor(distance) - 1

    def _get_label(self):
        '''
        group 1,2 into the same category, 3, 4, 5 separately
        designed for 2-layer GCN
        '''
        path_length = dict(nx.all_pairs_shortest_path_length(self.graph))
        distance = - np.ones((len(self.graph), len(self.graph))).astype(int)

        for u, p in path_length.items():
            for v, d in p.items():
                distance[u][v] = d

        distance[distance==-1] = distance.max() + 1

        # group 1, 2 in to one category
        distance = np.triu(distance)
        #distance[distance==1] = 2
        self.distance = distance - 1
        return torch.LongTensor(distance) - 2

    def sample(self, labels, ratio=0.1):
        # first sample k nodes
        # candidates = self.all
        candidates = np.arange(len(self.graph))
        perm = np.random.choice(candidates, int(ratio*len(candidates)), replace=False)
        # then sample k other nodes to make sure class balance
        node_pairs = []
        for i in range(1, labels.max()+1):
            tmp = np.array(np.where(labels==i)).transpose()
            indices = np.random.choice(np.arange(len(tmp)), 10, replace=False)
            node_pairs.append(tmp[indices])
        node_pairs = np.array(node_pairs).reshape(-1, 2).transpose()
        return node_pairs[0], node_pairs[1]

class Base:

    def __init__(self, adj, features, device):
        self.adj = adj
        self.features = features.to(device)
        self.device = device
        self.cached_adj_norm = None

    def get_adj_norm(self):
        if self.cached_adj_norm is None:
            adj_norm = preprocess_adj(self.adj, self.device)
            self.cached_adj_norm= adj_norm
        return self.cached_adj_norm

    def make_loss(self, embeddings):
        return 0

    def transform_data(self):
        return self.get_adj_norm(), self.features
class PairwiseDistance(Base):

    def __init__(self, adj,adj_ori, features, nhid, device,  regression=False):
        self.adj = adj
        self.adj_ori=adj_ori
        self.features = features.to(device)
        self.nfeat = features.shape[1]
        self.cached_adj_norm = None
        self.device = device

        #self.labeled = idx_train.cpu().numpy()
        self.all = np.arange(adj.shape[0])
        #self.unlabeled = np.array([n for n in self.all if n not in idx_train])

        self.regression = regression
        self.nclass = args.C
        self.classifier=nn.Sequential(
            nn.Linear(nhid,nhid*2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(nhid*2,self.nclass),
            ).to(device)

        self.pseudo_labels = None

        self.adj_csc=sp.csc_matrix(self.adj_ori)
        self.adj_csc.eliminate_zeros()
        self.adj_coo=self.adj_csc.tocoo()

        self.row=np.array(self.adj_coo.row)
        self.col=np.array(self.adj_coo.col)
        

    def transform_data(self):
        return self.get_adj_norm(), self.features

    def make_loss(self, embeddings):
        if self.regression:
            return self.regression_loss(embeddings)
        else:
            return self.classification_loss(embeddings)
    def evaluate(self,embeddings,labels):

        embeddings0=embeddings[self.row]
        embeddings1=embeddings[self.col]
        #concat=torch.cat([embeddings0,embeddings1],dim=1)
        concat=torch.abs(embeddings0-embeddings1)
        embed=self.classifier(concat)
        out=F.softmax(embed,dim=1)


        labels1=torch.zeros(out.shape[0]).to(self.device)

        out=torch.argmax(out,dim=1)
        #temp=np.zeros(self.adj_ori.shape)
        temp=sp.coo_matrix((out.detach().cpu().numpy(),(self.row,self.col)),shape=self.adj_ori.shape)
        temp=temp.toarray()

        temp_sum=np.sum(temp,axis=1)
        adj_sum=np.sum(np.array(self.adj_ori),axis=1)
        temp_average=temp_sum/(adj_sum+1e-12)
        return temp_average,out.detach().cpu().numpy()





    def classification_loss(self, embeddings):
        if self.pseudo_labels is None:
            self.agent = NodeDistance(self.adj, self.adj_ori,nclass=self.nclass)
            self.pseudo_labels = self.agent.get_label().to(self.device)
            print("max label",torch.max(self.pseudo_labels))
            print("min label",torch.min(self.pseudo_labels))


        # embeddings = F.dropout(embeddings, 0, training=True)
        self.node_pairs = self.sample(self.agent.distance,ratio=args.S)
        node_pairs = self.node_pairs
        embeddings0 = embeddings[node_pairs[0]]
        embeddings1 = embeddings[node_pairs[1]]
        #concat=torch.cat([embeddings0,embeddings1],dim=1)
        concat=torch.abs(embeddings0-embeddings1)
        embeddings = self.classifier(concat)

        output = F.softmax(embeddings, dim=1)
        loss=nn.CrossEntropyLoss()(embeddings,self.pseudo_labels[node_pairs])

        #from metric import accuracy
        #temp_l=self.pseudo_labels[node_pairs]
        #acc = accuracy(output, self.pseudo_labels[node_pairs])
        #acc_1=accuracy(output[temp_l==0],temp_l[temp_l==0])

        return loss

    def sample(self, labels, ratio=0.1, k=600):
        k=1e10
        for i in range(1,labels.max()+1):
            temp=np.array(np.where(labels==i)).transpose()
            if k>len(temp):
                k=len(temp)

        k=int(k*ratio)

        node_pairs = []
        for i in range(1, labels.max()+1):
            tmp = np.array(np.where(labels==i)).transpose()
            indices = np.random.choice(np.arange(len(tmp)), k, replace=False)
            node_pairs.append(tmp[indices])
        node_pairs = np.array(node_pairs).reshape(-1, 2).transpose()
        return node_pairs[0], node_pairs[1]


def preprocess_features_normalize(features):
    """
    Row-normalize feature matrix and convert to tuple representation
    """
    rowsum = np.array(features.sum(1)) # get sum of each row, [2708, 1]
    r_inv = np.power(rowsum, -1).flatten() # 1/rowsum, [2708]
    r_inv[np.isinf(r_inv)] = 0. # zero inf data
    #r_mat_inv = sp.diags(r_inv) # sparse diagonal matrix, [2708, 2708]
    r_mat_inv=np.diag(r_inv)
    #features = r_mat_inv.dot(features) # D^-1:[2708, 2708]@X:[2708, 2708]
    features=np.matmul(r_mat_inv,features)
    return features # [coordinates, data, shape], []
def preprocess_features(features, device):
    return features.to(device)

def preprocess_adj(adj, device):
    # adj_normalizer = fetch_normalization(normalization)
    adj_normalizer = aug_normalized_adjacency
    r_adj = adj_normalizer(adj)
    r_adj = sparse_mx_to_torch_sparse_tensor(r_adj).float()
    r_adj = r_adj.to(device)
    return r_adj

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def aug_normalized_adjacency(adj):
   adj = adj + sp.eye(adj.shape[0])
   adj = sp.coo_matrix(adj)
   row_sum = np.array(adj.sum(1))
   d_inv_sqrt = np.power(row_sum, -0.5).flatten()
   d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
   d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
   return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()






class GraphConvolution(Module):
    """Simple GCN layer, similar to https://github.com/tkipf/pygcn
    """

    def __init__(self, in_features, out_features, with_bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if with_bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # self.weight.data.fill_(1)
        # if self.bias is not None:
        #     self.bias.data.fill_(1)

        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        """ Graph Convolutional Layer forward function
        """
        if input.data.is_sparse:
            support = torch.spmm(input, self.weight)
        else:
            support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):
    """ 2 Layer Graph Convolutional Network.

    Parameter
    """

    def __init__(self, nfeat, nhid, dropout=0.5, lr=0.01, weight_decay=5e-4, with_relu=True, with_bias=True, device=None):

        super(GCN, self).__init__()
        global filename
        assert device is not None, "Please specify 'device'!"
        self.device = device
        self.nfeat = nfeat
        self.hidden_sizes = [nhid]

        if filename=="Enron" :
            self.n_layers=3
        elif  filename=="Amazon":
            self.n_layers=3
        elif filename=="ACM":
            self.n_layers=1
        else:
            self.n_layers=1
        self.convlist=nn.ModuleList()

        self.convlist.append(GraphConvolution(nfeat, nhid, with_bias=with_bias))

        for i in range(self.n_layers):
            self.convlist.append(GraphConvolution(nhid, nhid, with_bias=with_bias))
        self.dropout = dropout
        self.lr = lr
        if not with_relu:
            self.weight_decay = 0
        else:
            self.weight_decay = weight_decay
        self.with_relu = with_relu
        self.with_bias = with_bias
        self.output = None
        self.best_model = None
        self.best_output = None
        self.adj_norm = None
        self.features = None
    def forward(self, x, adj):

        for i in range(self.n_layers):
            x = F.relu(self.convlist[i](x, adj))
            x = F.dropout(x, self.dropout, training=self.training)
        x = self.convlist[-1](x, adj)
    
    
        return x

    def initialize(self):
        """Initialize parameters of GCN.
        """
        for i in range(self.n_layers+1):
            self.convlist[i].reset_parameters()

#drop edge ---------------------------------------------------

def drop_dissimilar_edges1(features, adj,ratio=0.05,binary=False):
    """Drop dissimilar edges.
    """
    adj1=copy.deepcopy(adj)
    if not sp.issparse(adj):
        adj1 = sp.csr_matrix(adj)
    modified_adj = adj1.copy().tolil()
    # preprocessing based on features
    if binary:
        feature=sp.csc_matrix(features)
    print('=== GCN-Jaccrad ===')
    # isSparse = sp.issparse(features)
    edges = np.array(modified_adj.nonzero()).T
    removed_cnt = 0
    temp_mask=np.zeros(modified_adj.shape)
    all_results=[]
    for edge in tqdm(edges):
        n1 = edge[0]
        n2 = edge[1]
        if n1 > n2:
            continue
        if binary:
            C=jaccard_similarity(feature[n1],feature[n2])
        else:
            C = cosine_similarity(features[n1], features[n2])
        all_results.append(C)
        temp_mask[n1,n2]=C
        temp_mask[n2,n1]=C
    n=int(len(all_results)*ratio)

    removed_cnt=n
    simi_ordered=np.sort(all_results)
    threshold=simi_ordered[n]
    modified_adj[temp_mask<=threshold]=0

    print('removed %s edges in the original graph' % removed_cnt)
    return modified_adj

dtype = torch.cuda.FloatTensor
learning_rate=0.001
param_noise_sigma = 1
def add_noise(model):
    for n in [x for x in model.parameters() if len(x.shape)==2]:
        noise = torch.randn(n.size())*param_noise_sigma*learning_rate
        noise = noise.type(dtype)
        n.data = n.data + noise
def jaccard_similarity( a, b):
    intersection = a.multiply(b).count_nonzero()
    J = intersection * 1.0 / (a.count_nonzero() + b.count_nonzero() - intersection)
    return J
def cosine_similarity( a, b):
    #inner_product = (a * b).sum()
    #C = inner_product / np.sqrt(np.square(a).sum() + np.square(b).sum())
    C=np.exp(-1*np.sqrt(np.square(a-b).sum())/10000)
    return C
#drop edge-----------------------------------------------------------------------

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--C', default=3, type=float)
    parser.add_argument('--R', default=0.2, type=float)
    parser.add_argument('--S', default=0.2, type=float)
    parser.add_argument('--filename', default='Amazon', choices=['Enron','Amazon','Disney','BlogCatalog','Flickr','ACM'])
    return parser.parse_args()

args = get_args()
device=torch.device("cuda")
weight_decay=5e-8
filename=args.filename

from sklearn.decomposition import PCA
if __name__=="__main__":

    if filename=="Enron" or filename=="Amazon" or filename=="Disney":
        adj,features,labels=load_data("./data/"+filename+".mat")        
        all_epoch=100

    elif filename=="BlogCatalog" or filename=="Flickr" or filename=="ACM":
                
        data=scipy.io.loadmat("./"+filename+"_anomaly_best.mat")

        if filename=="ACM":

            data=scipy.io.loadmat("./"+filename+"_anomaly_best_test.mat")
            adj_sparse=data["adj"]
            adj=adj_sparse.todense()
            features=data["X"]
            labels=data["labels"]
            #print("features shape",features.shape)

        else:
            adj_sparse=data["adj"]
            adj=adj_sparse.todense()
            features=data["X"]
            labels=data["labels"]
            print("features shape",features.shape)
        all_epoch=100
    else:
        print("wrong!")
    if filename=="Amazon":
        features=preprocess_features_normalize(features)
        hiden_dim=128
        all_epoch=300
    else:
        hiden_dim=128

    print(filename,"ration:",args.R)
    adj_cleaned=drop_dissimilar_edges1(features,adj,ratio=args.R,binary=False)  #sparse lil
    feat_dim=features.shape[1]
    net1=GCN(feat_dim,hiden_dim,device=device)
    net1.to(device)
    features=torch.tensor(features).float().to(device)

    classifer1=PairwiseDistance(adj_cleaned,adj,features,hiden_dim,device)
    #classifer.to(device)

    adj_tensor=preprocess_adj(adj,device)
    opt1=optim.Adam(list(net1.parameters())+list(classifer1.classifier.parameters()),lr=0.01)
   
    net1.train()
    auc=None
    auc_un=None
    auc_average=[]
    variance_results=[]
    variance_acc=None
    for epoch in range(all_epoch):
        #print("epoch",epoch)
        net1.train()
        classifer1.classifier.train()

        embed=net1(features,adj_tensor)
        loss1=classifer1.make_loss(embed)
        loss=loss1
        opt1.zero_grad()
        loss.backward()

        opt1.step()
        if epoch>=40:
            net1.train()
            classifer1.classifier.train()
            embed=net1(features,adj_tensor)
            rank1,variance_temp=classifer1.evaluate(embed,labels)
            variance_results.append(variance_temp)
            auc=rank1

            auc_average.append(auc)

            if len(variance_results)>10:
                #print("shape",np.array(variance_results).shape)
                variance_uncertainty=np.std(np.array(variance_results),axis=0)

                auc_average_temp=np.average(np.array(auc_average),axis=0)

                auc_std_temp=np.std(np.array(auc_average),axis=0)

                temp=np.zeros(adj.shape)
                temp[adj==1]=variance_uncertainty

                temp_sum=np.sum(temp,axis=1)
                adj_sum=np.sum(np.array(adj),axis=1)
                average_uncertainty=temp_sum/(adj_sum+1e-12)

                min_average_temp=np.min(auc_average_temp)
                max_average_temp=np.max(auc_average_temp)

                min_average_uncertainty=np.min(average_uncertainty)
                max_average_uncertainty=np.max(average_uncertainty)
                auc_un=auc_average_temp/max_average_temp+average_uncertainty/max_average_uncertainty
                auc_un[np.isnan(auc_un)]=0
                auc_un[np.isinf(auc_un)]=0
            
                if epoch%20==0:
                    print("auc 4 HAV",roc_auc_score(labels,auc_un))
                    print("auc AHP",roc_auc_score(labels,auc_average_temp))

            


