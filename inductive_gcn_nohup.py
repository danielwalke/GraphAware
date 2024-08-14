import torch
from torch import nn
from torch_geometric.nn import GCNConv, GATConv, ChebConv, SAGEConv
from torch.nn import Linear
import torch.nn.functional as F
from GNNNestedCVEvaluationInductive import GNNNestedCVEvaluationInductive
from torch_geometric.datasets import PPI
import torch_geometric.transforms as T
from torch_geometric.utils import add_self_loops
from hyperopt import hp
import numpy as np
from tqdm.notebook import tqdm
from IPython.display import clear_output

class GCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim = 121, dropout = .2, normalize = False, add_self_loops = True):
        super(GCN, self).__init__()
        hidden_dim = int(hidden_dim)
        self.conv1 = GCNConv(in_dim, hidden_dim, normalize = normalize, add_self_loops=add_self_loops)
        self.conv2 = GCNConv(hidden_dim, hidden_dim, normalize = normalize, add_self_loops=add_self_loops)
        self.conv3 = GCNConv(hidden_dim, out_dim, normalize = normalize, add_self_loops=add_self_loops)
        self.lin1 = Linear(in_dim, hidden_dim)
        self.lin2 = Linear(hidden_dim, hidden_dim)
        self.lin3 = Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, x, edge_index):
        x = self.dropout(x)
        x = self.conv1(x, edge_index) +self.lin1(x)
        x = F.elu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)+self.lin2(x)
        x = F.elu(x)
        x = self.dropout(x)
        x = self.conv3(x, edge_index)+self.lin3(x)
        return x

dataset = PPI(root='data/PPI')
dataset.transform = T.NormalizeFeatures()
dataset.num_classes

class GNNSpace():
    def __init__(self, dataset):
        EPS = 1e-6
        # self.hidden_dim_limits = (8, 1024)
        # self.dropout_limits = (0.0, 0.8)
        # self.weight_decay_limits = (1e-5, 1e-2)
        # self.lr_limits = (1e-4, 1e-1)
        # self.out_dim = [dataset.num_classes]
        # self.gnn_space = None
        # self.initialize_space()
        self.hidden_dim_limits = (8, 512)
        self.dropout_limits = (0.0, 0.8)
        self.weight_decay_limits = (1e-7, 1e-4)
        self.lr_limits = (1e-5, 1e-2)
        self.out_dim = [dataset.num_classes]
        self.gnn_space = None
        self.initialize_space()

    def initialize_space(self):
        gnn_choices = {
            # 'out_dim': self.out_dim
        }
         
        self.gnn_space = {
            **{key: hp.choice(key, value) for key, value in gnn_choices.items()},
            'lr': hp.loguniform('lr',np.log(self.lr_limits[0]), np.log(self.lr_limits[1])),
            'weight_decay': hp.loguniform('weight_decay',np.log(self.weight_decay_limits[0]), np.log(self.weight_decay_limits[1])),
            'dropout': hp.uniform('dropout', self.dropout_limits[0], self.dropout_limits[1]),
            'hidden_dim': hp.qloguniform('hidden_dim', low=np.log(self.hidden_dim_limits[0]), high=np.log(self.hidden_dim_limits[1]), q=16)
        }
        
    def add_choice(self, key, items):
        self.gnn_space[key] = hp.choice(key, items)
        
    def add_uniform(self, key, limits: tuple):
        self.gnn_space[key] = hp.uniform(key, limits[0], limits[1])
        
    def add_loguniform(self, key, limits: tuple):
        self.gnn_space[key] = hp.loguniform(key, np.log(limits[0]), np.log(limits[1]))
        
    def add_qloguniform(self, key, limits, q):
        self.gnn_space[key] = hp.qloguniform(key, low=np.log(limits[0]), high=np.log(limits[1]), q=q)

class GCNSpace(GNNSpace):
    def __init__(self, dataset):
        super().__init__(dataset)

    def get_space(self):
        self.add_choice('normalize', [True])
        self.add_choice('add_self_loops', [True]) #False
        return self.gnn_space    

data = dataset
device = torch.device("cuda:1")
score_store = {}
param_store = {}
gcn_space = GCNSpace(dataset)

gnn_nestedCV_evaluation = GNNNestedCVEvaluationInductive(device, GCN,data, max_evals= len(gcn_space.get_space().keys())*20, epochs  = 10_000, PATIENCE=10)
gnn_nestedCV_evaluation.nested_cross_validate(5,5, gcn_space.get_space())#5, 5
score_store[GCN.__name__] = gnn_nestedCV_evaluation.nested_inductive_cv.outer_scores
param_store[GCN.__name__] = gnn_nestedCV_evaluation.nested_inductive_cv.best_params_per_fold

print("Best params")
print(gnn_nestedCV_evaluation.nested_inductive_cv.best_params_per_fold)

print("Outer scores")
print(gnn_nestedCV_evaluation.nested_inductive_cv.outer_scores)
print(gnn_nestedCV_evaluation.nested_inductive_cv.outer_scores.mean())
print(gnn_nestedCV_evaluation.nested_inductive_cv.outer_scores.std())

print("Inner scores")
print(gnn_nestedCV_evaluation.nested_inductive_cv.inner_scores.mean())
print(gnn_nestedCV_evaluation.nested_inductive_cv.inner_scores.std())

print("Train times")
print(gnn_nestedCV_evaluation.nested_inductive_cv.train_times)
print(np.array(gnn_nestedCV_evaluation.nested_inductive_cv.train_times).mean())
print(np.array(gnn_nestedCV_evaluation.nested_inductive_cv.train_times).std())
