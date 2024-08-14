import torch
from torch import nn
from torch_geometric.nn import GCNConv, GATConv, ChebConv, SAGEConv
from torch.nn import Linear
import torch.nn.functional as F
from GNNNestedCVEvaluation import GNNNestedCVEvaluation
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.utils import add_self_loops
from hyperopt import hp
import numpy as np
from tqdm.notebook import tqdm
import sys

dataset_name = "PubMed" # sys.argv[1]
device = torch.device("cuda:1")
dataset = Planetoid(root='data/', name=dataset_name)
dataset.transform = T.NormalizeFeatures()
data = dataset[0]
print(f"Analyzing {dataset_name}: {data}")

class GCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout = .2, normalize = False, add_self_loops = True):
        super(GCN, self).__init__()
        hidden_dim = int(hidden_dim)
        self.conv1 = GCNConv(in_dim, hidden_dim, normalize = normalize, add_self_loops=add_self_loops)
        self.conv2 = GCNConv(hidden_dim, out_dim, normalize = normalize, add_self_loops=add_self_loops)
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return x

class GAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout = .2, heads = 1, add_self_loops = True):
        super(GAT, self).__init__()
        hidden_dim = int(hidden_dim)
        heads = int(heads)
        self.conv1 = GATConv(in_dim, hidden_dim, add_self_loops=add_self_loops, concat=True, dropout = dropout, heads = heads)
        self.conv2 = GATConv(hidden_dim*heads, out_dim, add_self_loops=add_self_loops, concat=False, dropout = dropout, heads = heads)
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return x

class Cheb(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout = .2, K = 2, normalization = "sym"):
        super(Cheb, self).__init__()
        hidden_dim = int(hidden_dim)
        K = int(K)
        self.conv1 = ChebConv(in_dim, hidden_dim, K = K, normalization = normalization)
        self.conv2 = ChebConv(hidden_dim, out_dim, K = K, normalization = normalization)
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return x

class SAGE(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout = .2, normalize = False, project = True, root_weight = True):
        super(SAGE, self).__init__()
        hidden_dim = int(hidden_dim)
        self.conv1 = SAGEConv(in_dim, hidden_dim, normalize = normalize, project = project, root_weight = root_weight)
        self.conv2 = SAGEConv(hidden_dim, out_dim, normalize = normalize, project = project, root_weight = root_weight)
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return x

class GNNSpace():
    def __init__(self, dataset):
        self.hidden_dim_limits = (8, 1024)
        self.dropout_limits = (0.0, 0.8)
        self.weight_decay_limits = (1e-5, 1e-2)
        self.lr_limits = (1e-4, 1e-1)
        self.out_dim = [dataset.num_classes]
        self.gnn_space = None
        self.initialize_space()

    def initialize_space(self):
        gnn_choices = {
            'out_dim': self.out_dim
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
        self.add_choice('add_self_loops', [True, False])
        return self.gnn_space    

class GATSpace(GNNSpace):
    def __init__(self, dataset):
        super().__init__(dataset)

    def get_space(self):
        self.add_qloguniform('heads', (1, 8), 2)
        self.add_choice('add_self_loops', [True, False])
        return self.gnn_space    

class ChebSpace(GNNSpace):
    def __init__(self, dataset):
        super().__init__(dataset)

    def get_space(self):
        self.add_qloguniform('K', (1, 4), 2)
        self.add_choice('normalization', ["sym", "rw", None])
        return self.gnn_space    

class SAGESpace(GNNSpace):
    def __init__(self, dataset):
        super().__init__(dataset)

    def get_space(self):
        self.add_choice('normalize', [True, False])
        self.add_choice('project', [True, False])
        self.add_choice('root_weight', [True, False])
        return self.gnn_space   

gcn_space = GCNSpace(dataset)
gat_space = GATSpace(dataset)
cheb_space = ChebSpace(dataset)
sage_space = SAGESpace(dataset)

gnns = [GCN, GAT, Cheb, SAGE]
gnn_spaces = [gcn_space.get_space(), gat_space.get_space(), cheb_space.get_space(), sage_space.get_space()]

score_store = {}
train_time_store = {}
param_store = {}

for i, space in tqdm(enumerate(gnn_spaces)):
    gnn_nestedCV_evaluation = GNNNestedCVEvaluation(device, gnns[i],data, max_evals= len(space.keys())*20)
    gnn_nestedCV_evaluation.nested_cross_validate(3, 3, space)
    score_store[gnns[i].__name__] = gnn_nestedCV_evaluation.nested_transd_cv.outer_scores
    param_store[gnns[i].__name__] = gnn_nestedCV_evaluation.nested_transd_cv.best_params_per_fold
    train_time_store[gnns[i].__name__] = np.array(gnn_nestedCV_evaluation.nested_transd_cv.train_times)

for key in score_store:
    print(f"Accuracy for {key}: {score_store[key].mean()} +- {score_store[key].std()}")
print(score_store)
print(param_store)

for key in train_time_store:
    print(f"Train time for {key}: {train_time_store[key].mean()} +- {train_time_store[key].std()}")