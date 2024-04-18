# GraphAware

A framework designed to make scikit learn classifiers "graph-aware". With GraphAware, classical machine learning algorithms like logistic regression or XGBoost are able to exploit features and the graph structures similar to the popular GraphNeuralNetworks.

## Table of Contents 
|content                          |
|---------------------------------|
|[Overview](#overview)     |
|[Installation](#installation) |
|[Usage](#usage) |
|[Automated Hyperparameter search](#hyperparameter) |
|[Examples](#examples) |
|[Contact](#contact) |
|[Fundings](#fundings)           |
|[Competing intrests](#competingIntrests) |

<a name="overview"/>

## Overview
<div style="text-align: justify">
Machine learning algorithms like XGBoost showed promising results in many applications like disease diagnosis. However, they cannot exploit the connections in graph structured data like citation networks (e.g., Cora) or protein-protein interaction networks (PPI). Although different graph learning algorithms like Graph Neural Networks (GNNs) were proposed, there is still a demand for new frameworks due to long training times and high number of trainable parameters in GNNs. We propose GraphAware, a new framework to analyze graph-structured data with machine learning algorithms. GraphAware trains separate machine learning classifiers on feature sets generated from aggregated neighborhoods of different orders and combines their outputs for the final prediction. We showed that the accuracy (Cora, CiteSeer, PubMed) or micro f1-score (PPI) for GraphAware (Cora: 0.831, CiteSeer: 0.720, PubMed: 0.802, PPI: 0.984) is higher or least comparable to the best performing GNN GAT (Cora: 0.831, CiteSeer: 0.708, PubMed: 0.790, PPI: 0.991). Furthermore, the training time required for GraphAware is much shorter (Cora: 1.09 s, CiteSeer: 2.56 s, PubMed: 1.77 s, PPI: 149.88 s) compared to GAT (Cora: 4.91 s, CiteSeer: 5.05 s, PubMed: 5.63 s, PPI: 338.72 s). GraphAware is compatible with popular python packages like sklearn and XGBoost and is open-sourced on https://github.com/danielwalke/GraphAware. 
</div>

<a name="installation"/>

## Installation/Setup

1) Clone the project
   ```bash
   git clone https://github.com/danielwalke/GraphAware
   ```
3) Navigate in te project
   ```bash
   cd GraphAware/
   ```
4) Install requirements
   ```bash
   pip install -r requirements.txt
   ```
5) Import the framework in python or jupyter notebooks
   ```bash
   from EnsembleFramework import Framework
   ```

<a name="usage"/>

## Usage
1) Import the Framework, import torch and import sklearn:
```bash
from EnsembleFramework import Framework
import sklearn
import torch
```
2) Define a list describing the order of neighborhoods you want to incorporate, e.g.:
```bash
hops = [0, 2]
```
3) Define a aggregation function, e.g.:
```bash
def user_function(kwargs):
    mean = kwargs["mean_neighbors"]
    orignal_features = kwargs["original_features"]
    return orignal_features - mean
```
4) Define a classifier that is used to analyze each order of neighborhood, e.g.:
```bash
clfs=[sklearn.ensemble.RandomForestClassifier(max_leaf_nodes=50, n_estimators=1000, random_state=42), sklearn.ensemble.RandomForestClassifier(max_leaf_nodes=50, n_estimators=1000, random_state=42)]
```
5) Optional: You can set influence score configurations if you want to weight the contributoin of individual neighbors based on their cosine similarity (use_pseudo_attention) or cut-off neighbors that are too dissimilar (cosine_eps):
attention_configs = [{'inter_layer_normalize': True,'use_pseudo_attention':True,'cosine_eps':.01, 'dropout_attn': None} for i  in hops]
6) Initialize the framework, e.g.:
```bash
framework = Framework(hops_list= hops,
                      clfs=clfs,
                      attention_configs=[None for i in hops],
                      handle_nan=0.0,
                      gpu_idx=None,
                      user_functions=[user_function for i in hops]
)
```
Note that the hops_list, clfs, attention_configs, and user_functions needs to have the same length. Detailed documentations how individual parameters have to be set can be found in EnsembleFramework.py.
7) Fit the data, e.g.:
X_train- features of the train set, edge_index - edge index in COO format for the train graph, y_train- labels for the train set, train_mask- boolean mask if you want to train it under transductive settings otherwise you can set it to None
```bash
framework.fit(X_train=features_train, edge_index=edge_index_train,y_train=labels_train, train_mask=torch.ones(features_train.shape[0]).type(torch.bool))
```
8) Get the final prediction probabilities:
features_test- features of the test set, edge_index_test - edge index in COO format for the test graph, test_mask- boolean mask if you want to train it under transductive settings otherwise you can set it to None
```bash
predict_proba = framework.predict_proba(features_test, edge_index_test, torch.ones(features_test.shape[0]).type(torch.bool))
```
9) Evaluations, e.g.:
y_test- labels for the test set
```bash
from sklearn.metrics import roc_auc_score
roc_auc_score(y_test, predict_proba[:,1])
```
<a name="hyperparameter"/>

## Automated Hyperparameter search
You can defined your own hyperparameters or use the automated hyperparameter search.
1) Import hyperopt, the preferred evaluation metric and your model, e.g.:
```
from sklearn.linear_model import LogisticRegression
from hyperopt import hp
from AutoTune2 import AutoSearch
from sklearn.metrics import accuracy_score
```

2) Define the dataset, e.g.:
```
from torch_geometric.datasets import Planetoid
cora_dataset = Planetoid(root = "./data", name = "Cora")
X =  dataset[0].x 
y =  dataset[0].y 

test =  dataset[0].test_mask
train = dataset[0].train_mask 
val =  dataset[0].val_mask

dataset = dict({})
dataset["X"] = X ## feature set
dataset["y"] = y ## labels
dataset["test"] = test  ## boolean test mask 
dataset["train"] = train ## boolean train mask 
dataset["val"] = val ## boolean val mask 
dataset["edge_index"] = edge_index
```
3)  Define the classifiers you want to evaluate:
```
clfs = [LogisticRegression]
```
4) Define the hyperparameter space, e.g.:
- Discrete values here:
```
lr_choices = {
    'penalty': ["l2"],
    'max_iter': [2**i for i in range(6, 15)],
    
}
```
- Numeric or continous values here:
```
space_lr = {
    **{key: hp.choice(key, value) for key, value in lr_choices.items()},
    'tol': hp.loguniform('tol', -11, -3),
    'C': hp.uniform('C', 0.0, 10)
}
clfs_space = dict({})
clfs_space["LogisticRegression"] = space_lr
```

5) Define the aggregation functions you want to evaluate, e.g.: 
```
def mean_user_function(kwargs):
    return  kwargs["original_features"] + kwargs["mean_neighbors"]
    
def sum_user_function(kwargs):
    return  kwargs["original_features"] + kwargs["summed_neighbors"]

user_functions = [mean_user_function, sum_user_function]
```  

6) Define the lists of order of neighborhoods you want to evaluate, e.g.: 
```
hops_lists = [[0,3,8], [0,3]]
```

7) Define the list of influence score configurations you wan tto evaluate, e.g.:
```
attention_configs = [None,{'inter_layer_normalize': False,
                     'use_pseudo_attention':True,
                     'cosine_eps':.01,
                     'dropout_attn': None}, 
                     {'inter_layer_normalize': True,
                     'use_pseudo_attention':True,
                     'cosine_eps':.01,
                     'dropout_attn': None},
                     {'inter_layer_normalize': True,
                     'use_pseudo_attention':True,
                     'cosine_eps':.001,
                     'dropout_attn': None}]
```
8) Initialize the searcher
max_evals - maximum number of evaluations per classifier, aggregation list, influence score configurations, and aggregation functions provided
pred_metric - dedired evaluation metric
parallelism - number of parallel executions
```
searcher = AutoSearch(dataset, max_evals=500, pred_metric = accuracy_score, parallelism=50)
```
9) Start the search:
```
store = searcher.search(clfs, clfs_space, hops=hops_lists, user_functions= user_functions,
                         attention_configs = attention_configs)
```
10) Evaluate the search results:
```
print(store)
```

<a name="examples"/>

## Examples
You can find examples for the usage under transudctive settings [here](https://github.com/danielwalke/GraphAware/blob/main/GraphAwareEvaluation_transductive.ipynb) and for inductive settings [here](https://github.com/danielwalke/GraphAware/blob/main/GraphAware_Evaluation_indcutive.ipynb)


<a name="contact"/>

## Contact
If you have any questions, struggles with the repository, want new features or want to cooperate to hesitate to contact me: 
daniel.walke@ovgu.de

<a name="fundings"/>

## Fundings
We thank the German Research Foundation (DFG) for funding this study under the project ‘Optimizing graph databases focusing on data processing and integration of machine learning for large clinical and biological datasets’ [project number 463414123; grant numbers HE 8077/2-1, SA 465/53-1]).

<a name="competingIntrests"/>

## Competing intrests
The authors declare that they have no competing interests.
