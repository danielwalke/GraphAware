from torch.nn.functional import normalize
from GraphAwareNestedCVEvaluationInductive import GraphAwareNestedCVEvaluationInductive
from torch_geometric.datasets import PPI
import torch_geometric.transforms as T
from torch_geometric.utils import add_self_loops
from hyperopt import hp
import numpy as np
from tqdm.notebook import tqdm
from sklearn.linear_model import LogisticRegression
import shap
import torch
from NestedCV import index_to_mask
from xgboost import XGBClassifier
from IPython.display import clear_output

dataset = PPI(root='data/PPI')
dataset.transform = T.NormalizeFeatures()

def user_function(kwargs):
    return  kwargs["original_features"] + kwargs["summed_neighbors"]
    
class ModelSpace():
    def __init__(self):
        self.space = None
        self.initialize_space()

    def initialize_space(self):
        framework_choices = {
            'hops': [[3]],
            'attention_config': [{'inter_layer_normalize': False,
    'use_pseudo_attention': True,
    'cosine_eps': 0.01,
    'dropout_attn': None}],
            'user_function': [user_function],
        }
         
        self.space = {
            **{key: hp.choice(key, value) for key, value in framework_choices.items()}
        }
        
    def add_choice(self, key, items):
        self.space[key] = hp.choice(key, items)
        
    def add_uniform(self, key, limits: tuple):
        self.space[key] = hp.uniform(key, limits[0], limits[1])
        
    def add_loguniform(self, key, limits: tuple):
        self.space[key] = hp.loguniform(key, np.log(limits[0]), np.log(limits[1]))
        
    def add_qloguniform(self, key, limits, q):
        self.space[key] = hp.qloguniform(key, low=np.log(limits[0]), high=np.log(limits[1]), q=q)

class XGBSpace(ModelSpace):
    def __init__(self):
        super().__init__()

    def get_space(self):
        self.add_choice("booster", ["gbtree"])
        self.add_choice("n_estimators", [i for i in range(1_000, 2_200, 200)])
        self.add_choice("max_depth", [None])
        self.add_choice("max_delta_step", [0, 1, 2, 3])
        self.add_choice("min_child_weight", [None])
        self.add_choice("device", ["cuda:0"])
        self.add_choice("tree_method", ["hist"])
        self.add_choice("scale_pos_weight", [])

        self.add_choice("early_stopping_rounds", [10])
        self.add_choice("eval_metric", ["error"])
        
        self.add_loguniform("eta", (0.05, 0.7))
        self.add_loguniform("reg_lambda", (0.005, 100))
        self.add_loguniform("reg_alpha", (0.005, 100))

        self.add_uniform("gamma", (0, 0.2))
        self.add_uniform("colsample_bytree", (0.8, 1))
        self.add_uniform("scale_pos_weight", (1, 2)) ## TODO: Change to only within 1+-.1?
        return self.space   

xgb_space = XGBSpace()

store = dict({})

graph_aware_nestedCV_evaluation = GraphAwareNestedCVEvaluationInductive(0, XGBClassifier, dataset, max_evals= 7*20) #len(lr_space.get_space().keys())*20
graph_aware_nestedCV_evaluation.nested_cross_validate(5, 5, xgb_space.get_space())

print("PRINT_GRAPH_AWARE_LOGS_START")
print("Best params")
print(graph_aware_nestedCV_evaluation.nested_inductive_cv.best_params_per_fold)

print("Outer scores")
print(graph_aware_nestedCV_evaluation.nested_inductive_cv.outer_scores)
print(graph_aware_nestedCV_evaluation.nested_inductive_cv.outer_scores.mean())
print(graph_aware_nestedCV_evaluation.nested_inductive_cv.outer_scores.std())

print("Inner scores")
print(graph_aware_nestedCV_evaluation.nested_inductive_cv.inner_scores.mean())
print(graph_aware_nestedCV_evaluation.nested_inductive_cv.inner_scores.std())

print("Train times")
print(graph_aware_nestedCV_evaluation.nested_inductive_cv.train_times)
print(np.array(graph_aware_nestedCV_evaluation.nested_inductive_cv.train_times).mean())
print(np.array(graph_aware_nestedCV_evaluation.nested_inductive_cv.train_times).std())
print("PRINT_GRAPH_AWARE_LOGS_END")