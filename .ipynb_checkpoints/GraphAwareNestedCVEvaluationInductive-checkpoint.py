from tqdm.notebook import tqdm
import time
import torch 
from NestedCV import NestedInductiveCV
import numpy as np
from EnsembleFramework import Framework
import copy
from sklearn.metrics import f1_score
from torch_geometric.loader import DataLoader

def train_val_data(train_data, manual_seed = None, train_size = 0.8):
    if manual_seed:
        torch.manual_seed(manual_seed)
    train_index = torch.arange(len(train_data))
    min = int(train_size*train_index.shape[0])
    rand_train_index = torch.randperm(train_index.shape[0])
    rand_train_index_train_index = rand_train_index[:min]
    rand_train_index_val_index = rand_train_index[min:]

    new_train_idx = train_index[rand_train_index_train_index]
    new_val_idx = train_index[rand_train_index_val_index]

    return train_data[new_train_idx.tolist()], train_data[new_val_idx.tolist()]

class GraphAwareNestedCVEvaluationInductive:

    def __init__(self,device_id, model, data, minimize = True, max_evals = 100, parallelism = 1):
        self.device_id = device_id
        self.model = model
        self.training_times = []
        self.minimize = minimize
        self.data = data
        self.nested_transd_cv = None
        self.max_evals = max_evals
        self.parallelism = parallelism

    def nested_cross_validate(self, k_outer, k_inner, space):  

        # spaces = space_to_spaces()        
        def evaluate_fun(fitted_model, data):
            loader = iter(DataLoader(data, batch_size=len(data)))
            data = next(loader)
            pred_proba = fitted_model.predict_proba(data.x, data.edge_index, torch.ones(data.x.shape[0], dtype = torch.bool))
            return f1_score(data.y, np.round(pred_proba), average = "micro")

        def train_fun(data, hyperparameters): 
            torch.set_float32_matmul_precision('high')
            train_data, val_data = train_val_data(data)
            train_loader = iter(DataLoader(train_data, batch_size=len(train_data)))
            train_data = next(train_loader)

            val_loader = iter(DataLoader(val_data, batch_size=len(val_data)))
            val_data = next(val_loader)
            
            def transform_kwargs_fit(framework, kwargs, i):
                mask = torch.ones(val_data.x.shape[0]).type(torch.bool)
                val_out = framework.get_features(val_data.x, val_data.edge_index, mask, is_training = False)[0].cpu()   
                return {"eval_set":[(val_out, val_data.y)], "verbose":False}
            
            hops = hyperparameters["hops"]
            
            attention_config = hyperparameters["attention_config"] 
            attention_configs = [attention_config for _ in hops]
            
            user_function = hyperparameters["user_function"] 
            user_functions = [user_function for _ in hops]

            filtered_keys = list(filter(lambda key: key not in ["user_function", "hops", "attention_config"], hyperparameters.keys()))
            model_hyperparams = {key: hyperparameters[key] for key in filtered_keys}
            model = self.model(**model_hyperparams)
            models = [model for _ in hops]

            
            
            framework = Framework(user_functions=user_functions, 
                             hops_list=hops,
                             clfs=models,
                             gpu_idx=self.device_id,
                             handle_nan=0.0,
                            attention_configs=attention_configs)
            
                
            framework.fit(train_data.x, train_data.edge_index,
                          train_data.y, torch.ones(train_data.y.shape[0], dtype = torch.bool), transform_kwargs_fit = transform_kwargs_fit)
            return framework
            
        self.nested_inductive_cv = NestedInductiveCV(self.data, k_outer, k_inner, train_fun, evaluate_fun,max_evals = self.max_evals, parallelism = self.parallelism, minimalize = self.minimize)
        self.nested_inductive_cv.outer_cv(space)
        return self.nested_inductive_cv