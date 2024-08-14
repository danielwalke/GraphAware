from tqdm.notebook import tqdm
import time
import torch 
from NestedCV import NestedTransductiveCV
import numpy as np
from EnsembleFrameworkThreads import Framework
import copy
from sklearn.metrics import accuracy_score

def train_val_masks(train_mask, manual_seed = None, train_size = 0.8):
    if manual_seed:
        torch.manual_seed(manual_seed)
    train_index = train_mask.nonzero().squeeze()
    min = int(train_size*train_index.shape[0])
    rand_train_index = torch.randperm(train_index.shape[0])
    rand_train_index_train_index = rand_train_index[:min]
    rand_train_index_val_index = rand_train_index[min:]

    train_mask = torch.zeros_like(train_mask)
    val_mask = torch.zeros_like(train_mask)
    
    new_train_idx = train_index[rand_train_index_train_index]
    new_val_idx = train_index[rand_train_index_val_index]

    train_mask[new_train_idx] = 1
    val_mask[new_val_idx] = 1
    return train_mask, val_mask

def space_to_spaces(space, hops):
    spaces = []
    for hop in hops:
        spaces.append(copy.deepcopy(space))
    return spaces

class GraphAwareNestedCVEvaluation:

    def __init__(self,device_id, model, data, minimize = False, max_evals = 100, parallelism = 1, classifier_on_device = False, threads = None):
        self.device_id = device_id
        self.model = model
        self.training_times = []
        self.minimize = minimize
        self.data = data
        self.nested_transd_cv = None
        self.max_evals = max_evals
        self.parallelism = parallelism
        self.classifier_on_device = classifier_on_device
        self.threads = threads

    def nested_cross_validate(self, k_outer, k_inner, space):  

        # spaces = space_to_spaces()        
        def evaluate_fun(fitted_model, data, mask):
            pred_proba = fitted_model.predict_proba(data.x, data.edge_index, mask)
            return accuracy_score(data.y[mask].cpu().numpy(), pred_proba.argmax(1))

        def train_fun(data, inner_train_mask, hyperparameters):    
            device = torch.device(f"cuda:{self.device_id}")
            data.y =  torch.from_numpy(data.y).to(device) if not torch.is_tensor(data.y) else data.y.to(device)
            data.x =  torch.from_numpy(data.x).to(device) if not torch.is_tensor(data.x) else data.x.to(device)
            data.edge_index =  torch.from_numpy(data.edge_index).to(device) if not torch.is_tensor(data.edge_index) else data.edge_index.to(device)
            inner_train_mask =  torch.from_numpy(inner_train_mask).to(device) if not torch.is_tensor(inner_train_mask) else inner_train_mask.to(device)
            
            hops = hyperparameters["hops"]
            attention_config = hyperparameters["attention_config"] 
            attention_configs = [attention_config for _ in hops]
            
            user_function = hyperparameters["user_function"] 
            user_functions = [user_function for _ in hops]

            filtered_keys = list(filter(lambda key: key not in ["user_function", "hops", "attention_config"], hyperparameters.keys()))
            model_hyperparams = {key: hyperparameters[key] for key in filtered_keys}
            if "max_iter" in model_hyperparams:
                model_hyperparams["max_iter"] = int(model_hyperparams["max_iter"]) 
            model = self.model(**model_hyperparams)
            if self.classifier_on_device:
                model = model.to(device)
            models = [model for _ in hops]
            
            framework = Framework(user_functions=user_functions, 
                             hops_list=hops,
                             clfs=models,
                             gpu_idx=self.device_id,
                             handle_nan=0.0,
                            attention_configs=attention_configs, classifier_on_device = self.classifier_on_device, threads = self.threads)
            
            start_time = time.time()
            framework.fit(data.x, data.edge_index,
                          data.y, inner_train_mask)
            train_time = time.time() - start_time

            data.y =  data.y.cpu()
            data.x =  data.x.cpu()
            data.edge_index =  data.edge_index.cpu()            
            return framework, train_time
            
        self.nested_transd_cv = NestedTransductiveCV(self.data, k_outer, k_inner, train_fun, evaluate_fun,max_evals = self.max_evals, parallelism = self.parallelism, minimalize = self.minimize)
        self.nested_transd_cv.outer_cv(space)
        return self.nested_transd_cv