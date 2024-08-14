from tqdm.notebook import tqdm
import time
import torch 
from NestedCV import NestedTransductiveCV
import numpy as np

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

class GNNNestedCVEvaluation:

    def __init__(self,device, GNN, data, max_evals, epochs = 10_000,  minimize = False, PATIENCE = 10, parallelism = 1):
        self.device = device
        self.epochs = epochs
        self.GNN = GNN
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.training_times = []
        self.minimize = minimize
        self.PATIENCE = PATIENCE
        self.data = data
        self.nested_transd_cv = None
        self.max_evals = max_evals
        self.parallelism = parallelism

    def nested_cross_validate(self, k_outer, k_inner, space):    
        def evaluate_fun(fitted_model, data, mask):
            fitted_model = fitted_model.to(self.device)
            with torch.inference_mode():
                fitted_model.eval()
                data = data.to(self.device)
                out = fitted_model(data.x, data.edge_index)
            fitted_model = fitted_model.cpu()
            check_equality = out.argmax(1)[mask] == data.y[mask]
            acc = check_equality.sum() / mask.sum()
            data = data.cpu()
            return acc.item()

        def train_fun(data, inner_train_mask, hyperparameters):
            start = time.time()
            scores = []
            lr = hyperparameters['lr']
            weight_decay = hyperparameters["weight_decay"]
            
            filtered_keys = list(filter(lambda key: key not in ["weight_decay", "lr"], hyperparameters.keys()))
            model_hyperparams = {key: hyperparameters[key] for key in filtered_keys}
            model = self.GNN(in_dim=data.x.shape[-1], **model_hyperparams).to(self.device)
            optim = torch.optim.Adam(model.parameters(), lr = lr, weight_decay=weight_decay)
            never_breaked = True
            train_mask, val_mask = train_val_masks(inner_train_mask, 42, 0.8)

            start_time = time.time()
            for epoch in range(self.epochs):
                model = model.to(self.device)
                data = data.to(self.device)
                optim.zero_grad()
                model.train()
                out = model(data.x, data.edge_index)
                loss = self.loss_fn(out.squeeze()[train_mask], data.y[train_mask])
                loss.backward()
                optim.step()

                score = evaluate_fun(model, data, val_mask)
                scores.append(score)
                worst_score = float("inf") if self.minimize else float("-inf")
                mean_score = np.mean(scores[-(self.PATIENCE + 1):]) if len(scores) > self.PATIENCE else worst_score
                not_improved = score > mean_score if self.minimize else  score < mean_score
                if epoch > (self.PATIENCE) and not_improved:
                    never_breaked = False
                    break
            train_time = time.time() - start_time
            data = data.cpu()
            return model, train_time
            
        self.nested_transd_cv = NestedTransductiveCV(self.data, k_outer, k_inner, train_fun, evaluate_fun,max_evals = self.max_evals, parallelism = self.parallelism, minimalize = self.minimize)
        self.nested_transd_cv.outer_cv(space)
        return self.nested_transd_cv