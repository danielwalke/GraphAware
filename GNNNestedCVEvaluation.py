import  matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from sklearn.metrics import f1_score
from torcheval.metrics.functional import multiclass_f1_score
import copy
import time
import torch 
from NestedCV import NestedTransductiveCV

TRAIN = "train"
VAL = "val"
TEST = "test"

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

    def __init__(self,device, GNN,data, epochs = 10_000,  minimize = True):
        self.device = device
        self.epochs = epochs
        self.GNN = GNN
        self.loss_fn = torch.nn.BCEWithLogitsLoss(reduction='mean')
        self.training_times = []
        self.minimize = minimize
        self.PATIENCE = 10
        self.data = data
        self.nested_transd_cv = None

    def nested_cross_validate(self, k_outer, k_inner, space):    
        def evaluate_fun(fitted_model, data, mask):
            with torch.inference_mode():
                fitted_model.eval()
                out = fitted_model(data.x, data.edge_index)
            check_equality = out.squeeze()[mask] == data.y[mask]
            acc = check_equality.sum()
            return acc

        def train_fun(data, inner_train_mask, hyperparameters):
            start = time.time()
            scores = []
            lr = params.pop('lr', 3e-4)
            weight_decay = params.pop('weight_decay', 3e-4)
            model = self.GNN(in_dim=data.x.shape[-1], **hyperparameters).to(self.device)
            optim = torch.optim.Adam(model.parameters(), lr = lr, weight_decay=weight_decay)
            never_breaked = True
            train_mask, val_mask = train_val_masks(inner_train_mask, 42, 0.8)
            
            for epoch in tqdm(range(self.epochs)):
                data = data.to(device)
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
                improved = score < mean_score if self.minimize else  score > mean_score
                if epoch > (self.PATIENCE) and improved:
                    never_breaked = False
                    break
            return model
        self.nested_transd_cv = NestedTransductiveCV(self.data, k_outer, k_inner, train_fun, eval_fun,max_evals = 100, parallelism = 1, minimalize = True)
        self.nested_transd_cv.outer_cv(space)
        return self.nested_transd_cv