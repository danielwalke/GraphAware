from tqdm.notebook import tqdm
import time
import torch 
from NestedCV import NestedInductiveCV
import numpy as np
from GNNNestedCVEvaluation import GNNNestedCVEvaluation
from sklearn.metrics import f1_score

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

def graphs_x_to_dtype(graphs, dtype):
    for graph in graphs:
        graph.x = graph.x.to(dtype)
    return graphs
def check_model_dtype(model):
    for name, param in model.named_parameters():
        print(f'Parameter: {name}, dtype: {param.dtype}')
    for name, buffer in model.named_buffers():
        print(f'Buffer: {name}, dtype: {buffer.dtype}')

# Example usage


class GNNNestedCVEvaluationInductive(GNNNestedCVEvaluation):

    def __init__(self,device, GNN, data, max_evals, epochs = 10_000,  minimize = False, PATIENCE = 100, parallelism = 1):
        super().__init__(device, GNN, data, max_evals, epochs,  minimize, PATIENCE, parallelism )
        self.loss_fn = torch.nn.BCEWithLogitsLoss(reduction='mean')
        self.eval_steps = 10

    def nested_cross_validate(self, k_outer, k_inner, space):    
        
        def evaluate_fun(fitted_model, graphs):
            preds = []
            y = []
            graphs = [graph.to(self.device) for graph in graphs]
            # graphs = graphs_x_to_dtype(graphs, torch.float16)
            with torch.inference_mode():
                fitted_model.eval()
                for graph in graphs:
                    # graph = graph.to(self.device)
                    out = fitted_model(graph.x, graph.edge_index)
                    preds.append((out > 0).float())
                    y.append(graph.y)
                graphs = [graph.cpu() for graph in graphs]
            preds = (torch.cat(preds).cpu().detach() > 0)
            y = torch.cat(y).cpu()
            
            return f1_score(y, preds, average = "micro")

        def train_fun(data, hyperparameters):
            scores = []
            lr = hyperparameters['lr']
            weight_decay = hyperparameters["weight_decay"]
            
            filtered_keys = list(filter(lambda key: key not in ["weight_decay", "lr"], hyperparameters.keys()))
            model_hyperparams = {key: hyperparameters[key] for key in filtered_keys}
            model = self.GNN(in_dim=data.x.shape[-1], **model_hyperparams).to(self.device) #.half()
            # model = torch.compile(model)
            optim = torch.optim.Adam(model.parameters(), lr = lr, weight_decay=weight_decay)
            never_breaked = True
            train_data, val_data = train_val_data(data, 42, 0.8)
            train_data = [graph.to(self.device) for graph in train_data]
            # train_data = graphs_x_to_dtype(train_data, torch.float16)

            start_time = time.time()
            for epoch in range(self.epochs):
                # model = model.to(self.device)
                for graph in train_data:
                    model.train()
                    out = model(graph.x, graph.edge_index)
                    loss = self.loss_fn(out, graph.y)
                    optim.zero_grad()
                    loss.backward()
                    optim.step()
                
                if epoch % self.eval_steps != 0: continue
                score = evaluate_fun(model, val_data)
                scores.append(score)
                worst_score = float("inf") if self.minimize else float("-inf")
                mean_score = np.mean(scores[-(self.PATIENCE + 1):]) if len(scores) > self.PATIENCE else worst_score
                not_improved = score > mean_score if self.minimize else  score < mean_score
                
                if epoch > (self.PATIENCE) and not_improved:
                    never_breaked = False
                    break
            train_time = time.time() - start_time
            train_data = [graph.cpu() for graph in train_data]
            return model, train_time
            
        self.nested_inductive_cv = NestedInductiveCV(self.data, k_outer, k_inner, train_fun, evaluate_fun,max_evals = self.max_evals, parallelism = self.parallelism, minimalize = self.minimize)
        self.nested_inductive_cv.outer_cv(space)
        return self.nested_inductive_cv