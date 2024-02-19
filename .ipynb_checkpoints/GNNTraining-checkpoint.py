import  matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from sklearn.metrics import f1_score
from torcheval.metrics.functional import multiclass_f1_score
import copy
import time
import torch 

TRAIN = "train"
VAL = "val"
TEST = "test"

class GNNTraining:

    def __init__(self,device, GNN, sets, hidden_dim, lr, dropout = 0.0,weight_decay=0.0, epochs = 1000, kwargs = {}):
        self.device = device
        self.model = GNN(in_dim=sets[TRAIN].x.shape[-1], hidden_dim=hidden_dim, out_dim=sets[TRAIN].y.shape[-1], dropout = dropout, **kwargs).to(self.device)
        self.optim = torch.optim.Adam(self.model.parameters(), lr = lr, weight_decay=weight_decay)
        self.loss_fn = torch.nn.BCEWithLogitsLoss(reduction='mean')
        self.best_model = self.model
        self.best_val_loss = float("inf")
        self.PATIENCE = 100
        self.PATIENCE_COUNT = 0
        self.epochs = epochs
        self.training_time = None
        self.sets = sets
        self.set_names = self.sets.keys()
        self.losses = dict()
        for set_name in self.set_names: self.losses[set_name] = []
        self.scores = dict()
        for set_name in self.set_names: self.scores[set_name] = []

    def validate(self, set_name):
        acc_loss = 0
        batch_size = 0
        ground_truth = []
        preds = []
        with torch.inference_mode():
            self.model.eval()
            set = self.sets[set_name]
            for loader in set:
                loader = loader.to(self.device)
                out = self.model(loader.x, loader.edge_index)
                loss = self.loss_fn(out, loader.y)
                ground_truth.append(loader.y)
                preds.append((out > 0).float())
                acc_loss += loss.item()
                batch_size += 1
                
        score = f1_score(torch.cat(ground_truth).cpu(), torch.cat(preds).detach().cpu(), average ="micro")
        self.scores[set_name].append(score)
        
        if set_name == VAL and self.best_val_loss >= acc_loss:
            self.best_val_loss = acc_loss
            self.PATIENCE_COUNT = 0
        else:
            if self.PATIENCE_COUNT == 0:
                self.best_model = copy.deepcopy(self.model)
            self.PATIENCE_COUNT += 1
        # models.append(copy.deepcopy(model).cpu())
        loss_per_batch = acc_loss / batch_size
        self.losses[set_name].append(loss_per_batch)
        return loss_per_batch

    def train(self):    
        start = time.time()
        
        never_breaked = True
        for epoch in tqdm(range(self.epochs)):
            acc_loss = 0
            for loader in self.sets[TRAIN]:
                loader = loader.to(self.device)
                self.optim.zero_grad()
                self.model.train()
                out = self.model(loader.x, loader.edge_index)
                loss = self.loss_fn(out, loader.y)
                acc_loss += loss.item()
                loss.backward()
                self.optim.step()
                
            for set_name in self.set_names:
                if set_name != VAL:
                    continue
                self.validate(set_name)
            if epoch > (self.PATIENCE_COUNT) and self.PATIENCE_COUNT == self.PATIENCE:
                never_breaked = False
                break
        if never_breaked:
            self.best_model = copy.deepcopy(self.model)
        self.training_time = time.time() - start
        return self.best_model