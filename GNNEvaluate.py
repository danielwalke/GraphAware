from sklearn.metrics import f1_score
import torch


TRAIN = "train"
VAL = "val"
TEST = "test"

class GNNEvaluate:
    def __init__(self, sets, device):
        self.sets = sets
        self.device = device
        
    def evaluate(self, best_model, set_name = TEST):
        ground_truth = []
        preds = []
        for loader in self.sets[set_name]:
            loader = loader.to(self.device)
            out = best_model.to(self.device)(loader.x, loader.edge_index)
            preds.append(out)
            ground_truth.append(loader.y)    
        return f1_score(torch.cat(ground_truth).cpu(), (torch.cat(preds).cpu().detach() > 0).float(), average = "micro")