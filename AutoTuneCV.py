from hyperopt import fmin, tpe, hp,STATUS_OK, SparkTrials, space_eval 
from sklearn.metrics import accuracy_score
from tqdm.notebook import tqdm
from EnsembleFramework import Framework
from torch.nn.functional import normalize
from sklearn.multioutput import MultiOutputClassifier
import torch 
from torch import nn
import torch
import numpy as np
from sklearn.model_selection import KFold

##Shift to other file
class NestedCVInductive:

    def __init__(self):
        pass

    def forward(self, graphs, k, scoring):
        kf = KFold(n_splits=k)
        scores = np.zeros(k)
        
        for i, (train_index, test_index) in enumerate(kf.split(graphs)):
            train_graphs = graphs[train_index]
            test_graphs = graphs[test_index]
            auto_tune_cv = AutoTuneCVInductive()
            best_clf = auto_tune_cv(train_graphs)
            score = evaluate(best_clf, test_graphs, scoring = scoring)
            scores[i] = score
        return scores.mean()

class AutoTuneCVInductive:
    def __init__(self):
        pass

    def forward(self, graphs, k):
        kf = KFold(n_splits=k)
        for i, (train_index, test_index) in enumerate(kf.split(graphs)):
            train_graphs = graphs[train_index]
            test_graphs = graphs[test_index]            

class NestedCVTransductive:
    def __init__(self):
        pass

    def forward(self, graph, k, scoring):
        kf = KFold(n_splits=k)
        
        scores = np.zeros(k)
        for i, (train_index, test_index) in enumerate(kf.split(graph.x)):
            auto_tune_cv = AutoTuneCVTransductive()
            best_clf = auto_tune_cv(graph, train_index)
            score = evaluate(best_clf, graph, test_index, scoring = scoring)
            scores[i] = score
        return scores.mean()
            
            
class AutoTuneCVTransductive:
    def __init__(self, k, n_jobs = 1, pred_metric,max_evals = 100, pred_metric_kwargs = {}, parallelism = 3):
        self.k = k
        self.n_jobs = n_jobs
        self.pred_metric_kwargs = pred_metric_kwargs
        self.pred_metric = pred_metric
        self.max_evals = max_evals
        self.parallelism = parallelism
        self.data = None
        pass

    def objective(self, params):
        k = self.k
        scores = np.zeros(k)
        for i, (train_index, test_index) in enumerate(kf.split(graph.x)):
            train_graphs = graphs[train_index]
            test_graphs = graphs[test_index] 
            user_function = params.pop('user_function', DEF_USER_FUNCTIONS[0])
            hop = params.pop('hop', 2)
            attention_config = params.pop('attention_config', DEF_ATTENTION_CONFIGS[0])
            
            model = self.clf(**params)
            framework = Framework([user_function], 
                             hops_list=[hop],
                             clfs=[model],
                             gpu_idx=0,
                             handle_nan=0.0,
                            attention_configs=[attention_config], multi_target_class=auto_search.multi_target_class)
        
            framework.fit(self.data.X, self.data.edge_index,
                          self.data.y, train_index, kwargs_multi_clf_list = [{"n_jobs":self.n_jobs}])
            y_pred = framework.predict(self.data.X, self.data.edge_index, val_index)
            score = self.pred_metric(self.data.y[val_index],
                                            y_pred,
                                            **self.pred_metric_kwargs)
            scores[i] = score
        return {'loss': -scores.mean(), 'status': STATUS_OK}
    
    def forward(self, graph, space):
        self.data = graph
        spark_trials = SparkTrials(parallelism = self.parallelism)
        best_params = fmin(self.objective, space, algo=tpe.suggest, max_evals=self.max_evals, trials=spark_trials, verbose = False)
        return best_params



            


            

def upd_user_function(kwargs):
    return  nn.functional.normalize(kwargs["updated_features"] + kwargs["summed_neighbors"], p = 2.0, dim = -1)
# DEF_ATTENTION_CONFIGS= [None,{'inter_layer_normalize': False,
#                      'use_pseudo_attention':True,
#                      'cosine_eps':.01,
#                      'dropout_attn': None}, 
#                      {'inter_layer_normalize': True,
#                      'use_pseudo_attention':True,
#                      'cosine_eps':.01,
#                      'dropout_attn': None},
#                      {'inter_layer_normalize': True,
#                      'use_pseudo_attention':True,
#                      'cosine_eps':.001,
#                      'dropout_attn': None}]
DEF_ATTENTION_CONFIGS= [{'inter_layer_normalize': False,
                     'use_pseudo_attention':True,
                     'cosine_eps':.01,
                     'dropout_attn': None}]
DEF_HOPS = [3, 5]
DEF_MAX_EVALS = 1000
def norm_user_function(kwargs):
    return  normalize(kwargs["original_features"] + kwargs["summed_neighbors"], p=2.0, dim = 1)
    
def user_function(kwargs):
    return  kwargs["original_features"] + kwargs["summed_neighbors"]
    
DEF_USER_FUNCTIONS = [user_function] #upd_user_function,norm_user_function

class Data():
    def __init__(self, X, y, edge_index):
        self.X = X
        self.y = y
        self.edge_index = edge_index
        self.train = None
        self.val = None
        self.test = None
        # self.X_train = None
        # self.X_val = None
        # self.X_test = None
        # self.y_train = None
        # self.y_val = None
        # self.y_test = None
    
    def set_train(self, train):
        self.train = train

    def set_test(self, test):
        self.test = test

    def set_val(self, val):
        self.val = val

    # def set_X_train(self, X):
    #     self.X_train = X

    # def set_X_val(self, X):
    #     self.X_val = X

    # def set_X_test(self, X):
    #     self.X_test = X

class SparkTune():
    def __init__(self, clf,user_function,hop,attention_config, auto_search):
        self.clf = clf
        self.auto_search = auto_search
        self.user_function = user_function
        self.hop = hop
        self.attention_config = attention_config
        
    def objective(self, params):
        model = self.clf(**params)
        auto_search = self.auto_search
        framework = Framework([self.user_function], 
                         hops_list=[self.hop],
                         clfs=[model],
                         gpu_idx=0,
                         handle_nan=0.0,
                        attention_configs=[self.attention_config], multi_target_class=auto_search.multi_target_class)
        score = None
        if auto_search.is_transductive:
            framework.fit(auto_search.data.X, auto_search.data.edge_index,
                          auto_search.data.y, auto_search.data.train, kwargs_multi_clf_list = [{"n_jobs":11}])
            y_pred = framework.predict(auto_search.data.X, auto_search.data.edge_index, auto_search.data.val)
            score = auto_search.pred_metric(auto_search.data.y[auto_search.data.val],
                                            y_pred,
                                            **auto_search.pred_metric_kwargs)
        if not auto_search.is_transductive:
            framework.fit(auto_search.train_data.X, auto_search.train_data.edge_index,
                          auto_search.train_data.y, torch.ones(auto_search.train_data.X.shape[0]).type(torch.bool), kwargs_multi_clf_list = [{"n_jobs":11}])
            y_pred = framework.predict(auto_search.val_data.X, auto_search.val_data.edge_index,
                                       torch.ones(auto_search.val_data.X.shape[0]).type(torch.bool))
            score = auto_search.pred_metric(auto_search.val_data.y,
                                            y_pred,
                                            **auto_search.pred_metric_kwargs)
        return {'loss': -score, 'status': STATUS_OK}
    
    def search(self, space):
        spark_trials = SparkTrials(parallelism = self.auto_search.parallelism)
        best_params = fmin(self.objective, space, algo=tpe.suggest, max_evals=self.auto_search.max_evals, trials=spark_trials, verbose = False)
        return best_params


class AutoSearch:
    
    def __init__(self, data_dict, max_evals = 200, multi_target_class = False, pred_metric= accuracy_score, pred_metric_kwargs = {}, is_transductive = True, parallelism = 3):
        self.data_dict = data_dict
        self.max_evals = max_evals
        self.multi_target_class = multi_target_class
        self.pred_metric = pred_metric
        self.pred_metric_kwargs = pred_metric_kwargs
        self.is_transductive = is_transductive
        self.data:Data = None
        self.train_data:Data = None
        self.val_data:Data = None
        self.test_data:Data = None
        self.parallelism = parallelism


    def parse_data(self):
        dataset = self.data_dict
        if self.is_transductive:
            self.data = Data(dataset["X"], dataset["y"], dataset["edge_index"])
            self.data.set_test(dataset["test"])
            self.data.set_val(dataset["val"])
            self.data.set_train(dataset["train"])
        if not self.is_transductive:
            self.train_data = Data(dataset["X_train"], dataset["y_train"], dataset["edge_index_train"])
            self.val_data = Data(dataset["X_val"], dataset["y_val"], dataset["edge_index_val"])
            self.test_data = Data(dataset["X_test"], dataset["y_test"], dataset["edge_index_test"])

    def search_hop_clf_attention_config(self, hop, clf, user_function, attention_config, space):
        self.parse_data()
        
        sparkTune = SparkTune(clf,user_function,hop,attention_config, self)
        params = sparkTune.search(space)
        params = space_eval(space, params) ## index choices to original choices
        
        model = clf(**params)
        framework = Framework([user_function], 
                         hops_list=[hop],
                         clfs=[model],
                         gpu_idx=0,
                         handle_nan=0.0,
                        attention_configs=[attention_config], multi_target_class=self.multi_target_class)
        if self.is_transductive:
            framework.fit(self.data.X, self.data.edge_index, self.data.y, self.data.train, kwargs_multi_clf_list = [{"n_jobs":11}])
        if not self.is_transductive:
            framework.fit(self.train_data.X, self.train_data.edge_index, self.train_data.y, torch.ones(self.train_data.X.shape[0]).type(torch.bool), kwargs_multi_clf_list = [{"n_jobs":11}])

        train_acc, val_acc, test_acc = None, None, None
        if self.is_transductive:
            val_pred = framework.predict(self.data.X, self.data.edge_index,
                                        self.data.val)
            train_pred = framework.predict(self.data.X, self.data.edge_index,
                                        self.data.train)
            test_pred = framework.predict(self.data.X, self.data.edge_index,
                                        self.data.test)
            train_acc = self.pred_metric(self.data.y[self.data.train], train_pred, **self.pred_metric_kwargs)
            val_acc = self.pred_metric(self.data.y[self.data.val], val_pred, **self.pred_metric_kwargs)
            test_acc = self.pred_metric(self.data.y[self.data.test], test_pred, **self.pred_metric_kwargs)
        if not self.is_transductive:
            val_pred = framework.predict(self.val_data.X, self.val_data.edge_index,
                                           torch.ones(self.val_data.X.shape[0]).type(torch.bool))
            train_pred = framework.predict(self.train_data.X, self.train_data.edge_index,
                                           torch.ones(self.train_data.X.shape[0]).type(torch.bool))
            test_pred = framework.predict(self.test_data.X, self.test_data.edge_index,
                                           torch.ones(self.test_data.X.shape[0]).type(torch.bool))
            train_acc = self.pred_metric(self.train_data.y, train_pred, **self.pred_metric_kwargs)
            val_acc = self.pred_metric(self.val_data.y, val_pred, **self.pred_metric_kwargs)
            test_acc = self.pred_metric(self.test_data.y, test_pred, **self.pred_metric_kwargs)
            
        search_dict = dict({})
        search_dict["train_acc"] = train_acc
        search_dict["val_acc"] = val_acc
        search_dict["test_acc"] = test_acc
        search_dict["model"] = model
        search_dict["user_function"] = user_function
        return search_dict

    def search(self, clfs, clfs_space, hops=DEF_HOPS, user_functions=DEF_USER_FUNCTIONS,  attention_configs=DEF_ATTENTION_CONFIGS):
        store = dict({})
        for clf in tqdm(clfs):
            clf_name = clf().__class__.__name__
            space = clfs_space[clf_name]
            store[clf_name] = dict({})
            for hop in tqdm(hops):
                best_search_dict = None
                best_val = float("-inf")
                for attention_config in tqdm(attention_configs):
                    for user_function in user_functions:
                        search_dict = self.search_hop_clf_attention_config(hop, clf, user_function, attention_config, space)
                        if search_dict["val_acc"] >= best_val:
                            best_val = search_dict["val_acc"]
                            best_search_dict = search_dict
                            best_search_dict["attention_config"] = attention_config
                        if hop == 0:
                            break
                    if hop == 0:
                            break
                store[clf_name][hop] = best_search_dict
        return store