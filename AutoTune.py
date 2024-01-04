from hyperopt import fmin, tpe, hp,STATUS_OK, SparkTrials, space_eval 
from sklearn.metrics import accuracy_score
from tqdm.notebook import tqdm
from EnsembleFramework import Framework
from torch.nn.functional import normalize
from sklearn.multioutput import MultiOutputClassifier

    
DEF_ATTENTION_CONFIGS= [None,{'inter_layer_normalize': False,
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
DEF_HOPS = [3, 5]
DEF_MAX_EVALS = 1000
def norm_user_function(kwargs):
    return  normalize(kwargs["original_features"] + kwargs["summed_neighbors"], p=2.0, dim = 1)
    
def user_function(kwargs):
    return  kwargs["original_features"] + kwargs["summed_neighbors"]
    
DEF_USER_FUNCTIONS = [norm_user_function, user_function]

class Data():
    def __init__(self, X, y, edge_index):
        self.X = X
        self.y = y
        self.edge_index = edge_index
        self.train = None
        self.val = None
        self.test = None
        self.X_train = None
        self.X_val = None
        self.X_test = None
    
    def set_train(self, train):
        self.train = train

    def set_test(self, test):
        self.test = test

    def set_val(self, val):
        self.val = val

    def set_X_train(self, X):
        self.X_train = X

    def set_X_val(self, X):
        self.X_val = X

    def set_X_test(self, X):
        self.X_test = X

class SparkTune():
    def __init__(self, data,clf, evals = 10, pred_metric = accuracy_score, pred_metric_kwargs = {}, multi_target_class = False):
        self.evals = evals
        self.data = data
        self.clf = clf
        self.pred_metric = pred_metric
        self.pred_metric_kwargs = pred_metric_kwargs
        self.multi_target_class = multi_target_class
        
    def objective(self, params):
        model = self.clf(**params)
        if self.multi_target_class:
            print("Multi class")
            model = MultiOutputClassifier(model, n_jobs=11)
        model.fit(self.data.X_train, self.data.y[self.data.train])
        y_pred = model.predict(self.data.X_val)
        score = self.pred_metric(self.data.y[self.data.val], y_pred, **self.pred_metric_kwargs)
        return {'loss': -score, 'status': STATUS_OK}
    
    def search(self, space):
        spark_trials = SparkTrials()
        best_params = fmin(self.objective, space, algo=tpe.suggest, max_evals=self.evals, trials=spark_trials)
        return best_params


class AutoSearch:
    def __init__(self, data_dict, max_evals = 200, multi_target_class = False, pred_metric= accuracy_score, pred_metric_kwargs = {}):
        self.data_dict = data_dict
        self.max_evals = max_evals
        self.multi_target_class = multi_target_class
        self.pred_metric = pred_metric
        self.pred_metric_kwargs = pred_metric_kwargs

    def get_data(self):
        dataset = self.data_dict
        data = Data(dataset["X"], dataset["y"], dataset["edge_index"])
        data.set_test(dataset["test"])
        data.set_val(dataset["val"])
        data.set_train(dataset["train"])
        return data

    def search_hop_clf_attention_config(self, hop, clf, user_function, attention_config, space):
        data:Data = self.get_data()
        framework = Framework([user_function], 
                         hops_list=[hop],
                         clfs=[],
                         gpu_idx=0,
                         handle_nan=0.0,
                        attention_configs=[attention_config])
        data.set_X_train(framework.get_features(data.X, data.edge_index, data.train)[0].cpu())
        data.set_X_val(framework.get_features(data.X, data.edge_index, data.val)[0].cpu())
        data.set_X_test(framework.get_features(data.X, data.edge_index, data.test)[0].cpu())
        
        sparkTune = SparkTune(data, clf, evals = self.max_evals, pred_metric = self.pred_metric, pred_metric_kwargs = self.pred_metric_kwargs,
                             multi_target_class = self.multi_target_class)
        params = sparkTune.search(space)
        
        params = space_eval(space, params)
    
        model = clf(**params)
        kwargs={}# {"eval_set":[(data.X_val, data.y[data.val])], "early_stopping_rounds":5} if model.__class__.__name__ == 'XGBClassifier' else {}
        if self.multi_target_class:
            print("Multi class")
            model = MultiOutputClassifier(model, n_jobs=11)
        print(model)
        model.fit(data.X_train,data.y[data.train],**kwargs)
        train_pred = model.predict(data.X_train)
        val_pred = model.predict(data.X_val)
        test_pred = model.predict(data.X_test)
        
        train_acc = self.pred_metric(data.y[data.train], train_pred, **self.pred_metric_kwargs)
        val_acc = self.pred_metric(data.y[data.val], val_pred, **self.pred_metric_kwargs)
        test_acc = self.pred_metric(data.y[data.test], test_pred, **self.pred_metric_kwargs)
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
                store[clf_name][hop] = best_search_dict
        return store