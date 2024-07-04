from sklearn.model_selection import KFold, StratifiedKFold
import numpy as np
from tqdm.notebook import tqdm
from hyperopt import fmin, tpe, hp,STATUS_OK


def index_to_mask(rows, index_array):
    mask_array = np.zeros(rows, dtype=int)
    mask_array[index_array] = 1
    return mask_array.astype(np.bool_)


class NestedCV:
    def __init__(self, model, k_outer, k_inner, train_fun, eval_fun, max_evals, parallelism, minimalize):
        self.k_outer = StratifiedKFold(n_splits=self.k_outer)
        self.k_inner = StratifiedKFold(n_splits=self.k_inner)
        self.train_fun = train_fun
        self.evaluate_fun = eval_fun
        self.outer_scores = np.zeros(k_outer)
        self.inner_scores = np.zeros((k_outer, k_inner))
        self.minimalize = minimalize
        self.parallelism = parallelism
        self.max_evals = max_evals

class NestedInductiveCV:
    pass
    
class NestedTransductiveCV(NestedCV):
    def __init__(self, data, k_outer, k_inner, train_fun, eval_fun,max_evals = 100, parallelism = 1, minimalize = True):
        self.cv_args = [k_outer, k_inner, train_fun, eval_fun,max_evals, parallelism, minimalize]
        super().__init__(*self.cv_args)
        self.data = data
        pass

    def outer_cv(self, space):        
        for outer_i, (train_index, test_index) in tqdm(enumerate(self.kf_outer.split(self.graph.X, self.graph.y))):
            inner_transd_cv = InnerTransductiveCV(train_index, outer_i, [self.data, *self.cv_args])
            fitted_model = inner_transd_cv.hyperparam_tuning(space)
            self.outer_scores[outer_i] = self.evaluate_fun(fitted_model, self.data, inner_test_mask)
        return self.outer_scores

class InnerTransductiveCV(NestedTransductiveCV):
    def __init__(self, train_index, outer_i, *args):
        super().__init__(*args)
        self.train_index = train_index
        self.outer_i = outer_i

    def inner_cv(self, hyperparameters):    
        outer_train_mask = index_to_mask(self.data.X.shape[0], self.train_index)
        for inner_i, (inner_train_index, inner_test_index) in enumerate(self.kf_inner.split(self.data.X[self.train_index], self.data.y[self.train_index])): 
            inner_train_mask = index_to_mask(self.data.X.shape[0], self.outer_train_index[inner_train_index])
            inner_test_mask = index_to_mask(self.data.X.shape[0], self.outer_train_index[inner_test_index])
            fitted_model = self.train_fun(self.data, inner_train_mask, hyperparameters)
            self.inner_scores[self.outer_i, inner_i] = self.evaluate_fun(fitted_model, self.data, inner_test_mask)
        score = -self.inner_scores.mean() if self.minimalize else self.inner_scores.mean()
        return {'loss': score, 'status': STATUS_OK}

    def hyperparam_tuning(self, space):
        spark_trials = SparkTrials(parallelism = self.parallelism)
        best_params = fmin(self.inner_cv, space, algo=tpe.suggest, max_evals=self.max_evals, trials=spark_trials, verbose = False)
        best_params = space_eval(space, best_params)
        fitted_model = self.train_fun(self.data, index_to_mask(self.data.X.shape[0], self.train_index), best_params)
        return fitted_model
            