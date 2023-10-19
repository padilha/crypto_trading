import numpy as np
import multiprocessing as mp
# https://pythonspeed.com/articles/python-multiprocessing/
# mp.set_start_method('spawn')
from sklearn.metrics import roc_curve, auc


def _fit_and_validate(model, params, X_train, y_train, X_valid, y_valid):
    model.set_params(**params)
    model.fit(X_train, y_train)
    if hasattr(model, 'predict_proba'):
        y_pred = model.predict_proba(X_valid)[:, 1]
    else:
        y_pred = model.decision_function(X_valid)
    fpr, tpr, _ = roc_curve(y_valid, y_pred)
    return auc(fpr, tpr)


class GridSearch(object):
    
    def __init__(self, model, param_grid, n_jobs):
        self.model = model
        self.param_grid = param_grid
        self.n_jobs = n_jobs
    
    def fit(self, X_train, y_train, X_valid, y_valid):
        param_grid = list(self.param_grid)
        func_params = [(self.model, params, X_train, y_train, X_valid, y_valid) for params in param_grid]
        chunksize = len(func_params) // self.n_jobs if len(func_params) % self.n_jobs == 0 else len(func_params) // self.n_jobs + 1
        
        pool = mp.Pool(processes=self.n_jobs)
        results = pool.starmap(_fit_and_validate, func_params, chunksize=chunksize)
        pool.close()
        pool.join()
        
        i_best = np.argmax(results)
        self.model.set_params(**param_grid[i_best])
        self.model.fit(X_train, y_train)
        return self.model
