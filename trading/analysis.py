import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc


class RocAnalysis(object):

    def __init__(self, y_true, y_pred, model_name):
        self.y_true = y_true
        self.y_pred = y_pred
        self.model_name = model_name
    
    def calculate(self):
        self.fpr_, self.tpr_, self.thresholds_ = roc_curve(self.y_true, self.y_pred)
        self.auc_ = auc(self.fpr_, self.tpr_)
    
    def plot(self, title=None):
        if not hasattr(self, 'auc_'):
            self.calculate()
        plt.plot(self.fpr_, self.tpr_, color='royalblue', label=f'{self.model_name} (AUC = {self.auc_:.4f})')
        plt.plot((0.0, 1.0), (0.0, 1.0), color='darkred', label='Random Classifier', linestyle='dashed')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.xlim((0.0, 1.0))
        plt.ylim((0.0, 1.0))

        if title:
            plt.title(title)
        
        plt.legend()
    
    def youden_j(self):
        return self.tpr_ - self.fpr_
    
    def gmean(self):
        return np.sqrt((1.0 - self.fpr_) * self.tpr_)
       
    def get_best_thr(self, criterion='gmean'):
        if not criterion in ('youden_j', 'gmean'):
            raise ValueError()
        values = getattr(self, criterion)()
        imax = np.argmax(values)
        return self.thresholds_[imax]
