from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
import numpy as np
from scripts.data_utils import get_connectome
from sklearn.metrics import confusion_matrix
#from scripts.gnn import *


#TODO graph model baseline


class LogRegPCA:
    def __init__(self, pca=True):
        self.pca = PCA() if pca else None
        self.model = LogisticRegression()
    
    def model_training(self, x, y):
        vecs = get_connectome(x).reshape((x.shape[0], -1))
        if self.pca is not None:
            vecs = self.pca.fit_transform(vecs)

        self.model.fit(vecs, y)
        acc = self.model.score(vecs, y)
        print('Accuracy on train:', round(acc, 3))

        return acc
    
    def model_testing(self, x, y):
        vecs = get_connectome(x).reshape((x.shape[0], -1))
        if self.pca is not None:
            vecs = self.pca.transform(vecs)

        y_pred = self.model.predict(vecs)
        acc = self.model.score(vecs, y)
        print('Accuracy on test:', round(acc, 3))
        cm = confusion_matrix(y, y_pred)

        return cm, acc
    




    










