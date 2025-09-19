from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
import numpy as np


class Regressor_LI:
    def __init__(self):
        self.pca = PCA(n_components=100)
        self.model_X = LinearRegression()
        self.model_Y = LinearRegression()


    def Train(self, X, Y):
        PCA_X = self.pca.fit_transform(X)
        #Fit rotation first
        self.model_X.fit(PCA_X, Y[:,0])
        self.model_Y.fit(PCA_X, Y[:,1])

    def Predict(self, X):
        target = np.array([0,0])
        pca_x = self.pca.transform(np.expand_dims(X, axis=0))
        target[0] = np.round(self.model_X.predict(pca_x)[0], 2)
        target[1] = np.round(self.model_Y.predict(pca_x)[0], 2)
        return target