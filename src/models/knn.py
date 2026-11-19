from .base import BaseModel


# K-Nearest Neighbors classifier
class KNN(BaseModel):
    # Stores k, distance metric, and Minkowski p
    def __init__(self, k=5, metric="euclidean", p=2):
        self.k, self.metric, self.p = k, metric, p

    # Stores the training data
    def fit(self, X, y):
        raise NotImplementedError

    # Majority vote among the k nearest neighbors
    def predict(self, X):
        raise NotImplementedError
