from .base import BaseModel


# Bayesian Network classifier
class BayesNetwork(BaseModel):
    # Stores the bin count for discretizing continuous features
    def __init__(self, n_bins=5):
        self.n_bins = n_bins

    # Learns the network structure and conditional probability tables
    def fit(self, X, y):
        raise NotImplementedError

    # Predicts the class with the highest posterior probability
    def predict(self, X):
        raise NotImplementedError
