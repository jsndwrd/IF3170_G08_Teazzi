from .base import BaseModel


# Support Vector Machine classifier
class SVM(BaseModel):
    # Stores C, kernel, learning rate, and iterations
    def __init__(self, C=1.0, kernel="linear", lr=0.001, n_iter=1000):
        self.C, self.kernel, self.lr, self.n_iter = C, kernel, lr, n_iter

    # Learns the maximum-margin decision boundary
    def fit(self, X, y):
        raise NotImplementedError

    # Predicts the class from the decision function sign
    def predict(self, X):
        raise NotImplementedError
