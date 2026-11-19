from .base import BaseModel


# Logistic Regression classifier
class LogisticRegression(BaseModel):
    # Stores learning rate, iterations, and regularization
    def __init__(self, lr=0.01, n_iter=1000, reg=None, lam=0.0):
        self.lr, self.n_iter, self.reg, self.lam = lr, n_iter, reg, lam

    # Learns weights with gradient descent on log-loss
    def fit(self, X, y):
        raise NotImplementedError

    # Predicts 1 when sigmoid(Xw + b) >= 0.5
    def predict(self, X):
        raise NotImplementedError
