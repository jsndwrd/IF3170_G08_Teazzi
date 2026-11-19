from .base import BaseModel


# Decision Tree Learning classifier
class DTL(BaseModel):
    # Stores split criterion and tree size limits
    def __init__(self, criterion="entropy", max_depth=None, min_samples_split=2):
        self.criterion, self.max_depth, self.min_samples_split = criterion, max_depth, min_samples_split

    # Grows the tree recursively from the training data
    def fit(self, X, y):
        raise NotImplementedError

    # Traverses the tree for each sample
    def predict(self, X):
        raise NotImplementedError
