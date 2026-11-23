import numpy as np

from .base import BaseModel


# Decision Tree Learning classifier (CART-style binary splits on continuous features)
class DTL(BaseModel):
    # Stores split criterion and tree size limits
    def __init__(self, criterion="entropy", max_depth=None, min_samples_split=2):
        self.criterion, self.max_depth, self.min_samples_split = criterion, max_depth, min_samples_split

    # Grows the tree recursively from the training data
    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        self.classes_, y = np.unique(np.asarray(y), return_inverse=True)
        self.feature_importances_ = np.zeros(X.shape[1])
        self.tree_ = self._grow(X, y, 0)
        total = self.feature_importances_.sum()
        if total > 0:
            self.feature_importances_ /= total
        return self

    # Traverses the tree for each sample and returns the majority class of its leaf
    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]

    # Class proportions of the training rows in each sample's leaf
    def predict_proba(self, X):
        return np.array([self._walk(x) for x in np.asarray(X, dtype=float)])

    # Inner node is (feature, threshold, left, right) with x[feature] <= threshold going left; leaf is an array of class proportions
    def _grow(self, X, y, depth):
        counts = np.bincount(y, minlength=len(self.classes_))
        leaf = counts / len(y)
        if counts.max() == len(y) or len(y) < self.min_samples_split or (self.max_depth is not None and depth >= self.max_depth):
            return leaf
        split = self._best_split(X, y, counts)
        if split is None:
            return leaf
        gain, feature, threshold = split
        self.feature_importances_[feature] += gain * len(y)
        mask = X[:, feature] <= threshold
        return feature, threshold, self._grow(X[mask], y[mask], depth + 1), self._grow(X[~mask], y[~mask], depth + 1)

    # Impurity of every row of class counts
    def _impurity(self, counts):
        p = counts / counts.sum(axis=-1, keepdims=True)
        if self.criterion == "gini":
            return 1 - (p ** 2).sum(axis=-1)
        if self.criterion == "entropy":
            return -(p * np.log2(np.where(p > 0, p, 1))).sum(axis=-1)
        raise ValueError(f"unknown criterion {self.criterion}")

    # Best (gain, feature, threshold) over every midpoint between consecutive distinct sorted values
    def _best_split(self, X, y, counts):
        n, parent, onehot, best = len(y), self._impurity(counts), np.eye(len(counts))[y], None
        for feature in range(X.shape[1]):
            order = np.argsort(X[:, feature], kind="stable")
            xs = X[order, feature]
            cut = np.flatnonzero(xs[:-1] < xs[1:])
            if len(cut) == 0:
                continue
            left = np.cumsum(onehot[order], axis=0)[cut]
            right = counts - left
            n_left = cut + 1
            gain = parent - (n_left * self._impurity(left) + (n - n_left) * self._impurity(right)) / n
            i = gain.argmax()
            if best is None or gain[i] > best[0]:
                a, b = xs[cut[i]], xs[cut[i] + 1]
                mid = (a + b) / 2
                best = gain[i], feature, mid if mid < b else a  # adjacent floats can round the midpoint up to b
        return best if best is not None and best[0] > 0 else None

    def _walk(self, x):
        node = self.tree_
        while isinstance(node, tuple):
            feature, threshold, left, right = node
            node = left if x[feature] <= threshold else right
        return node
