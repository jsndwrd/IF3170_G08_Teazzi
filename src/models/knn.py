import numpy as np

from .base import BaseModel


class KNN(BaseModel):
    def __init__(self, k=5, metric="euclidean", p=2):
        self.set_params(k=k, metric=metric, p=p)

    def set_params(self, **params):
        updated = {
            "k": getattr(self, "k", 5),
            "metric": getattr(self, "metric", "euclidean"),
            "p": getattr(self, "p", 2),
        }
        updated.update(params)

        k = updated["k"]
        if isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)) or k < 1:
            raise ValueError("k must be a positive integer.")

        metric = updated["metric"]
        valid_metrics = {"euclidean", "manhattan", "minkowski"}
        if not isinstance(metric, str) or metric.lower() not in valid_metrics:
            raise ValueError("metric must be 'euclidean', 'manhattan', or 'minkowski'.")

        p = updated["p"]
        if isinstance(p, (bool, np.bool_)) or not isinstance(p, (int, float, np.integer, np.floating)):
            raise ValueError("p must be a finite number greater than or equal to 1.")
        if not np.isfinite(p) or p < 1:
            raise ValueError("p must be a finite number greater than or equal to 1.")

        self.k = int(k)
        self.metric = metric.lower()
        self.p = float(p)
        return self

    def fit(self, X, y):
        features = self._as_features(X, "X")
        labels = np.asarray(y)
        if labels.ndim != 1:
            raise ValueError("y must be a one-dimensional array of class labels.")
        if len(features) != len(labels):
            raise ValueError("X and y must contain the same number of rows.")
        if len(labels) == 0:
            raise ValueError("At least one training example is required.")
        if self.k > len(labels):
            raise ValueError("k cannot be greater than the number of training examples.")
        if labels.dtype.kind in "fc" and not np.isfinite(labels).all():
            raise ValueError("y cannot contain missing or non-finite labels.")

        self.X_ = features.copy()
        self.classes_, self.y_encoded_ = np.unique(labels, return_inverse=True)
        self.n_features_in_ = self.X_.shape[1]
        return self

    def predict(self, X):
        if not hasattr(self, "X_"):
            raise ValueError("Call fit before predict.")

        queries = self._as_features(X, "X")
        if queries.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X must have {self.n_features_in_} features; received {queries.shape[1]}."
            )
        if self.k > len(self.X_):
            raise ValueError("k cannot be greater than the number of training examples.")

        predictions = np.empty(len(queries), dtype=self.classes_.dtype)
        train_squared_norms = None
        if self.metric == "euclidean":
            train_squared_norms = np.einsum("ij,ij->i", self.X_, self.X_)

        batch_size = 128
        for start in range(0, len(queries), batch_size):
            batch = queries[start:start + batch_size]
            if self.metric == "euclidean":
                query_squared_norms = np.einsum("ij,ij->i", batch, batch)
                distances = -2.0 * (batch @ self.X_.T)
                distances += query_squared_norms[:, None]
                distances += train_squared_norms[None, :]
                np.maximum(distances, 0.0, out=distances)
            else:
                distances = np.empty((len(batch), len(self.X_)), dtype=float)
                power = 1.0 if self.metric == "manhattan" else self.p
                for offset, query in enumerate(batch):
                    differences = np.abs(self.X_ - query)
                    if power == 1.0:
                        distances[offset] = differences.sum(axis=1)
                    else:
                        distances[offset] = np.power(differences, power).sum(axis=1)

            for offset, row in enumerate(distances):
                neighbors = self._nearest_indices(row)
                votes = np.bincount(self.y_encoded_[neighbors], minlength=len(self.classes_))
                predictions[start + offset] = self.classes_[np.argmax(votes)]

        return predictions

    @staticmethod
    def _as_features(values, name):
        try:
            array = np.asarray(values, dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name} must contain numeric, finite features.") from error
        if array.ndim != 2:
            raise ValueError(f"{name} must be a two-dimensional array.")
        if array.shape[1] == 0:
            raise ValueError(f"{name} must contain at least one feature.")
        if not np.isfinite(array).all():
            raise ValueError(f"{name} must contain numeric, finite features.")
        return array

    def _nearest_indices(self, distances):
        candidates = np.argpartition(distances, self.k - 1)[:self.k]
        boundary = distances[candidates].max()

        closer = np.flatnonzero(distances < boundary)
        tied_at_boundary = np.flatnonzero(distances == boundary)
        selected = np.concatenate((closer, tied_at_boundary[:self.k - len(closer)]))

        order = np.lexsort((selected, distances[selected]))
        return selected[order]
