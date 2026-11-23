from collections.abc import Mapping

import numpy as np

from .base import BaseModel


class KNN(BaseModel):
    def __init__(
        self, k=5, metric="euclidean", p=2, vote_weights="uniform", feature_weights=None, threshold=None
    ):
        self.set_params(
            k=k,
            metric=metric,
            p=p,
            vote_weights=vote_weights,
            feature_weights=feature_weights,
            threshold=threshold,
        )

    def set_params(self, **params):
        updated = {
            "k": getattr(self, "k", 5),
            "metric": getattr(self, "metric", "euclidean"),
            "p": getattr(self, "p", 2),
            "vote_weights": getattr(self, "vote_weights", "uniform"),
            "feature_weights": getattr(self, "feature_weights", None),
            "threshold": getattr(self, "threshold", None),
        }
        unknown = set(params) - set(updated)
        if unknown:
            raise ValueError(f"Unknown parameter(s): {', '.join(sorted(unknown))}")
        updated.update(params)

        k = updated["k"]
        if isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)) or k < 1:
            raise ValueError("k must be a positive integer.")

        metric = updated["metric"]
        valid_metrics = {"euclidean", "manhattan", "minkowski"}
        if not isinstance(metric, str) or metric.lower() not in valid_metrics:
            raise ValueError("metric must be 'euclidean', 'manhattan', or 'minkowski'.")

        p = updated["p"]
        if isinstance(p, (bool, np.bool_)) or not isinstance(
            p, (int, float, np.integer, np.floating)
        ):
            raise ValueError("p must be a finite number greater than or equal to 1.")
        if not np.isfinite(p) or p < 1:
            raise ValueError("p must be a finite number greater than or equal to 1.")

        vote_weights = updated["vote_weights"]
        if not isinstance(vote_weights, str) or vote_weights not in {
            "uniform",
            "inverse_square",
        }:
            raise ValueError("vote_weights must be 'uniform' or 'inverse_square'.")

        feature_weights = updated["feature_weights"]
        if feature_weights is not None:
            if not isinstance(feature_weights, Mapping):
                raise ValueError("feature_weights must map feature names to non-negative weights.")
            normalized_weights = {}
            for name, weight in feature_weights.items():
                if not isinstance(name, str) or not name:
                    raise ValueError("feature weight keys must be non-empty strings.")
                if isinstance(weight, (bool, np.bool_)) or not isinstance(
                    weight, (int, float, np.integer, np.floating)
                ):
                    raise ValueError("feature weights must be finite non-negative numbers.")
                if not np.isfinite(weight) or weight < 0:
                    raise ValueError("feature weights must be finite non-negative numbers.")
                normalized_weights[name] = float(weight)
            feature_weights = normalized_weights

        threshold = updated["threshold"]
        if threshold is not None:
            if isinstance(threshold, (bool, np.bool_)) or not isinstance(
                threshold, (int, float, np.integer, np.floating)
            ):
                raise ValueError("threshold must be None or a number between 0 and 1.")
            if not np.isfinite(threshold) or not 0 <= threshold <= 1:
                raise ValueError("threshold must be None or a number between 0 and 1.")

        self.k = int(k)
        self.metric = metric.lower()
        self.p = float(p)
        self.vote_weights = vote_weights
        self.feature_weights = feature_weights
        self.threshold = None if threshold is None else float(threshold)
        return self

    def fit(self, X, y):
        columns = getattr(X, "columns", None)
        self.feature_names_in_ = None if columns is None else tuple(str(name) for name in columns)
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

        self.classes_, self.y_encoded_ = np.unique(labels, return_inverse=True)
        if self.threshold is not None and set(self.classes_.tolist()) != {0, 1}:
            raise ValueError("threshold is only supported for binary labels 0 and 1.")
        self.n_features_in_ = features.shape[1]
        self.X_ = features.copy()
        self.feature_scale_ = np.ones(self.n_features_in_, dtype=float)
        if self.feature_weights:
            if self.feature_names_in_ is None:
                raise ValueError("named feature_weights require a pandas DataFrame input.")
            matched_groups = set()
            for index, name in enumerate(self.feature_names_in_):
                matches = [
                    group
                    for group in self.feature_weights
                    if name == group or name.startswith(f"{group}_")
                ]
                if matches:
                    group = max(matches, key=len)
                    self.feature_scale_[index] = self.feature_weights[group]
                    matched_groups.add(group)
            unmatched = set(self.feature_weights) - matched_groups
            if unmatched:
                raise ValueError(f"No input feature matches: {', '.join(sorted(unmatched))}")
        if not np.any(self.feature_scale_ > 0):
            raise ValueError("At least one feature must have a positive weight.")
        return self

    def predict_proba(self, X):
        if not hasattr(self, "X_"):
            raise ValueError("Call fit before predict.")
        if self.k > len(self.X_):
            raise ValueError("k cannot be greater than the number of training examples.")

        columns = getattr(X, "columns", None)
        if self.feature_weights and columns is None:
            raise ValueError("named feature_weights require a pandas DataFrame input.")
        if columns is not None and self.feature_names_in_ is not None:
            names = tuple(str(name) for name in columns)
            if names != self.feature_names_in_:
                raise ValueError("X must have the same feature names and order used during fit.")

        queries = self._as_features(X, "X")
        if queries.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X must have {self.n_features_in_} features; received {queries.shape[1]}."
            )

        probabilities = np.empty((len(queries), len(self.classes_)), dtype=float)
        if self.metric == "euclidean":
            train = self.X_ * self.feature_scale_
            train_squared_norms = np.einsum("ij,ij->i", train, train)
        else:
            train = self.X_
            train_squared_norms = None

        max_distance_elements = 16_000_000
        batch_size = max(
            1,
            min(128, max_distance_elements // (len(self.X_) * self.n_features_in_)),
        )
        for start in range(0, len(queries), batch_size):
            batch = queries[start:start + batch_size]
            if self.metric == "euclidean":
                weighted_batch = batch * self.feature_scale_
                query_squared_norms = np.einsum("ij,ij->i", weighted_batch, weighted_batch)
                distances = -2.0 * (weighted_batch @ train.T)
                distances += query_squared_norms[:, None]
                distances += train_squared_norms[None, :]
                np.maximum(distances, 0.0, out=distances)
                np.sqrt(distances, out=distances)
            else:
                differences = batch[:, None, :] - train[None, :, :]
                np.abs(differences, out=differences)
                differences *= self.feature_scale_
                if self.metric == "manhattan":
                    distances = differences.sum(axis=2)
                else:
                    np.power(differences, self.p, out=differences)
                    distances = np.power(differences.sum(axis=2), 1.0 / self.p)

            for offset, row in enumerate(distances):
                neighbors = self._nearest_indices(row)
                if self.vote_weights == "inverse_square":
                    neighbor_distances = row[neighbors]
                    exact = neighbor_distances == 0
                    vote_weights = (
                        exact.astype(float)
                        if exact.any()
                        else 1.0 / np.square(neighbor_distances)
                    )
                else:
                    vote_weights = np.ones(len(neighbors), dtype=float)
                votes = np.bincount(
                    self.y_encoded_[neighbors], weights=vote_weights, minlength=len(self.classes_)
                )
                probabilities[start + offset] = votes / votes.sum()
        return probabilities

    def predict(self, X):
        probabilities = self.predict_proba(X)
        if self.threshold is None:
            return self.classes_[np.argmax(probabilities, axis=1)]
        positive = np.flatnonzero(self.classes_ == 1)[0]
        return np.where(probabilities[:, positive] >= self.threshold, 1, 0).astype(
            self.classes_.dtype
        )

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
