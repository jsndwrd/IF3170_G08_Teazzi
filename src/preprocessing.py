from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import Settings


def load_data():
    read = lambda name: pd.read_csv(Settings.DATA_DIR / f"{name}.csv", dtype={Settings.ID: str})
    return read("train"), read("test")


def split(train):
    X, y = train.drop(columns=[Settings.ID, Settings.TARGET]), train[Settings.TARGET]
    return train_test_split(X, y, test_size=Settings.TEST_SIZE, stratify=y, random_state=Settings.SEED)


def drop_duplicates(X, y):
    keep = ~X.duplicated()
    return X[keep], y[keep]


# Adds derived features using only values from the same row
def engineer(df):
    df = df.copy()
    df["total_nights"] = df["weekend_nights"] + df["weekday_nights"]
    df["total_guests"] = df["num_adults"] + df["num_children"]
    return df


# Impute, engineer, select, cap, encode, and scale with statistics fit on X_train only
class Preprocessor:
    def __init__(self, zero_as_missing=(), num_strategy="median", drop=(), cap=(), cap_factor=1.5, ordinal=None, scaler="standard"):
        self.zero_as_missing, self.num_strategy, self.drop = list(zero_as_missing), num_strategy, list(drop)
        self.cap, self.cap_factor = list(cap), cap_factor
        self.ordinal, self.scaler = ordinal or {}, scaler

    # Learns fill values, IQR bounds, categories, and scaling stats from X
    def fit(self, X):
        X = self.zero_to_nan(X)
        num = X.select_dtypes("number").columns
        self.fill = {**getattr(X[num], self.num_strategy)().to_dict(), **X[X.columns.difference(num)].mode().iloc[0].to_dict()}
        X = engineer(X.fillna(self.fill)).drop(columns=self.drop)

        q1, q3 = X[self.cap].quantile(0.25), X[self.cap].quantile(0.75)
        self.bounds = (q1 - self.cap_factor * (q3 - q1), q3 + self.cap_factor * (q3 - q1))
        X = self.clip(X)

        cat = X.select_dtypes(exclude="number").columns
        self.onehot = {c: sorted(X[c].unique()) for c in cat if c not in self.ordinal}
        X = self.encode(X)
        self.columns = X.columns

        if self.scaler:
            center, spread = {
                "standard": lambda: (X.mean(), X.std()),
                "minmax": lambda: (X.min(), X.max() - X.min()),
                "robust": lambda: (X.median(), X.quantile(0.75) - X.quantile(0.25)),
            }[self.scaler]()
            self.center, self.spread = center, spread.replace(0, 1)
        return self

    # Applies the fitted steps to any split, keeping its index
    def transform(self, X):
        X = self.clip(engineer(self.zero_to_nan(X).fillna(self.fill)).drop(columns=self.drop))
        return self.scale(self.encode(X).reindex(columns=self.columns, fill_value=0))

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def zero_to_nan(self, X):
        X = X.copy()
        X[self.zero_as_missing] = X[self.zero_as_missing].replace(0, np.nan)
        return X

    def clip(self, X):
        if self.cap:
            X[self.cap] = X[self.cap].clip(*self.bounds, axis=1)
        return X

    def encode(self, X):
        X = X.copy()
        for c, order in self.ordinal.items():
            if c in X:
                X[c] = X[c].map({v: i for i, v in enumerate(order)}).fillna(-1)
        for c, cats in self.onehot.items():
            X[c] = pd.Categorical(X[c], categories=cats)
        return pd.get_dummies(X, columns=list(self.onehot), dtype=float)

    def scale(self, X):
        return (X - self.center) / self.spread if self.scaler else X


# PCA from the eigenvectors of the covariance matrix
class PCA:
    def __init__(self, n):
        self.n = n

    def fit(self, X):
        self.mean = X.mean(axis=0)
        eigvals, eigvecs = np.linalg.eigh(np.cov(X - self.mean, rowvar=False))
        order = np.argsort(eigvals)[::-1]
        self.explained_variance_ratio = eigvals[order] / eigvals.sum()
        k = int(np.searchsorted(np.cumsum(self.explained_variance_ratio), self.n) + 1) if self.n < 1 else int(self.n)
        self.components = eigvecs[:, order[:k]]
        return self

    def transform(self, X):
        return (X - self.mean) @ self.components


class Pipeline:
    # select keeps the top-k features ranked by the model itself, pca is passed to PCA
    def __init__(self, model, select=None, pca=None):
        self.model, self.select, self.pca = model, select, pca

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        self.keep = np.arange(X.shape[1])
        if self.select:
            self.keep = np.argsort(self.feature_scores(X, y))[::-1][:self.select]
        self.reducer = PCA(self.pca).fit(X[:, self.keep]) if self.pca else None
        self.model.fit(self.reduce(X), y)
        return self

    def predict(self, X):
        return self.model.predict(self.reduce(X))

    # Keeps the selected columns and applies PCA if enabled
    def reduce(self, X):
        X = np.asarray(X, dtype=float)[:, self.keep]
        return self.reducer.transform(X) if self.reducer else X

    def feature_scores(self, X, y):
        fitted = deepcopy(self.model).fit(X, y)
        scores = getattr(fitted, "feature_importances_", None)
        return np.abs(np.ravel(fitted.coef_)) if scores is None else scores

    def set_params(self, **params):
        for key, value in params.items():
            if key in ("select", "pca"):
                setattr(self, key, value)
            else:
                self.model.set_params(**{key: value})
        return self
