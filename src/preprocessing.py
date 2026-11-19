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


# Impute, engineer, cap, encode, and scale with statistics fit on X_train only
class Preprocessor:
    def __init__(self, zero_as_missing=(), num_strategy="median", cap=(), cap_factor=1.5, ordinal=None, scaler="standard"):
        self.zero_as_missing, self.num_strategy = list(zero_as_missing), num_strategy
        self.cap, self.cap_factor = list(cap), cap_factor
        self.ordinal, self.scaler = ordinal or {}, scaler

    # Learns fill values, IQR bounds, categories, and scaling stats from X
    def fit(self, X):
        X = self.zero_to_nan(X)
        num = X.select_dtypes("number").columns
        cat = X.columns.difference(num)
        self.fill = {**getattr(X[num], self.num_strategy)().to_dict(), **X[cat].mode().iloc[0].to_dict()}
        X = engineer(X.fillna(self.fill))

        q1, q3 = X[self.cap].quantile(0.25), X[self.cap].quantile(0.75)
        self.bounds = (q1 - self.cap_factor * (q3 - q1), q3 + self.cap_factor * (q3 - q1))
        X = self.clip(X)

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
        X = self.clip(engineer(self.zero_to_nan(X).fillna(self.fill)))
        X = self.encode(X).reindex(columns=self.columns, fill_value=0)
        return (X - self.center) / self.spread if self.scaler else X

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
            X[c] = X[c].map({v: i for i, v in enumerate(order)}).fillna(-1)
        for c, cats in self.onehot.items():
            X[c] = pd.Categorical(X[c], categories=cats)
        return pd.get_dummies(X, columns=list(self.onehot), dtype=float)
