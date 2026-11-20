from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

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


def arrival_date(df):
    first = pd.to_datetime(dict(year=df["arrival_year"], month=df["arrival_month"], day=1))
    return first + pd.to_timedelta(np.minimum(df["arrival_day"], first.dt.days_in_month) - 1, unit="D")


# Adds derived features using only values from the same row
def engineer(df):
    df = df.copy()
    df["total_nights"] = df["weekend_nights"] + df["weekday_nights"]
    df["total_guests"] = df["num_adults"] + df["num_children"]
    day_of_year = arrival_date(df).dt.dayofyear
    df["month_sin"], df["month_cos"] = np.sin(2 * np.pi * df["arrival_month"] / 12), np.cos(2 * np.pi * df["arrival_month"] / 12)
    df["doy_sin"], df["doy_cos"] = np.sin(2 * np.pi * day_of_year / 365.25), np.cos(2 * np.pi * day_of_year / 365.25)
    df["lead_x_price"] = np.log1p(df["days_before_arrival"]) * df["nightly_rate_idr"] / 1e6
    return df


# Date stats, impute, engineer, select, cap, encode, and scale with statistics fit on X_train only
class Preprocessor:
    def __init__(self, zero_as_missing=(), num_strategy="median", drop=(), cap=(), cap_factor=1.5, ordinal=None, scaler="standard", smoothing=20):
        self.zero_as_missing, self.num_strategy, self.drop = list(zero_as_missing), num_strategy, list(drop)
        self.cap, self.cap_factor = list(cap), cap_factor
        self.ordinal, self.scaler, self.smoothing = ordinal or {}, scaler, smoothing

    # Learns date stats, fill values, IQR bounds, categories, and scaling stats from X and y
    def fit(self, X, y):
        self.fit_dates(X, y)
        X = self.zero_to_nan(self.add_dates(X, self.train_rate))
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
    def transform(self, X, date_rate=None):
        X = self.zero_to_nan(self.add_dates(X, date_rate))
        X = self.clip(engineer(X.fillna(self.fill)).drop(columns=self.drop))
        return self.scale(self.encode(X).reindex(columns=self.columns, fill_value=0))

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X, self.train_rate)

    def fit_dates(self, X, y):
        self.price_by = X.groupby(["room_category", "arrival_month"])["nightly_rate_idr"].median()
        self.lead_by = X.groupby(["booking_channel", "arrival_month"])["days_before_arrival"].median()
        date, self.prior = arrival_date(X), y.mean()
        self.date_stats, self.date_count = self.rate_table(date, y), date.value_counts()
        self.train_rate = np.zeros(len(X))
        for tr, va in StratifiedKFold(5, shuffle=True, random_state=Settings.SEED).split(X, y):
            self.train_rate[va] = self.date_rate(self.rate_table(date.iloc[tr], y.iloc[tr]), date.iloc[va])

    def rate_table(self, date, y):
        return pd.DataFrame({"date": date.values, "y": np.asarray(y)}).groupby("date")["y"].agg(["sum", "count"])

    def date_rate(self, table, date):
        s = table.reindex(date.values)
        return ((s["sum"].fillna(0) + self.smoothing * self.prior) / (s["count"].fillna(0) + self.smoothing)).values

    def add_dates(self, X, rate=None):
        X = X.copy()
        for col, by, table in (("nightly_rate_idr", ["room_category", "arrival_month"], self.price_by),
                               ("days_before_arrival", ["booking_channel", "arrival_month"], self.lead_by)):
            X[col] = X[col].fillna(pd.Series(table.reindex(pd.MultiIndex.from_frame(X[by])).values, index=X.index))
        date = arrival_date(X)
        X["date_rate"] = self.date_rate(self.date_stats, date) if rate is None else rate
        X["date_count"] = np.log1p(date.map(self.date_count).fillna(0).values)
        return X

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
