from copy import deepcopy

import numpy as np
import pandas as pd

from config import Settings


def load_data():
    read = lambda name: pd.read_csv(Settings.DATA_DIR / f"{name}.csv", dtype={Settings.ID: str})
    return read("train"), read("test")


# (fit, eval) row positions for each fold, every class shuffled and dealt round-robin so each fold keeps the class ratio
def stratified_folds(y, n_splits=Settings.N_FOLDS, seed=Settings.SEED):
    y, rng = np.asarray(y), np.random.default_rng(seed)
    fold = np.empty(len(y), dtype=int)
    for c in np.unique(y):
        rows = rng.permutation(np.flatnonzero(y == c))
        fold[rows] = np.arange(len(rows)) % n_splits
    return [(np.flatnonzero(fold != k), np.flatnonzero(fold == k)) for k in range(n_splits)]


# Stratified holdout, one fold of round(1 / TEST_SIZE) becomes the validation set
def split(train):
    X, y = train.drop(columns=[Settings.ID, Settings.TARGET]), train[Settings.TARGET]
    fit_rows, val_rows = stratified_folds(y, round(1 / Settings.TEST_SIZE))[0]
    return X.iloc[fit_rows], X.iloc[val_rows], y.iloc[fit_rows], y.iloc[val_rows]


def drop_duplicates(X, y):
    keep = ~X.duplicated()
    return X[keep], y[keep]


NEAR_KEY = ["arrival_year", "arrival_month", "weekend_nights", "weekday_nights", "booking_channel",
            "is_returning_guest", "prior_cancellations", "prior_completed_bookings"]
NEAR_EQUAL = ["room_category", "meal_package", "num_adults", "num_children", "special_requests", "needs_parking"]


# For each query row, positions of ref rows that look like the same group booking: equal on NEAR_KEY, arrival day, lead time and
# log price within tol = (days, days, log ratio), and equal on NEAR_EQUAL wherever both values are known (the data has jittered and blanked cells)
def near_duplicates(query, ref, exclude_self=False, tol=(0, 10, 0.06)):
    day_tol, lead_tol, price_tol = tol
    query, ref = query.reset_index(drop=True), ref.reset_index(drop=True)
    out, ref_groups = [np.array([], dtype=int)] * len(query), ref.groupby(NEAR_KEY).indices
    for key, qi in query.groupby(NEAR_KEY).indices.items():
        ri = ref_groups.get(key)
        if ri is None:
            continue
        q, r = query.iloc[qi], ref.iloc[ri]
        gap = lambda c: np.abs(q[c].values.astype(float)[:, None] - r[c].values.astype(float)[None])
        ok = (gap("arrival_day") <= day_tol) & ~(gap("days_before_arrival") > lead_tol)  # NaN gaps pass
        log_price = lambda df: np.log(df["nightly_rate_idr"].where(df["nightly_rate_idr"] > 0).values.astype(float))
        ok &= ~(np.abs(log_price(q)[:, None] - log_price(r)[None]) > price_tol)
        for c in NEAR_EQUAL:
            a, b = q[c].to_numpy(dtype=object)[:, None], r[c].to_numpy(dtype=object)[None]
            ok &= (a == b) | pd.isna(a) | pd.isna(b)
        if exclude_self:
            ok &= qi[:, None] != ri[None]
        for row, mask in zip(qi, ok):
            out[row] = ri[mask]
    return out


def arrival_date(df):
    first = pd.to_datetime(dict(year=df["arrival_year"], month=df["arrival_month"], day=1))
    return first + pd.to_timedelta(np.minimum(df["arrival_day"], first.dt.days_in_month) - 1, unit="D")


LEAD_BANDS = [-1, 7, 30, 60, 90, 150, 240, 1000]


# String key per row from raw columns and the derived parts 'date' (arrival date), 'week' (arrival ISO week),
# 'lead_band' (days_before_arrival in LEAD_BANDS, missing is its own band), 'lead20' (20-day lead bins), 'price10' (10%-wide log price bins) and 'nights' (weekend + weekday nights)
def te_key(X, parts):
    derived = {"date": lambda: arrival_date(X).dt.strftime("%Y-%m-%d"), "week": lambda: arrival_date(X).dt.strftime("%G-%V"),
               "lead_band": lambda: pd.cut(X["days_before_arrival"], LEAD_BANDS).astype(str),
               "lead20": lambda: (X["days_before_arrival"] // 20).astype(str),
               "price10": lambda: np.floor(10 * np.log(X["nightly_rate_idr"].where(X["nightly_rate_idr"] > 0))).astype(str),
               "nights": lambda: X["weekend_nights"].astype(str) + "+" + X["weekday_nights"].astype(str)}
    columns = [(derived[p]() if p in derived else X[p]).astype(str).fillna("nan") for p in parts]
    return pd.Series(["|".join(v) for v in zip(*columns)], index=X.index)


# Adds derived features using only values from the same row
def engineer(df):
    df = df.copy()
    df["total_nights"] = df["weekend_nights"] + df["weekday_nights"]
    df["total_guests"] = df["num_adults"] + df["num_children"]
    date = arrival_date(df)
    day_of_year = date.dt.dayofyear
    df["month_sin"], df["month_cos"] = np.sin(2 * np.pi * df["arrival_month"] / 12), np.cos(2 * np.pi * df["arrival_month"] / 12)
    df["doy_sin"], df["doy_cos"] = np.sin(2 * np.pi * day_of_year / 365.25), np.cos(2 * np.pi * day_of_year / 365.25)
    df["lead_x_price"] = np.log1p(df["days_before_arrival"]) * df["nightly_rate_idr"] / 1e6
    booked = date - pd.to_timedelta(df["days_before_arrival"], unit="D")
    df["arrival_weekday"], df["booking_month"], df["booking_weekday"] = date.dt.dayofweek, booked.dt.month, booked.dt.dayofweek
    df["price_per_guest"] = df["nightly_rate_idr"] / df["total_guests"].clip(lower=1)
    return df


# Date stats, impute, engineer, select, cap, encode, and scale with statistics fit on X_train only
class Preprocessor:
    # flag_missing adds a 0/1 <column>_missing feature before imputation, binned adds a one-hot <column>_bin of n_bins quantile bins,
    # near_dup adds near_count (training rows that look like the same group booking) and near_rate (their cancel rate, smoothed by
    # near_smoothing) for each tolerance in near_tols; the second tolerance gets near2_count and near2_rate, and so on,
    # target_encode lists te_key parts, each adding te_<parts> (cancel rate of rows with the same key, smoothed by te_smoothing);
    # te_counts also adds te_<parts>_n, the log count of training rows behind that rate (the row itself excluded)
    def __init__(self, zero_as_missing=(), num_strategy="median", drop=(), cap=(), cap_factor=1.5, ordinal=None, scaler="standard", smoothing=20,
                 flag_missing=(), binned=(), n_bins=10, near_dup=False, near_smoothing=1.0, near_tols=((0, 10, 0.06),),
                 target_encode=(), te_smoothing=5.0, te_counts=False):
        self.zero_as_missing, self.num_strategy, self.drop = list(zero_as_missing), num_strategy, list(drop)
        self.flag_missing, self.binned, self.n_bins = list(flag_missing), list(binned), n_bins
        self.near_dup, self.near_smoothing, self.near_tols = near_dup, near_smoothing, [tuple(t) for t in near_tols]
        self.target_encode, self.te_smoothing, self.te_counts = [tuple(k) for k in target_encode], te_smoothing, te_counts
        self.cap, self.cap_factor = list(cap), cap_factor
        self.ordinal, self.scaler, self.smoothing = ordinal or {}, scaler, smoothing

    # Learns date stats, fill values, IQR bounds, categories, and scaling stats from X and y
    def fit(self, X, y):
        self.fit_dates(X, y)
        X = self.zero_to_nan(self.add_dates(X, self.oof))
        num = X.select_dtypes("number").columns
        self.fill = {**getattr(X[num], self.num_strategy)().to_dict(), **X[X.columns.difference(num)].mode().iloc[0].to_dict()}
        X = engineer(X.fillna(self.fill)).drop(columns=self.drop)

        q1, q3 = X[self.cap].quantile(0.25), X[self.cap].quantile(0.75)
        self.bounds = (q1 - self.cap_factor * (q3 - q1), q3 + self.cap_factor * (q3 - q1))
        X = self.clip(X)
        self.bin_edges = {c: np.unique(X[c].quantile(np.linspace(0, 1, self.n_bins + 1)[1:-1])) for c in self.binned}
        X = self.bin(X)

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

    # Applies the fitted steps to any split, keeping its index; oof holds the out-of-fold target stats of the training rows
    def transform(self, X, oof=None):
        X = self.zero_to_nan(self.add_dates(X, oof))
        X = self.bin(self.clip(engineer(X.fillna(self.fill)).drop(columns=self.drop)))
        return self.scale(self.encode(X).reindex(columns=self.columns, fill_value=0))

    # Bin labels are strings so encode one-hots them like any other category
    def bin(self, X):
        X = X.copy()
        for c, edges in self.bin_edges.items():
            X[f"{c}_bin"] = "b" + pd.Series(np.searchsorted(edges, X[c], side="right"), index=X.index).astype(str)
        return X

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X, self.oof)

    # Target stats of the training rows come from the other folds only, so a row never sees its own label
    def fit_dates(self, X, y):
        self.price_by = X.groupby(["room_category", "arrival_month"])["nightly_rate_idr"].median()
        self.lead_by = X.groupby(["booking_channel", "arrival_month"])["days_before_arrival"].median()
        date, self.prior = arrival_date(X), y.mean()
        self.date_stats, self.date_count = self.rate_table(date, y), date.value_counts()
        folds, fold = stratified_folds(y), np.empty(len(y), dtype=int)
        self.oof = {"date_rate": np.zeros(len(X))}
        for k, (tr, va) in enumerate(folds):
            fold[va] = k
            self.oof["date_rate"][va] = self.date_rate(self.rate_table(date.iloc[tr], y.iloc[tr]), date.iloc[va])
        self.te_tables = {}
        for parts in self.target_encode:
            name, key = "te_" + "_".join(parts), te_key(X, parts)
            self.te_tables[name], self.oof[name] = self.rate_table(key, y), np.zeros(len(X))
            self.oof[name + "_n"] = np.log1p(self.te_tables[name]["count"].reindex(key.values).values - 1)
            for tr, va in folds:
                self.oof[name][va] = self.date_rate(self.rate_table(key.iloc[tr], y.iloc[tr]), key.iloc[va], self.te_smoothing)
        if self.near_dup:
            self.near_X, self.near_y = X.reset_index(drop=True), np.asarray(y)
            for name, tol in self.near_names():
                neighbours = near_duplicates(X, X, exclude_self=True, tol=tol)
                self.oof[f"{name}_count"] = np.array([len(n) for n in neighbours])
                self.oof[f"{name}_rate"] = np.array([self.near_rate(n[fold[n] != fold[i]]) for i, n in enumerate(neighbours)])

    def near_names(self):
        return [("near" if k == 0 else f"near{k + 1}", tol) for k, tol in enumerate(self.near_tols)]

    def near_rate(self, neighbours):
        return (self.near_y[neighbours].sum() + self.near_smoothing * self.prior) / (len(neighbours) + self.near_smoothing)

    def rate_table(self, date, y):
        return pd.DataFrame({"date": date.values, "y": np.asarray(y)}).groupby("date")["y"].agg(["sum", "count"])

    def date_rate(self, table, date, smoothing=None):
        s, a = table.reindex(date.values), self.smoothing if smoothing is None else smoothing
        return ((s["sum"].fillna(0) + a * self.prior) / (s["count"].fillna(0) + a)).values

    # Near-duplicate stats on the raw values, missing flags, group-median fill of price and lead time,
    # each one's ratio to its group median, and arrival-date stats
    def add_dates(self, X, oof=None):
        X = X.copy()
        for parts in self.target_encode:
            name = "te_" + "_".join(parts)
            X[name] = self.date_rate(self.te_tables[name], te_key(X, parts), self.te_smoothing) if oof is None else oof[name]
            if self.te_counts:
                X[name + "_n"] = np.log1p(self.te_tables[name]["count"].reindex(te_key(X, parts).values).fillna(0).values) if oof is None else oof[name + "_n"]
        for name, tol in self.near_names() if self.near_dup else []:
            if oof is None:
                neighbours = near_duplicates(X, self.near_X, tol=tol)
                X[f"{name}_count"], X[f"{name}_rate"] = [len(n) for n in neighbours], [self.near_rate(n) for n in neighbours]
            else:
                X[f"{name}_count"], X[f"{name}_rate"] = oof[f"{name}_count"], oof[f"{name}_rate"]
        for c in self.flag_missing:
            X[f"{c}_missing"] = X[c].isna().astype(float)
        for col, by, table in (("nightly_rate_idr", ["room_category", "arrival_month"], self.price_by),
                               ("days_before_arrival", ["booking_channel", "arrival_month"], self.lead_by)):
            group = pd.Series(table.reindex(pd.MultiIndex.from_frame(X[by])).values, index=X.index)
            X[col] = X[col].fillna(group)
            X[f"{col}_ratio"] = X[col] / group.replace(0, np.nan)
        date = arrival_date(X)
        X["date_rate"] = self.date_rate(self.date_stats, date) if oof is None else oof["date_rate"]
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
    # select keeps the top-k features ranked by ranker (default: the model itself), pca is passed to PCA
    def __init__(self, model, select=None, pca=None, ranker=None):
        self.model, self.select, self.pca, self.ranker = model, select, pca, ranker

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
        fitted = deepcopy(self.ranker or self.model).fit(X, y)
        scores = getattr(fitted, "feature_importances_", None)
        return np.abs(np.ravel(fitted.coef_)) if scores is None else scores

    def set_params(self, **params):
        for key, value in params.items():
            if key in ("select", "pca", "ranker"):
                setattr(self, key, value)
            else:
                self.model.set_params(**{key: value})
        return self
