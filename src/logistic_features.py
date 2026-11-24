import numpy as np
import pandas as pd


class LogisticFeatureEngineer:
    LEAD_TIME_EDGES = [-1, 3, 7, 14, 30, 60, 90, 180, 365, np.inf]
    COUNT_COLUMNS = (
        "num_adults",
        "num_children",
        "weekend_nights",
        "weekday_nights",
        "special_requests",
        "prior_cancellations",
        "prior_completed_bookings",
    )

    def __init__(self, rate_bins=10):
        self.rate_bins = rate_bins

    def fit(self, X):
        if not isinstance(self.rate_bins, int) or self.rate_bins < 1:
            raise ValueError("rate_bins must be a positive integer")
        if "nightly_rate_idr" not in X:
            raise ValueError("X must contain 'nightly_rate_idr'")

        rates = pd.to_numeric(X["nightly_rate_idr"], errors="coerce")
        rates = rates.dropna()
        if rates.empty:
            self.rate_edges_ = np.array([-np.inf, np.inf])
        else:
            edges = pd.qcut(
                rates,
                q=self.rate_bins,
                labels=False,
                retbins=True,
                duplicates="drop",
            )[1]
            if len(edges) < 2:
                self.rate_edges_ = np.array([-np.inf, np.inf])
            else:
                edges = edges.copy()
                edges[0], edges[-1] = -np.inf, np.inf
                self.rate_edges_ = edges
        self.feature_names_in_ = tuple(X.columns)
        return self

    def transform(self, X):
        if not hasattr(self, "rate_edges_"):
            raise ValueError("feature engineer is not fitted; call fit(X) first")
        missing = set(self.feature_names_in_) - set(X.columns)
        if missing:
            raise ValueError(f"X is missing columns: {sorted(missing)}")

        df = X.copy()
        month = pd.to_numeric(df["arrival_month"], errors="coerce")
        day = pd.to_numeric(df["arrival_day"], errors="coerce")
        df["arrival_month_sin"] = np.sin(2 * np.pi * (month - 1) / 12)
        df["arrival_month_cos"] = np.cos(2 * np.pi * (month - 1) / 12)
        df["arrival_day_sin"] = np.sin(2 * np.pi * (day - 1) / 31)
        df["arrival_day_cos"] = np.cos(2 * np.pi * (day - 1) / 31)
        df = df.drop(columns=["arrival_month", "arrival_day"])

        for column in ("days_before_arrival", "nightly_rate_idr"):
            values = pd.to_numeric(df[column], errors="coerce").clip(lower=0)
            df[f"log_{column}"] = np.log1p(values)

        lead_time = pd.to_numeric(df["days_before_arrival"], errors="coerce")
        df["lead_time_bucket"] = (
            pd.cut(lead_time, bins=self.LEAD_TIME_EDGES, include_lowest=True)
            .astype("string")
            .fillna("missing")
        )

        rates = pd.to_numeric(df["nightly_rate_idr"], errors="coerce")
        df["nightly_rate_bucket"] = (
            pd.cut(rates, bins=self.rate_edges_, labels=False, include_lowest=True)
            .astype("string")
            .fillna("missing")
        )


        for column in self.COUNT_COLUMNS:
            df[f"{column}_category"] = df[column].astype("string").fillna("missing")
        total_nights = df["weekend_nights"].fillna(0) + df["weekday_nights"].fillna(0)
        total_guests = df["num_adults"].fillna(0) + df["num_children"].fillna(0)
        df["total_nights_category"] = total_nights.astype("string")
        df["total_guests_category"] = total_guests.astype("string")
        return df

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class NumericPairwiseInteractions:
    COLUMNS = (
        "num_adults",
        "num_children",
        "weekend_nights",
        "weekday_nights",
        "needs_parking",
        "days_before_arrival",
        "arrival_year",
        "meal_package",
        "is_returning_guest",
        "prior_cancellations",
        "prior_completed_bookings",
        "nightly_rate_idr",
        "special_requests",
        "total_nights",
        "total_guests",
        "arrival_month_sin",
        "arrival_month_cos",
        "arrival_day_sin",
        "arrival_day_cos",
        "log_days_before_arrival",
        "log_nightly_rate_idr",
    )

    def fit(self, X):
        self.columns_ = tuple(column for column in self.COLUMNS if column in X.columns)
        if len(self.columns_) < 2:
            raise ValueError("X must contain at least two selected numeric features")
        self.left_, self.right_ = np.triu_indices(len(self.columns_), k=1)
        values = X.loc[:, self.columns_].to_numpy(dtype=float)
        products = values[:, self.left_] * values[:, self.right_]
        self.center_ = products.mean(axis=0)
        self.scale_ = products.std(axis=0)
        self.scale_[self.scale_ == 0] = 1.0
        return self

    def transform(self, X):
        if not hasattr(self, "columns_"):
            raise ValueError("interaction transformer is not fitted; call fit(X) first")
        missing = set(self.columns_) - set(X.columns)
        if missing:
            raise ValueError(f"X is missing columns: {sorted(missing)}")

        values = X.loc[:, self.columns_].to_numpy(dtype=float)
        products = values[:, self.left_] * values[:, self.right_]
        products = (products - self.center_) / self.scale_
        names = [
            f"interaction__{self.columns_[left]}__{self.columns_[right]}"
            for left, right in zip(self.left_, self.right_)
        ]
        interactions = pd.DataFrame(products, index=X.index, columns=names)
        return pd.concat([X, interactions], axis=1)

    def fit_transform(self, X):
        return self.fit(X).transform(X)
