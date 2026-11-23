import pickle
from copy import deepcopy
from datetime import datetime
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import Settings
from models import BaseModel
from preprocessing import Pipeline, stratified_folds

# 'undersample' drops random majority rows down to the minority count, 'smote' adds minority rows on the
# segment between a minority row and one of its 5 nearest minority neighbours until both classes are equal
def resample(X, y, method=None):
    if method is None:
        return X, y
    X, y, rng = np.asarray(X, dtype=float), np.asarray(y), np.random.default_rng(Settings.SEED)
    classes, counts = np.unique(y, return_counts=True)
    minority, majority = classes[counts.argmin()], classes[counts.argmax()]
    if method == "undersample":
        keep = np.concatenate([np.flatnonzero(y == minority), rng.choice(np.flatnonzero(y == majority), counts.min(), replace=False)])
        return X[keep], y[keep]
    if method != "smote":
        raise ValueError(f"unknown resampling method {method}")
    M = X[y == minority]
    sq, neighbours = (M ** 2).sum(axis=1), []
    for start in range(0, len(M), 500):
        D = sq[start:start + 500, None] + sq - 2 * M[start:start + 500] @ M.T
        D[np.arange(len(D)), np.arange(start, start + len(D))] = np.inf
        neighbours.append(np.argpartition(D, 5, axis=1)[:, :5])
    neighbours = np.vstack(neighbours)
    base = rng.integers(len(M), size=counts.max() - counts.min())
    partner = M[neighbours[base, rng.integers(5, size=len(base))]]
    new = M[base] + rng.random((len(base), 1)) * (partner - M[base])
    return np.vstack([X, new]), np.concatenate([y, np.full(len(new), minority)])

def fit(model, X, y, method=None):
    X, y = resample(X, y, method)
    model = deepcopy(model)
    model.fit(np.asarray(X), np.asarray(y))
    return model

def fit_predict(model, X_fit, y_fit, X_eval, method=None):
    return np.asarray(fit(model, X_fit, y_fit, method).predict(np.asarray(X_eval)))

# Mean over classes of F1 = 2TP / (2TP + FP + FN)
def macro_f1(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.mean([2 * ((y_true == c) & (y_pred == c)).sum() / ((y_true == c).sum() + (y_pred == c).sum())
                          for c in np.union1d(y_true, y_pred)]))

# Higher means more likely canceled: P(y=1) when the model has predict_proba, the SVM margin otherwise
def scores(model, X):
    X = np.asarray(X, dtype=float)
    if isinstance(model, Pipeline):
        model, X = model.model, model.reduce(X)
    return model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X)

# Score of every row from the fold model that did not train on it
# With prep, X is raw and a copy of prep is refit on each training fold, so label-based features never see the scored fold
def oof_scores(model, X, y, method=None, prep=None):
    out = np.zeros(len(y))
    for f, e in stratified_folds(y):
        A, B = X.iloc[f], X.iloc[e]
        if prep is not None:
            p = deepcopy(prep)
            A, B = p.fit_transform(A, y.iloc[f]), p.transform(B)
        out[e] = scores(fit(model, A, y.iloc[f], method), B)
    return out

# Cut-off on the scores with the best macro F1, searched over score quantiles
def best_threshold(y, s):
    cuts = np.unique(np.quantile(s, np.linspace(0.05, 0.95, 181)))
    f1s = [macro_f1(y, (s >= t).astype(int)) for t in cuts]
    return float(cuts[int(np.argmax(f1s))]), float(max(f1s))

# Same model with its decision rule moved from the default cut-off to score >= threshold
class Threshold(BaseModel):
    def __init__(self, model, threshold):
        self.model, self.threshold = model, threshold

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return (scores(self.model, X) >= self.threshold).astype(int)

# Tunes the cut-off on out-of-fold scores of X, y and returns the wrapped model with that cut-off
def with_threshold(model, X, y, method=None, prep=None):
    threshold, _ = best_threshold(np.asarray(y), oof_scores(model, X, y, method, prep))
    return Threshold(deepcopy(model), threshold)

def skip(name):
    print(f"[skip] {name}: not implemented yet")

# X arguments take one matrix for every model, or a {model name: matrix} dict when models use different preprocessing
def pick(X, name):
    return X[name] if isinstance(X, dict) else X

def holdout(models, X_train, y_train, X_val, y_val, method=None):
    scores = {}
    for name, model in models.items():
        try:
            scores[name] = macro_f1(y_val, fit_predict(model, pick(X_train, name), y_train, pick(X_val, name), method))
        except NotImplementedError:
            skip(name)
    return pd.Series(scores, name="holdout", dtype=float).to_frame()

def kfold_scores(model, X, y, method=None):
    return np.array([
        macro_f1(y.iloc[e], fit_predict(model, X.iloc[f], y.iloc[f], X.iloc[e], method))
        for f, e in stratified_folds(y)
    ])

def kfold(models, X, y, method=None):
    rows = {}
    for name, model in models.items():
        try:
            scores = kfold_scores(model, pick(X, name), y, method)
        except NotImplementedError:
            skip(name)
            continue
        rows[name] = {"kfold_mean": scores.mean(), "kfold_std": scores.std()}
    return pd.DataFrame(rows, index=["kfold_mean", "kfold_std"]).T

def compare(holdout_table, kfold_table):
    return holdout_table.join(kfold_table, how="outer").sort_values("kfold_mean", ascending=False)

def tune(model, grid, X, y, method=None):
    rows = []
    try:
        for values in product(*grid.values()):
            params = dict(zip(grid, values))
            scores = kfold_scores(deepcopy(model).set_params(**params), X, y, method)
            rows.append({**params, "kfold_mean": scores.mean(), "kfold_std": scores.std(), "params": params})
    except NotImplementedError:
        skip(type(model).__name__)
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("kfold_mean", ascending=False, ignore_index=True)

def best(model, grid_table):
    return model if grid_table.empty else deepcopy(model).set_params(**grid_table["params"].iloc[0])

def importance(model, X, y):
    fitted = fit(model, X, y)
    values = getattr(fitted, "feature_importances_", None)
    values = np.ravel(fitted.coef_) if values is None else values
    ranked = pd.Series(values, index=X.columns)
    return ranked.reindex(ranked.abs().sort_values(ascending=False).index)

def confusion(models, X_train, y_train, X_val, y_val, method=None):
    _, axes = plt.subplots(1, len(models), figsize=(4 * len(models), 3.6), squeeze=False)
    y_val = np.asarray(y_val)
    for axis, (name, model) in zip(axes.flat, models.items()):
        try:
            predicted = fit_predict(model, pick(X_train, name), y_train, pick(X_val, name), method)
        except NotImplementedError:
            axis.axis("off")
            continue
        matrix = np.array([[((y_val == t) & (predicted == p)).sum() for p in (0, 1)] for t in (0, 1)])
        axis.imshow(matrix, cmap="Blues")
        for (t, p), count in np.ndenumerate(matrix):
            axis.text(p, t, count, ha="center", va="center", color="white" if count > matrix.max() / 2 else "black")
        axis.set(xticks=[0, 1], yticks=[0, 1], xlabel="Predicted label", ylabel="True label", title=name)
    plt.tight_layout()
    plt.show()

# Precision, recall, F1 and support per class, plus macro average and accuracy
def classification_table(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    rows = {}
    for c in np.unique(y_true):
        tp = ((y_true == c) & (y_pred == c)).sum()
        precision, recall = tp / max((y_pred == c).sum(), 1), tp / (y_true == c).sum()
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows[str(c)] = {"precision": precision, "recall": recall, "f1-score": f1, "support": (y_true == c).sum()}
    table = pd.DataFrame(rows).T
    table.loc["macro avg"] = [*table[["precision", "recall", "f1-score"]].mean(), len(y_true)]
    table.loc["accuracy"] = [np.nan, np.nan, (y_true == y_pred).mean(), len(y_true)]
    return table

def report(models, X_train, y_train, X_val, y_val, method=None):
    tables = {}
    for name, model in models.items():
        try:
            predicted = fit_predict(model, pick(X_train, name), y_train, pick(X_val, name), method)
        except NotImplementedError:
            continue
        tables[name] = classification_table(y_val, predicted)
    return pd.concat(tables) if tables else pd.DataFrame()

def misclassified(model, train, X_train, y_train, X_val, y_val, method=None):
    predicted = pd.Series(fit_predict(model, X_train, y_train, X_val, method), index=X_val.index)
    wrong = predicted[predicted != y_val]
    return train.loc[wrong.index].assign(predicted=wrong)

# Raw rows whose segment has a 0% cancel rate in train: 3+ special requests or a complimentary stay
def never_cancel(raw):
    return ((raw["special_requests"] >= 3) | (raw["booking_channel"] == "Complimentary")).values

# force_zero is an optional boolean mask of test rows predicted 0 regardless of the model
def submit(model, name, X, y, X_test, ids, method=None, force_zero=None):
    model = fit(model, X, y, method)
    path = Settings.OUTPUT_DIR / f"{name}_{datetime.now():%Y%m%d-%H%M%S}.csv"
    predicted = model.predict(np.asarray(X_test))
    if force_zero is not None:
        predicted = np.where(force_zero, 0, predicted)
    result = pd.DataFrame({Settings.ID: np.asarray(ids), Settings.TARGET: predicted})
    result.to_csv(path, index=False)
    with open(path.with_suffix(".pkl"), "wb") as f:
        pickle.dump(model, f)
    print(f"{len(result)} rows written to {path}")
    return result.head()
