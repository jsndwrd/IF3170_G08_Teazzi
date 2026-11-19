import pickle
from copy import deepcopy
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, f1_score
from sklearn.model_selection import ParameterGrid, StratifiedKFold

from config import Settings

def resample(X, y, method=None):
    if method is None:
        return X, y
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import RandomUnderSampler
    sampler = {"smote": SMOTE, "undersample": RandomUnderSampler}[method]
    return sampler(random_state=Settings.SEED).fit_resample(X, y)

def fit(model, X, y, method=None):
    X, y = resample(X, y, method)
    model = deepcopy(model)
    model.fit(np.asarray(X), np.asarray(y))
    return model

def fit_predict(model, X_fit, y_fit, X_eval, method=None):
    return np.asarray(fit(model, X_fit, y_fit, method).predict(np.asarray(X_eval)))

def skip(name):
    print(f"[skip] {name}: not implemented yet")

def holdout(models, X_train, y_train, X_val, y_val, method=None):
    scores = {}
    for name, model in models.items():
        try:
            scores[name] = f1_score(y_val, fit_predict(model, X_train, y_train, X_val, method), average="macro")
        except NotImplementedError:
            skip(name)
    return pd.Series(scores, name="holdout", dtype=float).to_frame()

def kfold_scores(model, X, y, method=None):
    folds = StratifiedKFold(n_splits=Settings.N_FOLDS, shuffle=True, random_state=Settings.SEED)
    return np.array([
        f1_score(y.iloc[e], fit_predict(model, X.iloc[f], y.iloc[f], X.iloc[e], method), average="macro")
        for f, e in folds.split(X, y)
    ])

def kfold(models, X, y, method=None):
    rows = {}
    for name, model in models.items():
        try:
            scores = kfold_scores(model, X, y, method)
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
        for params in ParameterGrid(grid):
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
    for axis, (name, model) in zip(axes.flat, models.items()):
        try:
            predicted = fit_predict(model, X_train, y_train, X_val, method)
        except NotImplementedError:
            axis.axis("off")
            continue
        ConfusionMatrixDisplay.from_predictions(y_val, predicted, ax=axis, colorbar=False)
        axis.set_title(name)
    plt.tight_layout()
    plt.show()

def report(models, X_train, y_train, X_val, y_val, method=None):
    tables = {}
    for name, model in models.items():
        try:
            predicted = fit_predict(model, X_train, y_train, X_val, method)
        except NotImplementedError:
            continue
        tables[name] = pd.DataFrame(classification_report(y_val, predicted, output_dict=True)).T
    return pd.concat(tables) if tables else pd.DataFrame()

def misclassified(model, train, X_train, y_train, X_val, y_val, method=None):
    predicted = pd.Series(fit_predict(model, X_train, y_train, X_val, method), index=X_val.index)
    wrong = predicted[predicted != y_val]
    return train.loc[wrong.index].assign(predicted=wrong)

def submit(model, name, X, y, X_test, ids, method=None):
    model = fit(model, X, y, method)
    path = Settings.OUTPUT_DIR / f"{name}_{datetime.now():%Y%m%d-%H%M%S}.csv"
    result = pd.DataFrame({Settings.ID: np.asarray(ids), Settings.TARGET: model.predict(np.asarray(X_test))})
    result.to_csv(path, index=False)
    with open(path.with_suffix(".pkl"), "wb") as f:
        pickle.dump(model, f)
    print(f"{len(result)} rows written to {path}")
    return result.head()
