import pickle
from collections.abc import Mapping
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

def model_input(model, X):
    if hasattr(model, "feature_weights") and isinstance(X, pd.DataFrame):
        return X
    return np.asarray(X)

def model_preprocessor_factory(model, factories):
    if isinstance(factories, Mapping):
        factories = factories.get(type(model).__name__, factories.get("default"))
        if factories is None:
            raise ValueError(f"No preprocessor factory configured for {type(model).__name__}.")
    return factories

def fit(model, X, y, method=None):
    X, y = resample(X, y, method)
    model = deepcopy(model)
    model.fit(model_input(model, X), np.asarray(y))
    return model

def fit_predict(model, X_fit, y_fit, X_eval, method=None, preprocessor_factory=None):
    if preprocessor_factory is not None:
        preprocessor_factory = model_preprocessor_factory(model, preprocessor_factory)
        preprocessor = preprocessor_factory()
        X_fit = preprocessor.fit_transform(X_fit, y_fit)
        X_eval = preprocessor.transform(X_eval)
    fitted = fit(model, X_fit, y_fit, method)
    return np.asarray(fitted.predict(model_input(fitted, X_eval)))

def skip(name):
    print(f"[skip] {name}: not implemented yet")

def holdout(models, X_train, y_train, X_val, y_val, method=None, preprocessor_factory=None):
    scores = {}
    for name, model in models.items():
        try:
            scores[name] = f1_score(
                y_val,
                fit_predict(model, X_train, y_train, X_val, method, preprocessor_factory),
                average="macro",
            )
        except NotImplementedError:
            skip(name)
    return pd.Series(scores, name="holdout", dtype=float).to_frame()

def kfold_scores(model, X, y, method=None, preprocessor_factory=None):
    folds = StratifiedKFold(n_splits=Settings.N_FOLDS, shuffle=True, random_state=Settings.SEED)
    return np.array([
        f1_score(
            y.iloc[e],
            fit_predict(model, X.iloc[f], y.iloc[f], X.iloc[e], method, preprocessor_factory),
            average="macro",
        )
        for f, e in folds.split(X, y)
    ])

def kfold(models, X, y, method=None, preprocessor_factory=None):
    rows = {}
    for name, model in models.items():
        try:
            scores = kfold_scores(model, X, y, method, preprocessor_factory)
        except NotImplementedError:
            skip(name)
            continue
        rows[name] = {"kfold_mean": scores.mean(), "kfold_std": scores.std()}
    return pd.DataFrame(rows, index=["kfold_mean", "kfold_std"]).T

def compare(holdout_table, kfold_table):
    return holdout_table.join(kfold_table, how="outer").sort_values("kfold_mean", ascending=False)

def tune(model, grid, X, y, method=None, preprocessor_factory=None):
    rows = []
    try:
        for params in ParameterGrid(grid):
            scores = kfold_scores(
                deepcopy(model).set_params(**params), X, y, method, preprocessor_factory
            )
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

def confusion(models, X_train, y_train, X_val, y_val, method=None, preprocessor_factory=None):
    _, axes = plt.subplots(1, len(models), figsize=(4 * len(models), 3.6), squeeze=False)
    for axis, (name, model) in zip(axes.flat, models.items()):
        try:
            predicted = fit_predict(model, X_train, y_train, X_val, method, preprocessor_factory)
        except NotImplementedError:
            axis.axis("off")
            continue
        ConfusionMatrixDisplay.from_predictions(y_val, predicted, ax=axis, colorbar=False)
        axis.set_title(name)
    plt.tight_layout()
    plt.show()

def report(models, X_train, y_train, X_val, y_val, method=None, preprocessor_factory=None):
    tables = {}
    for name, model in models.items():
        try:
            predicted = fit_predict(model, X_train, y_train, X_val, method, preprocessor_factory)
        except NotImplementedError:
            continue
        tables[name] = pd.DataFrame(classification_report(y_val, predicted, output_dict=True)).T
    return pd.concat(tables) if tables else pd.DataFrame()

def misclassified(model, train, X_train, y_train, X_val, y_val, method=None, preprocessor_factory=None):
    predicted = pd.Series(
        fit_predict(model, X_train, y_train, X_val, method, preprocessor_factory), index=X_val.index
    )
    wrong = predicted[predicted != y_val]
    return train.loc[wrong.index].assign(predicted=wrong)

def submit(model, name, X, y, X_test, ids, method=None, preprocessor=None):
    model = fit(model, X, y, method)
    path = Settings.OUTPUT_DIR / f"{name}_{datetime.now():%Y%m%d-%H%M%S}.csv"
    result = pd.DataFrame({Settings.ID: np.asarray(ids), Settings.TARGET: model.predict(model_input(model, X_test))})
    result.to_csv(path, index=False)
    with open(path.with_suffix(".pkl"), "wb") as f:
        pickle.dump(model, f)
    if preprocessor is not None:
        preprocessor_path = path.with_name(f"{path.stem}_preprocessor.pkl")
        with open(preprocessor_path, "wb") as f:
            pickle.dump(preprocessor, f)
    print(f"{len(result)} rows written to {path}")
    if preprocessor is not None:
        print(f"Preprocessor saved to {preprocessor_path}")
    return result.head()
