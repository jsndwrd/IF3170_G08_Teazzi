import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import Settings


def as_list(columns):
    return [columns] if isinstance(columns, str) else list(columns)


def make_axes(count, ncols=3):
    nrows = -(-count // ncols)
    _, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.4 * nrows), squeeze=False)
    for axis in axes.flat[count:]:
        axis.axis("off")
    return axes.flat[:count]


def types(df):
    kinds = ["numeric" if pd.api.types.is_numeric_dtype(df[c]) else "categorical" for c in df]
    return pd.DataFrame({"dtype": df.dtypes.astype(str), "kind": kinds, "unique": df.nunique()})


def missing(train, test):
    features = test.columns.drop(Settings.ID)
    combined = pd.concat([train[features], test[features]])
    table = pd.DataFrame({
        "train": train[features].isna().mean() * 100,
        "test": test[features].isna().mean() * 100,
        "combined": combined.isna().mean() * 100,
    }).round(2)
    return table[table["combined"] > 0]


def duplicates(train, test):
    features = test.columns.drop(Settings.ID)
    combined = pd.concat([train[features], test[features]])
    return pd.Series({
        "train": train[features].duplicated().sum(),
        "test": test[features].duplicated().sum(),
        "combined": combined.duplicated().sum(),
    })


def pie(df, column=Settings.TARGET):
    df[column].value_counts().plot.pie(autopct="%.1f%%", ylabel="")
    plt.show()


def bar(df, column=Settings.TARGET):
    df[column].value_counts().plot.bar()
    plt.show()


def histogram(df, columns, by=None, bins=30):
    for axis, column in zip(make_axes(len(as_list(columns))), as_list(columns)):
        edges = np.histogram_bin_edges(df[column].dropna(), bins=bins)
        if by is None:
            axis.hist(df[column].dropna(), bins=edges)
        else:
            for label, group in df.groupby(by):
                axis.hist(group[column].dropna(), bins=edges, alpha=0.5, label=str(label))
            axis.legend(title=by)
        axis.set_title(column)
    plt.tight_layout()
    plt.show()


def boxplot(df, columns, by=None):
    for axis, column in zip(make_axes(len(as_list(columns))), as_list(columns)):
        if by is None:
            axis.boxplot(df[column].dropna())
            axis.set_xticks([])
        else:
            groups = list(df.groupby(by)[column])
            axis.boxplot([values.dropna() for _, values in groups])
            axis.set_xticks(range(1, len(groups) + 1), [str(label) for label, _ in groups])
        axis.set_title(column)
    plt.tight_layout()
    plt.show()


def iqr_outliers(df, columns, factor=1.5):
    q1, q3 = df[columns].quantile(0.25), df[columns].quantile(0.75)
    lower, upper = q1 - factor * (q3 - q1), q3 + factor * (q3 - q1)
    return ((df[columns] < lower) | (df[columns] > upper)).sum().rename("outlier")


def target_rate(df, columns):
    for axis, column in zip(make_axes(len(as_list(columns)), ncols=2), as_list(columns)):
        df.groupby(column)[Settings.TARGET].mean().plot.bar(ax=axis)
        axis.set_title(column)
        axis.set_ylabel(f"rate {Settings.TARGET}")
    plt.tight_layout()
    plt.show()


def correlation(df):
    values = df.select_dtypes("number").drop(columns=[Settings.TARGET]).corrwith(df[Settings.TARGET])
    return values.reindex(values.abs().sort_values(ascending=False).index)