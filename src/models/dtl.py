import numpy as np

from .base import BaseModel

# Referensi:
#   L. Breiman, J. Friedman, R. Olshen, C. Stone, Classification and Regression Trees, Wadsworth, 1984
#   (CART, subset split kategori).
#   J. R. Quinlan, C4.5: Programs for Machine Learning, Morgan Kaufmann, 1993 (gain dikali proporsi nilai yang diketahui).


# A leaf has left = None; an inner node sends a row left if x[feature] <= test (numeric) or x[feature] in test (categorical codes)
class Node:
    def __init__(self, counts, feature=None, test=None, seen=None, missing_left=True, left=None, right=None, score=0.0):
        self.counts, self.feature, self.test, self.seen = counts, feature, test, seen
        self.missing_left, self.left, self.right, self.score = missing_left, left, right, score

    # Rows of col that go left; a missing value, or a category this node never saw, follows missing_left
    def goes_left(self, col):
        if self.seen is None:
            known, left = ~np.isnan(col), col <= self.test
        else:
            known, left = np.isin(col, self.seen), np.isin(col, self.test)
        return np.where(known, left, self.missing_left)


# CART: binary splits on numeric thresholds and on category subsets, with missing values
class DTL(BaseModel):
    def __init__(self, criterion="gini", max_depth=None, min_samples_split=2):
        self.criterion, self.max_depth, self.min_samples_split = criterion, max_depth, min_samples_split

    def fit(self, X, y):
        X = np.asarray(X)
        # A column holding any string is categorical; its categories become integer codes
        self.categories_ = {} if X.dtype != object else {
            f: np.unique([v for v in X[:, f] if isinstance(v, str)])
            for f in range(X.shape[1]) if any(isinstance(v, str) for v in X[:, f])
        }
        X = self._encode(X)
        self.classes_, y = np.unique(np.asarray(y), return_inverse=True)
        self.tree_ = self._grow(X, y, 0)
        self.feature_importances_ = np.zeros(X.shape[1])
        self._add_importance(self.tree_)
        total = self.feature_importances_.sum()
        if total > 0:
            self.feature_importances_ /= total
        return self

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]

    # Class proportions of the training rows in each sample's leaf
    def predict_proba(self, X):
        X = self._encode(np.asarray(X))
        out = np.zeros((len(X), len(self.classes_)))
        self._fill(self.tree_, X, np.arange(len(X)), out)
        return out

    # Float matrix with categories replaced by codes; None, NaN and unseen categories become NaN
    def _encode(self, X):
        out = np.empty(X.shape)
        for f in range(X.shape[1]):
            if f in self.categories_:
                code = {c: i for i, c in enumerate(self.categories_[f])}
                out[:, f] = [code.get(v, np.nan) for v in X[:, f]]
            else:
                out[:, f] = [np.nan if v is None else v for v in X[:, f]] if X.dtype == object else X[:, f]
        return out

    def _grow(self, X, y, depth):
        counts = np.bincount(y, minlength=len(self.classes_))
        if counts.max() == len(y) or len(y) < self.min_samples_split or (self.max_depth is not None and depth >= self.max_depth):
            return Node(counts)
        node = self._best_split(X, y, counts)
        if node is None:
            return Node(counts)
        mask = node.goes_left(X[:, node.feature])
        node.left, node.right = self._grow(X[mask], y[mask], depth + 1), self._grow(X[~mask], y[~mask], depth + 1)
        return node

    # Impurity of every row of class counts
    def _impurity(self, counts):
        p = counts / counts.sum(axis=-1, keepdims=True)
        if self.criterion == "gini":
            return 1 - (p ** 2).sum(axis=-1)
        if self.criterion == "entropy":
            return -(p * np.log2(np.where(p > 0, p, 1))).sum(axis=-1)
        raise ValueError(f"unknown criterion {self.criterion}")

    # Best split over every feature, scored on the rows where the feature is known; the gain is scaled by the
    # known fraction so a feature with many missing values is not favoured (C4.5), and missing rows go to the larger child
    def _best_split(self, X, y, counts):
        n, onehot, best = len(y), np.eye(len(counts))[y], None
        for feature in range(X.shape[1]):
            known = ~np.isnan(X[:, feature])
            xs, ys = X[known, feature], onehot[known]
            if feature in self.categories_:
                # Categories ordered by their share of the last class: for two classes the best subset split is a prefix (Breiman et al.)
                cats, inverse = np.unique(xs, return_inverse=True)
                per_cat = np.zeros((len(cats), len(counts)))
                np.add.at(per_cat, inverse, ys)
                order = np.argsort(per_cat[:, -1] / per_cat.sum(axis=1), kind="stable")
                left = np.cumsum(per_cat[order], axis=0)[:-1]
            else:
                order = np.argsort(xs, kind="stable")
                xs = xs[order]
                cut = np.flatnonzero(xs[:-1] < xs[1:])
                left = np.cumsum(ys[order], axis=0)[cut]
            if len(left) == 0:
                continue
            known_counts = ys.sum(axis=0)
            n_known, n_left = known_counts.sum(), left.sum(axis=1)
            right = known_counts - left
            child = (n_left * self._impurity(left) + (n_known - n_left) * self._impurity(right)) / n_known
            gain = n_known / n * (self._impurity(known_counts) - child)
            i = gain.argmax()
            if gain[i] <= 0 or (best is not None and gain[i] <= best.score):
                continue
            if feature in self.categories_:
                test, seen = cats[order[:i + 1]], cats
            else:
                a, b = xs[cut[i]], xs[cut[i] + 1]
                test, seen = ((a + b) / 2 if (a + b) / 2 < b else a), None  # adjacent floats can round the midpoint up to b
            best = Node(counts, feature, test, seen, n_left[i] >= n_known - n_left[i], score=gain[i])
        if best is not None:
            best.score *= n  # n_t * gain, summed per feature for feature_importances_
        return best

    def _add_importance(self, node):
        if node.left is not None:
            self.feature_importances_[node.feature] += node.score
            self._add_importance(node.left)
            self._add_importance(node.right)

    # Sends the rows down the tree together and writes each leaf's class proportions
    def _fill(self, node, X, rows, out):
        if node.left is None:
            out[rows] = node.counts / node.counts.sum()
            return
        mask = node.goes_left(X[rows, node.feature])
        self._fill(node.left, X, rows[mask], out)
        self._fill(node.right, X, rows[~mask], out)

    def n_leaves(self, node=None):
        node = self.tree_ if node is None else node
        return 1 if node.left is None else self.n_leaves(node.left) + self.n_leaves(node.right)


# Draws the top `top_n` split levels of a fitted DTL; deeper subtrees are shown as "..."
def plot_tree(model, feature_names, top_n=3, path=None):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.axis("off")
    n_leaves = [0]

    # Returns the x position of node; leaves take consecutive x slots, inner nodes sit above their children
    def draw(node, depth):
        if node.left is not None and depth < top_n:
            xs = [draw(node.left, depth + 1), draw(node.right, depth + 1)]
            x = sum(xs) / 2
            for child_x, edge in zip(xs, ["ya", "tidak"]):
                ax.plot([x, child_x], [-depth, -depth - 1], color="gray", zorder=1)
                ax.text((x + child_x) / 2, -depth - 0.5, edge, ha="center", fontsize=8, color="gray")
            name = feature_names[node.feature]
            if node.seen is None:
                test = f"<= {node.test:.4g}"
            else:
                test = "in {" + ", ".join(str(c) for c in model.categories_[node.feature][node.test.astype(int)]) + "}"
            label = f"{name}\n{test}\nkosong: {'ya' if node.missing_left else 'tidak'}"
        else:
            x = n_leaves[0]
            n_leaves[0] += 1
            p = node.counts / node.counts.sum()
            label = "..." if node.left is not None else f"kelas {model.classes_[p.argmax()]}\n{p.max():.0%}"
        ax.text(x, -depth, label, ha="center", va="center", fontsize=8, bbox=dict(boxstyle="round", fc="white"))
        return x

    draw(model.tree_, 0)
    ax.set_xlim(-0.5, n_leaves[0] - 0.5)
    ax.set_ylim(-top_n - 0.5, 0.5)
    fig.set_size_inches(max(6, 1.6 * n_leaves[0]), 1.6 * (top_n + 1))
    ax.set_title(f"Decision Tree ({top_n} level percabangan teratas)")
    if path:
        fig.savefig(path, dpi=160, bbox_inches="tight")
    return fig
