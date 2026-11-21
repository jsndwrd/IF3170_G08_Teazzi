import numpy as np

from .base import BaseModel


# Klasifikator Bayesian network atas fitur terdiskretisasi: Naive Bayes (tree=False) atau TAN (tree=True),
# dengan aturan keputusan sensitif-biaya lewat cost_fp dan cost_fn.
#
# Referensi:
# - TAN: N. Friedman, D. Geiger, M. Goldszmidt, "Bayesian Network Classifiers", Machine Learning 29(2-3), 131-163, 1997.
#   https://doi.org/10.1023/A:1007465528199
# - Pohon dependensi dengan bobot informasi bersama: C. K. Chow, C. N. Liu, "Approximating Discrete Probability
#   Distributions with Dependence Trees", IEEE Trans. Information Theory 14(3), 462-467, 1968.
#   https://doi.org/10.1109/TIT.1968.1054142
# - Algoritma Prim: R. C. Prim, "Shortest Connection Networks and Some Generalizations", Bell System Technical
#   Journal 36(6), 1389-1401, 1957. https://doi.org/10.1002/j.1538-7305.1957.tb01515.x
# - Aturan keputusan sensitif-biaya: C. Elkan, "The Foundations of Cost-Sensitive Learning", Proc. IJCAI, 973-977, 2001.
#   https://ijcai.org/proceedings/2001/133
class BayesNetwork(BaseModel):
    def __init__(self, n_bins=5, alpha=1.0, tree=False, cost_fp=1.0, cost_fn=1.0):
        self.n_bins, self.alpha, self.tree, self.cost_fp, self.cost_fn = n_bins, alpha, tree, cost_fp, cost_fn

    def fit_bins(self, X):
        self.edges = []
        for j in range(X.shape[1]):
            column = X[:, j]
            values = np.unique(column)
            if len(values) <= self.n_bins:
                edges = (values[:-1] + values[1:]) / 2
            else:
                quantile_levels = np.linspace(0, 1, self.n_bins + 1)[1:-1]
                edges = np.unique(np.quantile(column, quantile_levels))
            self.edges.append(edges)

    def discretize(self, X):
        bin_columns = []
        for j, edges in enumerate(self.edges):
            bin_columns.append(np.searchsorted(edges, X[:, j], side="left"))
        return np.column_stack(bin_columns)

    # I(Xi; Xj | C), bobot edge untuk pohon TAN
    def conditional_mutual_info(self, xi, xj, y_idx, ki, kj):
        joint = np.zeros((len(self.classes), ki, kj))
        np.add.at(joint, (y_idx, xi, xj), 1)

        count_c = joint.sum(axis=(1, 2))
        count_ci = joint.sum(axis=2)
        count_cj = joint.sum(axis=1)

        seen = joint > 0
        numerator = (joint * count_c[:, None, None])[seen]
        denominator = (count_ci[:, :, None] * count_cj[:, None, :])[seen]
        weight = joint[seen] / joint.sum()
        return float((weight * np.log(numerator / denominator)).sum())

    # Maximum weight spanning tree (Prim) dari fitur 0, mengembalikan parent tiap fitur (-1 untuk root)
    def learn_tree(self, B, y_idx):
        n_features = B.shape[1]
        n_bins = [len(edges) + 1 for edges in self.edges]

        weight = np.zeros((n_features, n_features))
        for i in range(n_features):
            for j in range(i + 1, n_features):
                cmi = self.conditional_mutual_info(B[:, i], B[:, j], y_idx, n_bins[i], n_bins[j])
                weight[i, j] = weight[j, i] = cmi

        parent = -np.ones(n_features, dtype=int)
        in_tree = np.zeros(n_features, dtype=bool)
        in_tree[0] = True
        best_weight = weight[0].copy()
        best_source = np.zeros(n_features, dtype=int)

        for _ in range(n_features - 1):
            candidates = np.where(in_tree, -np.inf, best_weight)
            new = int(np.argmax(candidates))
            in_tree[new] = True
            parent[new] = best_source[new]

            better = ~in_tree & (weight[new] > best_weight)
            best_weight[better] = weight[new][better]
            best_source[better] = new
        return parent

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y)
        self.classes = np.unique(y)
        y_idx = np.searchsorted(self.classes, y)
        self.fit_bins(X)
        B = self.discretize(X)

        self.parent = self.learn_tree(B, y_idx) if self.tree else -np.ones(B.shape[1], dtype=int)
        class_counts = np.bincount(y_idx)
        self.log_prior = np.log(class_counts / len(y_idx))

        self.log_cpt = []
        for j, edges in enumerate(self.edges):
            n_values = len(edges) + 1
            p = self.parent[j]
            if p < 0:
                n_parent_values = 1
                parent_bin = np.zeros(len(B), dtype=int)
            else:
                n_parent_values = len(self.edges[p]) + 1
                parent_bin = B[:, p]

            counts = np.zeros((len(self.classes), n_parent_values, n_values))
            np.add.at(counts, (y_idx, parent_bin, B[:, j]), 1)

            smoothed = counts + self.alpha
            total = counts.sum(axis=2, keepdims=True) + self.alpha * n_values
            self.log_cpt.append(np.log(smoothed / total))
        return self

    def joint_log_likelihood(self, X):
        B = self.discretize(np.asarray(X, dtype=float))
        scores = np.tile(self.log_prior, (len(B), 1))
        for j, table in enumerate(self.log_cpt):
            p = self.parent[j]
            parent_bin = 0 if p < 0 else B[:, p]
            scores += table[:, parent_bin, B[:, j]].T
        return scores

    def predict_proba(self, X):
        scores = self.joint_log_likelihood(X)
        scores = scores - scores.max(axis=1, keepdims=True)
        unnormalized = np.exp(scores)
        return unnormalized / unnormalized.sum(axis=1, keepdims=True)

    # Untuk dua kelas, memprediksi 1 jika cost_fn * P(1) > cost_fp * P(0)
    def predict(self, X):
        proba = self.predict_proba(X)
        if len(self.classes) == 2:
            cost_if_predict_0 = self.cost_fn * proba[:, 1]
            cost_if_predict_1 = self.cost_fp * proba[:, 0]
            predict_1 = cost_if_predict_1 < cost_if_predict_0
            return self.classes[predict_1.astype(int)]
        return self.classes[proba.argmax(axis=1)]

    def feature_edges(self):
        pairs = []
        for child, parent in enumerate(self.parent):
            if parent >= 0:
                pairs.append((int(parent), child))
        return pairs
