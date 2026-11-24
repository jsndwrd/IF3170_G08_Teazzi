import numpy as np

from .base import BaseModel


# Klasifier Bayesian network buat fitur yang udah didiskretisasi: Naive Bayes (default), TAN (tree=True), atau KDB (k>0),
# plus aturan keputusan pakai biaya lewat cost_fp dan cost_fn.
#
# Referensi:
# - Dasar Bayesian network (DAG, CPT, chain rule), Naive Bayes, MAP, dan Laplace smoothing: materi kuliah IF3170, "BayesNet" (Materi 14), Edunex ITB, 2026.
# - TAN: N. Friedman, D. Geiger, M. Goldszmidt, "Bayesian Network Classifiers", Machine Learning 29(2-3), 131-163, 1997.
#   https://doi.org/10.1023/A:1007465528199
# - Pohon dependensi dengan bobot informasi bersama: C. K. Chow, C. N. Liu, "Approximating Discrete Probability
#   Distributions with Dependence Trees", IEEE Trans. Information Theory 14(3), 462-467, 1968.
#   https://doi.org/10.1109/TIT.1968.1054142
# - Algoritma Prim: R. C. Prim, "Shortest Connection Networks and Some Generalizations", Bell System Technical
#   Journal 36(6), 1389-1401, 1957. https://doi.org/10.1002/j.1538-7305.1957.tb01515.x
# - KDB: M. Sahami, "Learning Limited Dependence Bayesian Classifiers", Proc. KDD-96, 335-338, 1996.
#   https://www.osti.gov/biblio/421307
# - Aturan keputusan sensitif-biaya: C. Elkan, "The Foundations of Cost-Sensitive Learning", Proc. IJCAI, 973-977, 2001.
#   https://ijcai.org/proceedings/2001/133
class SparseCPT:
    def __init__(self, keys, sizes, n_values, alpha):
        self.sizes, self.n_values, self.alpha = sizes, n_values, alpha
        context = np.ravel_multi_index(keys[:-1], sizes)  # (kelas, bin parent...) -> satu angka
        self.cell_keys, self.cell_counts = np.unique(context * n_values + keys[-1], return_counts=True)
        self.context_keys, self.context_counts = np.unique(context, return_counts=True)

    @staticmethod
    def lookup(keys, counts, query):
        i = np.minimum(np.searchsorted(keys, query), len(keys) - 1)
        return np.where(keys[i] == query, counts[i], 0)

    # keys: [kelas, bin parent..., bin fitur], masing-masing array per sampel
    def log_prob(self, keys):
        context = np.ravel_multi_index(keys[:-1], self.sizes)
        n_cell = self.lookup(self.cell_keys, self.cell_counts, context * self.n_values + keys[-1])
        n_context = self.lookup(self.context_keys, self.context_counts, context)
        return np.log((n_cell + self.alpha) / (n_context + self.alpha * self.n_values))


class BayesNetwork(BaseModel):
    DENSE_LIMIT = 100_000

    def __init__(self, n_bins=5, alpha=1.0, tree=False, k=0, cost_fp=1.0, cost_fn=1.0):
        self.n_bins, self.alpha, self.tree, self.k = n_bins, alpha, tree, k
        self.cost_fp, self.cost_fn = cost_fp, cost_fn

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

    # I(Xi; Xj | C), jadi bobot edge buat TAN dan KDB
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

    # I(X; C), dipakai KDB buat ngurutin fitur dari yang paling informatif
    def mutual_info(self, x, y_idx, kx):
        joint = np.zeros((len(self.classes), kx))
        np.add.at(joint, (y_idx, x), 1)
        p = joint / joint.sum()
        expected = p.sum(axis=1, keepdims=True) * p.sum(axis=0, keepdims=True)
        seen = p > 0
        return float((p[seen] * np.log(p[seen] / expected[seen])).sum())

    def cmi_matrix(self, B, y_idx):
        n_features = B.shape[1]
        n_bins = [len(edges) + 1 for edges in self.edges]
        weight = np.zeros((n_features, n_features))
        for i in range(n_features):
            for j in range(i + 1, n_features):
                cmi = self.conditional_mutual_info(B[:, i], B[:, j], y_idx, n_bins[i], n_bins[j])
                weight[i, j] = weight[j, i] = cmi
        return weight

    # Maximum weight spanning tree (Prim) mulai dari fitur 0, balikin parent tiap fitur (-1 buat root)
    def learn_tree(self, B, y_idx):
        n_features = B.shape[1]
        weight = self.cmi_matrix(B, y_idx)

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

    # KDB: urutin fitur berdasarkan I(X; C), terus tiap fitur ambil maksimal k parent dari fitur
    def learn_kdb(self, B, y_idx):
        weight = self.cmi_matrix(B, y_idx)
        relevance = [self.mutual_info(B[:, j], y_idx, len(edges) + 1) for j, edges in enumerate(self.edges)]
        order = np.argsort(relevance)[::-1]

        parents = [[] for _ in range(B.shape[1])]
        for position, j in enumerate(order):
            earlier = order[:position]
            strongest = earlier[np.argsort(weight[j, earlier])[::-1]]
            parents[j] = [int(p) for p in strongest[:self.k]]
        return parents

    def learn_structure(self, B, y_idx):
        if self.k > 0:
            return self.learn_kdb(B, y_idx)
        if self.tree:
            return [[] if p < 0 else [int(p)] for p in self.learn_tree(B, y_idx)]
        return [[] for _ in range(B.shape[1])]

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y)
        self.classes = np.unique(y)
        y_idx = np.searchsorted(self.classes, y)
        self.fit_bins(X)
        B = self.discretize(X)

        self.parents = self.learn_structure(B, y_idx)  # daftar parent tiap fitur
        class_counts = np.bincount(y_idx)
        self.log_prior = np.log(class_counts / len(y_idx))

        self.log_cpt = []  # satu CPT (log) per fitur, aksesnya [kelas, bin parent..., feature_bin]
        for j, edges in enumerate(self.edges):
            n_values = len(edges) + 1
            parents = self.parents[j]
            if parents:
                parent_sizes = [len(self.edges[p]) + 1 for p in parents]
                parent_bins = [B[:, p] for p in parents]
            else:
                parent_sizes = [1]
                parent_bins = [np.zeros(len(B), dtype=int)]

            sizes = [len(self.classes), *parent_sizes]
            if np.prod(sizes, dtype=float) * n_values > self.DENSE_LIMIT:
                self.log_cpt.append(SparseCPT([y_idx, *parent_bins, B[:, j]], sizes, n_values, self.alpha))
                continue

            counts = np.zeros((len(self.classes), *parent_sizes, n_values))
            np.add.at(counts, (y_idx, *parent_bins, B[:, j]), 1)

            smoothed = counts + self.alpha
            total = counts.sum(axis=-1, keepdims=True) + self.alpha * n_values
            self.log_cpt.append(np.log(smoothed / total))
        return self

    def joint_log_likelihood(self, X):
        B = self.discretize(np.asarray(X, dtype=float))
        scores = np.tile(self.log_prior, (len(B), 1))
        for j, table in enumerate(self.log_cpt):
            parents = self.parents[j]
            if parents:
                # bin tiap parent fitur buat tiap sampel
                parent_bins = [B[:, p] for p in parents]
            else:
                # fitur ga punya parent, selalu pakai parent_bin 0
                parent_bins = [0]

            feature_bin = B[:, j]
            if isinstance(table, SparseCPT):
                log_probs = np.stack([table.log_prob([np.full(len(B), c), *parent_bins, feature_bin])
                                      for c in range(len(self.classes))])
            else:
                log_probs = table[(slice(None), *parent_bins, feature_bin)]  # bentuknya (n_classes, n_samples)
            scores += log_probs.T
        return scores

    def predict_proba(self, X):
        scores = self.joint_log_likelihood(X)
        scores = scores - scores.max(axis=1, keepdims=True)
        unnormalized = np.exp(scores)
        return unnormalized / unnormalized.sum(axis=1, keepdims=True)

    # Kalau cuma dua kelas, prediksi 1 kalau cost_fn * P(1) > cost_fp * P(0)
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
        for child, parents in enumerate(self.parents):
            for parent in parents:
                pairs.append((int(parent), child))
        return pairs
