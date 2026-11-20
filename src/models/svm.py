import numpy as np

from .base import BaseModel

class SVM(BaseModel):
    # C bounds every alpha (0 <= alpha <= C), kernel is 'linear' | 'poly' | 'rbf' | 'sigmoid'
    def __init__(self, C=1.0, kernel="linear", gamma=0.1, degree=3, coef0=1.0, tol=1e-3, max_iter=100000):
        self.C, self.kernel, self.gamma, self.degree, self.coef0 = C, kernel, gamma, degree, coef0
        self.tol, self.max_iter = tol, max_iter

    # K(a, b) for every pair of rows in A and B
    def kernel_matrix(self, A, B):
        if self.kernel == "linear":
            return A @ B.T
        if self.kernel == "poly":
            return (self.gamma * A @ B.T + self.coef0) ** self.degree
        if self.kernel == "rbf":
            sq = (A ** 2).sum(axis=1)[:, None] + (B ** 2).sum(axis=1)[None, :] - 2 * A @ B.T
            return np.exp(-self.gamma * np.maximum(sq, 0))
        if self.kernel == "sigmoid":
            return np.tanh(self.gamma * A @ B.T + self.coef0)
        raise ValueError(f"unknown kernel {self.kernel}")

    # K(x_k, x) for every row x of X, reusing the precomputed squared norms sq for the RBF kernel
    def kernel_row(self, X, sq, k):
        if self.kernel == "rbf":
            return np.exp(-self.gamma * np.maximum(sq + sq[k] - 2 * (X @ X[k]), 0))
        return self.kernel_matrix(X, X[[k]])[:, 0]

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.where(np.asarray(y) == 1, 1.0, -1.0)
        alpha, u = np.zeros(len(y)), np.zeros(len(y))
        diag = np.concatenate([np.diag(self.kernel_matrix(part, part)) for part in np.array_split(X, len(X) // 500 + 1)])
        sq = (X ** 2).sum(axis=1)

        for self.n_iter_ in range(self.max_iter):
            g = y - u
            up = ((y > 0) & (alpha < self.C)) | ((y < 0) & (alpha > 0))
            low = ((y < 0) & (alpha < self.C)) | ((y > 0) & (alpha > 0))
            i = np.where(up, g, -np.inf).argmax()
            if g[i] - np.where(low, g, np.inf).min() < self.tol:
                break

            Ki = self.kernel_row(X, sq, i)
            gain = np.where(low & (g < g[i]), (g[i] - g) ** 2 / np.maximum(diag[i] + diag - 2 * Ki, 1e-12), -np.inf)
            j = gain.argmax()
            Kj = self.kernel_row(X, sq, j)
            eta = max(diag[i] + diag[j] - 2 * Ki[j], 1e-12)
            if y[i] != y[j]:
                L, H = max(0, alpha[j] - alpha[i]), min(self.C, self.C + alpha[j] - alpha[i])
            else:
                L, H = max(0, alpha[i] + alpha[j] - self.C), min(self.C, alpha[i] + alpha[j])
            aj = self.snap(np.clip(alpha[j] + y[j] * (g[j] - g[i]) / eta, L, H))
            ai = self.snap(alpha[i] + y[i] * y[j] * (alpha[j] - aj))

            u += (ai - alpha[i]) * y[i] * Ki + (aj - alpha[j]) * y[j] * Kj
            alpha[i], alpha[j] = ai, aj

        sv = alpha > 1e-8
        free = sv & (alpha < self.C - 1e-8)
        g = y - u
        self.b = g[free].mean() if free.any() else g[sv].mean()
        self.alpha, self.support_vectors = alpha[sv], X[sv]
        self.alpha_y = alpha[sv] * y[sv]
        if self.kernel == "linear":
            self.coef_ = self.alpha_y @ self.support_vectors
        return self

    def snap(self, a):
        eps = 1e-8 * self.C
        return 0.0 if a < eps else self.C if a > self.C - eps else float(a)

    # f(x) = sum over support vectors of alpha_i y_i K(x_i, x) + b
    def decision_function(self, X):
        X = np.asarray(X, dtype=float)
        if self.kernel == "linear":
            return X @ self.coef_ + self.b
        parts = np.array_split(X, len(X) // 1000 + 1)
        return np.concatenate([self.kernel_matrix(part, self.support_vectors) @ self.alpha_y for part in parts]) + self.b

    def predict(self, X):
        return (self.decision_function(X) >= 0).astype(int)
