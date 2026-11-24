import numpy as np

from .base import BaseModel


class LogisticRegression(BaseModel):
    def __init__(
        self,
        lr=0.01,
        n_iter=1000,
        reg=None,
        lam=0.0,
        l1_ratio=0.5,
        threshold=0.5,
        random_state=42,
        eval_interval=1000,
        history_interval=100,
    ):
        self.lr = lr
        self.n_iter = n_iter
        self.reg = reg
        self.lam = lam
        self.l1_ratio = l1_ratio
        self.threshold = threshold
        self.random_state = random_state
        self.eval_interval = eval_interval
        self.history_interval = history_interval

    def fit(self, X, y):
        X = self._as_features(X)
        y = np.asarray(y, dtype=float).reshape(-1)

        if X.shape[0] == 0:
            raise ValueError("X must contain at least one sample")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must contain the same number of samples")
        if not np.isfinite(X).all() or not np.isfinite(y).all():
            raise ValueError("X and y must contain only finite values")
        if not np.isin(y, (0.0, 1.0)).all():
            raise ValueError("y must contain binary labels 0 and 1")
        if np.unique(y).size != 2:
            raise ValueError("y must contain both binary classes 0 and 1")
        self._validate_params()

        n_samples, n_features = X.shape
        self.weights_ = np.zeros(n_features, dtype=float)
        self.bias_ = 0.0
        self.classes_ = np.array([0, 1], dtype=int)
        self.loss_history_ = []
        self.loss_steps_ = []
        self.coef_history_ = []
        self.coef_history_steps_ = []

        self.loss_history_.append(self._objective(X, y))
        self.loss_steps_.append(0)
        self.coef_history_.append(self._parameters())
        self.coef_history_steps_.append(0)
        rng = np.random.default_rng(self.random_state)

        for step in range(int(self.n_iter)):
            sample_index = rng.integers(n_samples)
            sample = X[sample_index]
            label = y[sample_index]
            logit = float(sample @ self.weights_ + self.bias_)
            probability = float(self._sigmoid(np.array([logit]))[0])
            residual = label - probability

            self.weights_ += self.lr * (
                residual * sample - self._regularization_gradient()
            )
            self.bias_ += self.lr * residual

            completed_steps = step + 1
            if completed_steps % self.history_interval == 0:
                self.coef_history_.append(self._parameters())
                self.coef_history_steps_.append(completed_steps)
            if completed_steps % self.eval_interval == 0:
                self.loss_history_.append(self._objective(X, y))
                self.loss_steps_.append(completed_steps)

        if self.coef_history_steps_[-1] != self.n_iter:
            self.coef_history_.append(self._parameters())
            self.coef_history_steps_.append(int(self.n_iter))
        if self.loss_steps_[-1] != self.n_iter:
            self.loss_history_.append(self._objective(X, y))
            self.loss_steps_.append(int(self.n_iter))


        self.coef_ = self.weights_[np.newaxis, :]
        self.intercept_ = np.array([self.bias_])
        self.n_iter_ = int(self.n_iter)
        self.loss_history_ = np.asarray(self.loss_history_, dtype=float)
        self.loss_steps_ = np.asarray(self.loss_steps_, dtype=int)
        self.coef_history_ = np.asarray(self.coef_history_, dtype=float)
        self.coef_history_steps_ = np.asarray(self.coef_history_steps_, dtype=int)
        return self

    def predict_proba(self, X):
        probabilities = self._positive_probability(X)
        return np.column_stack((1.0 - probabilities, probabilities))

    def decision_function(self, X):
        X = self._as_features(X)
        self._check_fitted(X)
        return X @ self.weights_ + self.bias_

    def predict(self, X):
        return (self._positive_probability(X) >= self.threshold).astype(int)

    def _positive_probability(self, X):
        return self._sigmoid(self.decision_function(X))

    @staticmethod
    def _as_features(X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a two-dimensional array")
        return X

    @staticmethod
    def _sigmoid(logits):
        probabilities = np.empty_like(logits, dtype=float)
        positive = logits >= 0
        probabilities[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
        exp_logits = np.exp(logits[~positive])
        probabilities[~positive] = exp_logits / (1.0 + exp_logits)
        return probabilities

    def _regularization_gradient(self):
        if self.reg is None or self.lam == 0:
            return np.zeros_like(self.weights_)
        if self.reg == "l1":
            return self.lam * np.sign(self.weights_)
        if self.reg == "l2":
            return self.lam * self.weights_
        return self.lam * (
            self.l1_ratio * np.sign(self.weights_)
            + (1.0 - self.l1_ratio) * self.weights_
        )

    def _regularization_loss(self):
        if self.reg is None or self.lam == 0:
            return 0.0
        if self.reg == "l1":
            penalty = np.abs(self.weights_).sum()
        elif self.reg == "l2":
            penalty = 0.5 * np.square(self.weights_).sum()
        else:
            penalty = (
                self.l1_ratio * np.abs(self.weights_).sum()
                + 0.5 * (1.0 - self.l1_ratio) * np.square(self.weights_).sum()
            )
        return self.lam * penalty

    def _objective(self, X, y):
        logits = X @ self.weights_ + self.bias_
        log_loss = np.mean(np.logaddexp(0.0, logits) - y * logits)
        return float(log_loss + self._regularization_loss())

    def _parameters(self):
        return np.concatenate(([self.bias_], self.weights_))

    def _check_fitted(self, X):
        if not hasattr(self, "weights_"):
            raise ValueError("model is not fitted; call fit(X, y) first")
        if X.shape[1] != self.weights_.size:
            raise ValueError(
                f"X has {X.shape[1]} features, but the model was fitted with "
                f"{self.weights_.size} features"
            )

    def _validate_params(self):
        if not np.isfinite(self.lr) or self.lr <= 0:
            raise ValueError("lr must be a positive finite number")
        if not isinstance(self.n_iter, (int, np.integer)) or self.n_iter <= 0:
            raise ValueError("n_iter must be a positive integer")
        if not isinstance(self.eval_interval, (int, np.integer)) or self.eval_interval <= 0:
            raise ValueError("eval_interval must be a positive integer")
        if not isinstance(self.history_interval, (int, np.integer)) or self.history_interval <= 0:
            raise ValueError("history_interval must be a positive integer")
        if self.reg not in (None, "l1", "l2", "elasticnet"):
            raise ValueError("reg must be None, 'l1', 'l2', or 'elasticnet'")
        if not np.isfinite(self.lam) or self.lam < 0:
            raise ValueError("lam must be a non-negative finite number")
        if not 0.0 <= self.l1_ratio <= 1.0:
            raise ValueError("l1_ratio must be between 0 and 1")
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1")
