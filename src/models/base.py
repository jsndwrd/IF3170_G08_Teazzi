import pickle


# Shared interface: fit(X, y) on numpy arrays returns self, predict(X) returns 0/1 labels
class BaseModel:
    def fit(self, X, y):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)
