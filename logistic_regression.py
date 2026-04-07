import numpy as np


class LogisticRegression:
    def __init__(
        self,
        epoch: int | None = None,
        lr: float | None = None,
        epochs: int | None = None,
        learning_rate: float | None = None,
        threshold: float = 0.5,
    ):
        # Support both naming styles: (epoch, lr) and (epochs, learning_rate).
        self.epoch = epoch if epoch is not None else epochs
        self.lr = lr if lr is not None else learning_rate
        if self.epoch is None:
            self.epoch = 100
        if self.lr is None:
            self.lr = 0.01

        self.threshold = threshold
        self.w = None
        self.losses = []

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-z))

    def loss(self, y_true: np.ndarray, y_prob: np.ndarray) -> float:
        eps = 1e-12
        y_prob = np.clip(y_prob, eps, 1.0 - eps)
        return -np.mean(y_true * np.log(y_prob) + (1.0 - y_true) * np.log(1.0 - y_prob))

    def fit(self, X: np.ndarray, Y: np.ndarray, batch_size: int | None = None) -> None:
        n_samples, n_features = X.shape
        self.w = np.zeros((n_features, 1), dtype=np.float64)

        for _ in range(self.epoch):
            # Full-batch gradient descent when batch_size is not provided.
            if batch_size is None or batch_size >= n_samples:
                y_prob = self.predict(X)
                grad = (X.T @ (y_prob - Y)) / n_samples
                self.w -= self.lr * grad
                self.losses.append(self.loss(Y, y_prob))
                continue

            # Mini-batch training if batch_size is explicitly provided.
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            Y_shuffled = Y[indices]

            for start in range(0, n_samples, batch_size):
                end = start + batch_size
                xb = X_shuffled[start:end]
                yb = Y_shuffled[start:end]
                y_prob = self.predict(xb)
                grad = (xb.T @ (y_prob - yb)) / xb.shape[0]
                self.w -= self.lr * grad

            self.losses.append(self.loss(Y, self.predict(X)))

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.w is None:
            raise ValueError("Model has not been trained. Call fit() before predict().")
        logits = X @ self.w
        return self.sigmoid(logits)

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        y_true = y_true.reshape(-1)
        y_pred = y_pred.reshape(-1)

        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))

        precision = tp / (tp + fp + 1e-12)
        recall = tp / (tp + fn + 1e-12)
        f1 = 2 * precision * recall / (precision + recall + 1e-12)
        return {"precision": precision, "recall": recall, "f1_score": f1}