import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm

class LogisticRegression:
    def __init__(self, epochs: int, learning_rate: float, threshold: float = 0.5) -> None:
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.losses = []
        self.metrics = []
        self.weights = None 
        self.threshold = threshold

    # Sử dụng hàm sigmoid để chuyển đổi giá trị tuyến tính thành xác suất, sử dụng where để đảm bảo nó không chia cho 0
    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        logits = (1 + np.exp(-z))
        logits = np.where(logits == 0, 10e-5, logits)
        return 1 / logits

    # Sử dụng hàm log loss để tính toán mất mát, thêm một hằng số nhỏ vào log để tránh log(0) không xác định
    def compute_loss(self, y: np.ndarray, y_pred: np.ndarray) -> float:
        return -(y * np.log(y_pred + 10e-5) + (1 - y) * np.log(1 - y_pred + 10e-5)).mean()
    
    def accuracy(self, y: np.ndarray, y_pred: np.ndarray) -> float:
        return (1 - np.abs(y - y_pred)).mean()

    def fit(self, X: np.ndarray, y: np.ndarray, batch_size: int) -> None:
        n_samples = X.shape[0]
        n_features = X.shape[1]
        
        # Khởi tạo trọng số ban đầu là 0
        self.weights = np.zeros(n_features)

        # Bắt đầu quá trình huấn luyện
        pbar = tqdm(range(self.epochs), desc="Đang Training")
        for epoch in pbar:
            indices = np.arange(n_samples)
            np.random.shuffle(indices)

            X_shuffled = X[indices]
            y_shuffled = y[indices]
            # Tối ưu hóa bằng mini-batch gradient descent
            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i: i + batch_size]
                y_batch = y_shuffled[i: i + batch_size]
                
                # Kiểm tra kích thước hiện tại của mini-batch
                current_batch_size = X_batch.shape[0]
                
                # Forward
                linear_model = np.matmul(X_batch, self.weights)
                y_pred = self.sigmoid(linear_model)
                
                # Backward
                mini_batch_grad = (1 / current_batch_size) * np.matmul(X_batch.T, (y_pred - y_batch))
                self.weights -= self.learning_rate * mini_batch_grad
                
            # Evaluating
            y_pred_full = self.sigmoid(np.matmul(X, self.weights))
            loss = self.compute_loss(y, y_pred_full)
            acc = self.accuracy(y, (y_pred_full >= self.threshold).astype(int))

            pbar.set_postfix({
                'Loss': f"{loss:.4f}", 
                'Accuracy': f"{acc:.4f}"
            })

            self.losses.append(loss)
            self.metrics.append(acc)

    def predict(self, X: np.ndarray) -> np.ndarray:
        linear_model = np.matmul(X, self.weights)
        y_pred = self.sigmoid(linear_model)
        return (y_pred >= self.threshold).astype(int)
        