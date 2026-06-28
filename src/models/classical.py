import numpy as np
import pickle
import scipy.sparse as sp
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

class LogisticRegressionModel:

    def __init__(self, C: float = 1.0, max_iter: int = 1000, random_state: int = 42):
        self.model = LogisticRegression(
            C=C,
            max_iter=max_iter,
            random_state=random_state,
            n_jobs=-1,
        )

    def fit(self, X: sp.csc_matrix, y: np.ndarray) -> None:
        self.model.fit(X, y)

    def predict(self, X: sp.csr_matrix) -> np.ndarray:
        return self.model.predict(X)
    
    def predict_proba(self, X: sp.csr_matrix) -> np.ndarray:
        return self.model.predict_proba(X)
    
    def save(self, save_path: str) -> None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.model, f)

    def load(self, load_path: str) -> None:
        path = Path(load_path)
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")
        with open(path, "rb") as f:
            self.model = pickle.load(f)


class SVMModel:
    
    def __init__(self, C: float = 1.0, max_iter: int = 1000, random_state: int = 42):
        self.model = LinearSVC(
            C=C,
            max_iter=max_iter,
            random_state=random_state,
        )
 
    def fit(self, X: sp.csr_matrix, y: np.ndarray) -> None:
        self.model.fit(X, y)
 
    def predict(self, X: sp.csr_matrix) -> np.ndarray:
        return self.model.predict(X)
 
    def save(self, save_path: str) -> None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.model, f)
 
    def load(self, load_path: str) -> None:
        path = Path(load_path)
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")
        with open(path, "rb") as f:
            self.model = pickle.load(f)    