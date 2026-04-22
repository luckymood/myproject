from sklearn.svm import SVC
import joblib
import os

class TitanicSVM:
    def __init__(self):
        self.model = SVC(random_state=42)

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def save_model(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)

    def load_model(self, path):
        self.model = joblib.load(path)