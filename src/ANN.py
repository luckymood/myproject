import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

class ANN:
    def __init__(self, task_type='classification', hidden_dims=[64, 32], output_dim=1):
        self.task_type = task_type
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.weights = []
        self.biases = []
        self.loss_history = []
        self.acc_history = []

    def sigmoid(self, x):
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)

    def init_weights(self, input_dim):
        self.weights = []
        self.biases = []
        prev_dim = input_dim
        for dim in self.hidden_dims:
            w = np.random.randn(prev_dim, dim) * np.sqrt(2 / prev_dim)
            b = np.zeros((1, dim))
            self.weights.append(w)
            self.biases.append(b)
            prev_dim = dim
        w_out = np.random.randn(prev_dim, self.output_dim) * np.sqrt(1 / prev_dim)
        b_out = np.zeros((1, self.output_dim))
        self.weights.append(w_out)
        self.biases.append(b_out)

    def forward(self, X):
        activations = [X]
        zs = []
        a = X
        for i in range(len(self.weights) - 1):
            z = np.dot(a, self.weights[i]) + self.biases[i]
            zs.append(z)
            a = self.relu(z)
            activations.append(a)
        z_out = np.dot(a, self.weights[-1]) + self.biases[-1]
        zs.append(z_out)
        if self.task_type == 'classification':
            if self.output_dim == 1:
                a_out = self.sigmoid(z_out)
            else:
                exp_x = np.exp(z_out - np.max(z_out, axis=1, keepdims=True))
                a_out = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        else:
            a_out = z_out
        activations.append(a_out)
        return activations, zs

    def backward(self, activations, zs, y_true, lr):
        n = len(y_true)
        if self.task_type == 'classification':
            if self.output_dim == 1:
                delta = activations[-1] - y_true.reshape(-1, 1)
            else:
                delta = activations[-1] - y_true
        else:
            delta = activations[-1] - y_true.reshape(-1, 1)

        for i in reversed(range(len(self.weights))):
            prev_act = activations[i]
            dw = np.dot(prev_act.T, delta) / n
            db = np.mean(delta, axis=0, keepdims=True)
            self.weights[i] -= lr * dw
            self.biases[i] -= lr * db

            if i > 0:
                delta = np.dot(delta, self.weights[i].T)
                delta *= self.relu_derivative(zs[i-1])

    def train(self, X, y, epochs=100, lr=0.0001, batch_size=32):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        self.init_weights(X.shape[1])
        if self.task_type == 'classification' and self.output_dim > 1:
            y_onehot = np.zeros((len(y), self.output_dim))
            y_onehot[np.arange(len(y)), y.astype(int)] = 1
            y = y_onehot

        for epoch in range(epochs):
            indices = np.random.permutation(len(X))
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            for i in range(0, len(X), batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                activations, zs = self.forward(X_batch)
                self.backward(activations, zs, y_batch, lr)

            activations, _ = self.forward(X)
            y_pred = activations[-1]
            if self.task_type == 'classification':
                if self.output_dim == 1:
                    loss = -np.mean(y * np.log(y_pred + 1e-8) + (1 - y) * np.log(1 - y_pred + 1e-8))
                    acc = np.mean((y_pred > 0.5).astype(int) == y.reshape(-1,1))
                else:
                    loss = -np.mean(np.sum(y * np.log(y_pred + 1e-8), axis=1))
                    acc = np.mean(np.argmax(y_pred, axis=1) == np.argmax(y, axis=1))
                self.acc_history.append(acc)
            else:
                loss = np.mean((y_pred - y.reshape(-1,1))**2)

            self.loss_history.append(loss)
            if epoch % 10 == 0:
                print(f"Epoch {epoch}/{epochs}, Loss: {loss:.6f}")

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        activations, _ = self.forward(X)
        y_pred = activations[-1]
        if self.task_type == 'classification':
            if self.output_dim == 1:
                return (y_pred > 0.5).astype(int).flatten()
            else:
                return np.argmax(y_pred, axis=1)
        else:
            return y_pred.flatten()

    def save_model(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'weights': self.weights,
                'biases': self.biases,
                'task_type': self.task_type,
                'hidden_dims': self.hidden_dims,
                'output_dim': self.output_dim
            }, f)

    def plot_loss_curve(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.figure()
        plt.plot(self.loss_history)
        plt.title('Loss Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig(path)
        plt.close()

    def plot_accuracy_curve(self, path):
        if self.task_type == 'classification':
            os.makedirs(os.path.dirname(path), exist_ok=True)
            plt.figure()
            plt.plot(self.acc_history)
            plt.title('Accuracy Curve')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.savefig(path)
            plt.close()