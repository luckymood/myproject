import numpy as np
from tensorflow.keras.datasets import cifar10
from sklearn.preprocessing import StandardScaler
import pickle
from src.ANN import ANN

# 加载CIFAR10数据
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_test = X_test.reshape(X_test.shape[0], -1) / 255.0

# 标准化
scaler = StandardScaler()
X_test = scaler.fit_transform(X_test)

# 加载模型
with open('./models/ann_cifar10.pkl', 'rb') as f:
    model_data = pickle.load(f)
ann = ANN(task_type='classification', hidden_dims=model_data['hidden_dims'], output_dim=model_data['output_dim'])
ann.weights = model_data['weights']
ann.biases = model_data['biases']

# 预测并输出准确率
y_pred = ann.predict(X_test)
acc = np.mean(y_pred == y_test.flatten())
print(f"✅ CIFAR10测试集准确率：{acc:.4f}")